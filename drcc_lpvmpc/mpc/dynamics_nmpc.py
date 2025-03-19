from drcc_lpvmpc.vehicle.orca import ORCA
import casadi as ca
import numpy as np

from casadi import DM
import sys
import traceback
import os
import time as tm



class BicycleDynamicsNMPC:
    def __init__(self,path_ptr,center_ptr,obs_map,current_state,dt,horizon,track_with) -> None:
        self.initialize = True
        self.init_vxy = True

        self.option_ = {"max_iter":3000,"tol":1e-9,"linear_solver":"ma27"}

        self.track_width_ = track_with

        script = os.path.dirname(__file__)

        self.output_log = os.path.join(os.path.split(script)[0],'output','DynamicNMPC_solver_output.txt')
        params = ORCA()
        self.n_states = 6 # x, y, phi, vx, vy, omega
        self.n_inputs = 2 # steer, acc
        """	specify model params here
		"""
        self.lf = params['lf']
        self.lr = params['lr']
        self.dr = self.lr/(self.lf+self.lr)
        self.mass = params['mass']
        self.Iz = params['Iz']

        self.Cf = params['Cf']
        self.Cr = params['Cr']

        self.Bf = params['Bf']
        self.Br = params['Br']
        self.Df = params['Df']
        self.Dr = params['Dr']

        self.Caf = params['Caf']
        self.Car = params['Car']

        self.g = params['g'] # gravity acc
        self.d = params['d']
        self.w = params['w']

        self.max_x = 1.79
        self.min_x = -1.12

        self.max_y = 1.65
        self.min_y = -1.85

        self.max_vx = params['max_vx']
        self.min_vx = params['min_vx']

        self.max_vy = params['max_vy']
        self.min_vy = params['min_vy']

        self.max_omega = params['max_omega']
        self.min_omega = params['min_omega']

        self.max_acc = params['max_acc']
        self.min_acc = params['min_acc']

        self.max_steer = params['max_steer']
        self.min_steer = params['min_steer']

        self.max_rate = params['max_rate']
        self.min_rate = params['min_rate']


        self.path_ptr_ = path_ptr
        # self.bound_ptr_ = bound_ptr
        self.center_ptr_ = center_ptr
        self.obs_map = obs_map

        self.obs_side_avoid = self.obs_map.side_avoid

        self.dt = dt

        self.horizon = horizon
        self.s_max = self.path_ptr_.get_max_length()

        # ################################ for a rectangle obs ######################################

        obs_end_ls = self.obs_map.end_ls
        self.obs_cons_atau = []
        self.obs_cons_btau = []

        self.obs_cons_an = []
        self.obs_cons_bn = []


        for _, end_pts in enumerate(obs_end_ls):
            end_pts = np.array(end_pts).transpose()
            # print("end pts:",end_pts)
            ab_tau = self.path_ptr_.xy_to_tau(end_pts)
            print(ab_tau)
            if ab_tau[0,0] < ab_tau[0,1]:
                ind = 0
                indb = 1
            else:
                ind = 1
                indb = 0
            # print("min ab :",ind)
            tau_a = ca.mmin(ab_tau)
            tau_b = ca.mmax(ab_tau)
            self.obs_cons_atau.append(tau_a)
            self.obs_cons_btau.append(tau_b)

            an = self.path_ptr_.f_xy_to_taun(ca.DM(end_pts[:,ind]),tau_a)
            bn = self.path_ptr_.f_xy_to_taun(ca.DM(end_pts[:,indb]),tau_b)

            print("an, bn :",an,bn)

            self.obs_cons_an.append(an)
            self.obs_cons_bn.append(bn)


        self.obs_cons_an = ca.vertcat(*self.obs_cons_an)
        self.obs_cons_bn = ca.vertcat(*self.obs_cons_bn)

        obs_safe_dis = ca.DM.ones(self.obs_cons_an.shape[0],1) * self.max_vx*self.dt*self.horizon*2
        obs_safe_disb = ca.DM.ones(self.obs_cons_bn.shape[0],1) * self.max_vx*self.dt*self.horizon*2

        self.obs_cons_phi_fre = ca.atan2(self.obs_cons_an,obs_safe_dis)
        self.obs_cons_phi_freb = ca.atan2(self.obs_cons_bn,obs_safe_disb)

        # print("self obs cons phi fre:",self.obs_cons_phi_fre)

        # print("self obs cons a:",self.obs_cons_atau)
        self.obs_cons_atau = ca.vertcat(*self.obs_cons_atau)
        self.obs_cons_btau = ca.vertcat(*self.obs_cons_btau)

        # print("self obs cons atau:",self.obs_cons_atau)
        self.obs_cons_sa = self.path_ptr_.tau_to_s_lookup(self.obs_cons_atau)
        self.obs_cons_s0 = self.obs_cons_sa - self.max_vx*self.dt*self.horizon*2

        self.obs_cons_sb = self.path_ptr_.tau_to_s_lookup(self.obs_cons_atau)
        self.obs_cons_s1 = self.obs_cons_sb + self.max_vx*self.dt*self.horizon*2
        # print("obs cons s:",self.obs_cons_s0)
        self.obs_cons_tau0 = self.path_ptr_.s_to_tau_lookup(self.obs_cons_s0)
        self.obs_cons_tau1 = self.path_ptr_.s_to_tau_lookup(self.obs_cons_s1)
        # print("self obs cons tau:",self.obs_cons_tau0)

        ############################### initialize reference state#####################################
        self.ref_xy = []
        self.v_arr = []
        getxy = self.get_ref_xy(current_state)
        self.current_state = current_state # x, y, phi, vx, vy, omega

        # now the ref_pre_x, ref_pre_y, ref_pre_phi are already setup
        # now the reference_x, reference_y, reference_phi are already setup

        # unwrap phi for continuous problem
        self.reference_phi = np.array(self.reference_phi)
        self.reference_phi = np.unwrap(self.reference_phi)
        self.reference_phi = ca.DM(self.reference_phi)


        self.get_ref_xyk()
        self.ref_pre_phi = self.reference_phi

        ############################### initialize weight matrix ###########################################

        self.P = ca.DM.ones(1,self.horizon+1)*2

        self.Q = ca.DM.ones(self.n_inputs,self.horizon) *0.0001# weights of delta, acc

        self.track_weight = ca.linspace(1,10,self.horizon+1).T

        ################################# initialize noise matrix ###########################################

        self.reach_end = False

        self.start = tm.time()
        self.end = tm.time()

        self.tau0 = 0

        self.obs_detect = False




    def get_Updated_local_path(self,current_state,usedro=True,print_=False):

        """
        return optimized_result, optimized_state, optimized_control
        """

        optimized_path = self.make_plan(current_state,print_=print_)
        self.end = tm.time()


        
        if optimized_path[0]:
            print("Solve Success, time: {:.8f}".format(self.end-self.start))

            # op_z = np.array(optimized_path[1][0])
            # op_u = np.array(optimized_path[1][1])

            # op_x = np.array(optimized_path[1][0][0,:]).squeeze().tolist()
            # op_y = np.array(optimized_path[1][0][1,:]).squeeze().tolist()
            # op_phi = np.array(optimized_path[1][0][2,:]).squeeze().tolist()
            # op_vx = np.array(optimized_path[1][0][3,:]).squeeze().tolist()
            # op_vy = np.array(optimized_path[1][0][4,:]).squeeze().tolist()
            # op_omega = np.array(optimized_path[1][0][5,:]).squeeze().tolist()

            # op_delta = np.array(optimized_path[1][1][0,:]).squeeze().tolist()
            # op_acc = np.array(optimized_path[1][1][1,:]).squeeze().tolist()

            return optimized_path[0],optimized_path[1][0],optimized_path[1][1]
        else:
            if self.reach_end:
                print("################################")
            else:
                print("optimization fails")
            return optimized_path[0],DM(),DM()


    def make_plan(self,current_state,print_=False):
        
        # reset the obs index
        self.obs_detect = False
        self.sol_gamma = 0

        casadi_option = {"print_time":print_}

        x0 = ca.DM([current_state[0],current_state[1],
                    current_state[2],current_state[3],
                    current_state[4],current_state[5]]) 

        ############################### get reference path #################################
        getxy = self.get_ref_xy(current_state)
        self.tau0 = self.tau_arr[0,0]
        if not getxy:
            print("Reached end")
            self.reach_end = True
            return False,DM(),DM()
        # now self.reference_xy, self.reference_x, self.reference_y, self.reference_phi is updated

        # unwrap the reference phi for continuous concern
        self.reference_phi = np.array(self.reference_phi)
        self.reference_phi = np.unwrap(self.reference_phi)

        self.reference_phi = ca.DM(self.reference_phi)
        if self.reference_phi[0,0] - self.ref_pre_phi[0,1] >= 2*ca.pi-0.5:
            self.reference_phi -= 2*ca.pi
        elif self.reference_phi[0,0] - self.ref_pre_phi[0,1] <= -2*ca.pi+0.5:
            self.reference_phi += 2*ca.pi


        if getxy is False:
            return False
        self.get_ref_xyk()


        # update previous reference phi, x, y
        self.ref_pre_phi = self.reference_phi
        self.ref_pre_x = self.reference_x
        self.ref_pre_y = self.reference_y

        # self.reference_vx = (xkpre - self.xkpre)/self.dt
        self.reference_vx = self.v_arr[:self.horizon+1,0].T
        # self.reference_vy = (ykpre - self.ykpre)/self.dt
        self.reference_vy = ca.DM.zeros(1,self.horizon+1)


        # update old state
        self.current_state = x0

        ################### calculate boundary function ax + by = c ###########################

        a11, b11, c11, a12, b12, c12, a13, b13, c13, a14, b14, c14 = self.get_corridor_func()

        ########################################### get rectangle tangent function ax + by = c #######################

        # ref_xy = np.array(self.ref_xy).transpose().tolist()
        # print("ref xy :",ref_xy)

        a3 = np.ones((1,self.horizon+1)) # a1,b1,c1,a2,b2,c2 * (horizon+1)
        b3 = np.ones((1,self.horizon+1)) # a1,b1,c1,a2,b2,c2 * (horizon+1)
        c3 = np.ones((1,self.horizon+1)) # a1,b1,c1,a2,b2,c2 * (horizon+1)

        a3 *= 1
        b3 *= 1
        c3 *= -1e10

        for j in range(self.obs_cons_tau0.shape[0]):
            for i in range(self.horizon,-1,-1):
                if self.tau_arr[0,i] <= self.obs_cons_tau0[j]:
                    break
                elif self.tau_arr[0,i] <= self.obs_cons_atau[j]:
                    collid_n = self.obs_cons_an[j] * (self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,i]) - self.obs_cons_s0[j])/(self.obs_cons_sa[j]-self.obs_cons_s0[j])
                    shift_n = collid_n + self.obs_side_avoid[j]*0.1
                    s_a3,s_b3,s_c3 = self.get_safe_line_equation(self.tau_arr[0,i],collid_n,shift_n,self.obs_cons_phi_fre[j])
                    self.update_safe_reference_track(i,collid_n,self.obs_cons_phi_fre[j])

                elif self.tau_arr[0,i] <= self.obs_cons_btau[j]:
                    collid_n = self.obs_cons_bn[j]
                    shift_n = collid_n + self.obs_side_avoid[j]*0.1
                    s_a3,s_b3,s_c3 = self.get_safe_line_equation(self.tau_arr[0,i],collid_n,shift_n,0)
                    self.update_safe_reference_track(i,collid_n,0)

                
                elif self.tau_arr[0,i] >= self.obs_cons_btau[j] and self.tau_arr[0,i] <= self.obs_cons_tau1[j]:
                    collid_n = self.obs_cons_bn[j] * (-self.path_ptr_.tau_to_s_lookup(self.tau_arr[0,i]) + self.obs_cons_s1[j])/(self.obs_cons_s1[j]-self.obs_cons_sb[j])
                    shift_n = collid_n + self.obs_side_avoid[j]*0.1
                    s_a3,s_b3,s_c3 = self.get_safe_line_equation(self.tau_arr[0,i],collid_n,shift_n,-self.obs_cons_phi_freb[j])
                    self.update_safe_reference_track(i,collid_n,self.obs_cons_phi_freb[j])

                else:
                    continue


                a3[0,i] = s_a3
                b3[0,i] = s_b3
                c3[0,i] = s_c3

            if self.tau_arr[0,1] >= self.obs_cons_atau[j] and self.tau_arr[0,1] <= self.obs_cons_btau[j]:
                self.obs_detect = j
        ##################################### define objective function ################################

        opti = ca.Opti()

        z = opti.variable(self.n_states,self.horizon+1)
        u = opti.variable(self.n_inputs,self.horizon)
        dxdtc = opti.variable(self.n_states, 1)


        ############################ update objective function ############################

        ref_state = np.array(ca.vertcat(self.reference_x,self.reference_y,self.reference_phi,
                    self.reference_vx,self.reference_vy,self.reference_omega))
        
        self.ref_xy = ca.DM(ref_state[:3,:])

        ref_state = ca.DM(ref_state)

        for idh in range(self.horizon):
            dxdt = self.dynamic_model(z[:,idh], u[:,idh], dxdtc)
            opti.subject_to(z[:,idh+1] - z[:,idh] - self.dt * dxdt == 0)

        for i in range(self.horizon):
            opti.subject_to(a11[i+1,0]*z[0,i+1]+b11[i+1,0]*z[1,i+1]-c11[i+1] <= 0)
            opti.subject_to(a12[i+1,0]*z[0,i+1]+b12[i+1,0]*z[1,i+1]-c12[i+1] <= 0)
            opti.subject_to(a13[i+1,0]*z[0,i+1]+b13[i+1,0]*z[1,i+1]-c13[i+1] <= 0)
            opti.subject_to(a14[i+1,0]*z[0,i+1]+b14[i+1,0]*z[1,i+1]-c14[i+1] <= 0)
            # opti.subject_to(-a3[0,i+1]*z[0,i+1]-b3[0,i+1]*z[1,i+1]+c3[0,i+1] <= 0)


            

        # ######################################## specify constraints #########################################
        A = ca.DM.zeros(self.n_states, self.n_states+self.n_inputs*self.horizon)
        b = ca.DM.zeros(self.n_states,1)
        for i in range(self.n_states):
            A[i,i] = 1
            b[i,0] = x0[i,0]

        opti.subject_to(z[:,0] == x0)

        ####################################### control boundary ########################################

        opti.subject_to(opti.bounded(self.min_acc, u[1,:], self.max_acc))
        opti.subject_to(opti.bounded(self.min_steer, u[0,:] , self.max_steer))

        ###################################### state boundary ############################################

        opti.subject_to(opti.bounded(self.min_x,z[0,:],self.max_x))
        opti.subject_to(opti.bounded(self.min_y,z[1,:],self.max_y))

        opti.subject_to(opti.bounded(0,z[3,:],self.max_vx))
        # opti.subject_to(opti.bounded(self.min_vy,z[4,:],self.max_vy))
        # opti.subject_to(opti.bounded(self.min_omega,z[5,:],self.max_omega))

        ###################################### minimize the objective function ##########################################

        xn_obj = 0
        for i in range(self.horizon+1):
            xn_obj += self.P[0,i]*ca.dot(z[0,i]-ref_state[0,i],z[0,i]-ref_state[0,i])+ca.dot(z[1,i]-ref_state[1,i],z[1,i]-ref_state[1,i])

        for i in range(self.horizon):
            xn_obj += self.Q[0,i]*ca.dot(u[0,i],u[0,i])



        opti.minimize(xn_obj)
        
        #########################################################################################################################################

        opti.solver("ipopt",casadi_option,self.option_)
        # opti.solver("qpoases",casadi_option)

        output_to_file = True
        self.start = tm.time()

        if output_to_file:

            with open(self.output_log,'w') as output_file:
                stdout_old = sys.stdout
                sys.stdout = output_file

                try:
                    sol = opti.solve()

                    sol_z = sol.value(z)
                    sol_u = sol.value(u)

                    print("sol z:",sol_z)
                    print("sol u:",sol_u)

                    return tuple((True,tuple((sol_z,sol_u))))
                except:

                    print("OPTIMIZED SOLVER FAILED")
                    traceback.print_exc(file=stdout_old)
                    print("Solve optimal problem fails")
                    return tuple((False,tuple((DM(),DM()))))
                finally:
                    sys.stdout = stdout_old
        else:
            try:
                sol = opti.solve()

                sol_z = sol.value(z)
                sol_u = sol.value(u)

                print("sol z:",sol_z)
                print("sol u:",sol_u)

                return tuple((True,tuple((sol_z,sol_u))))
            except:
                debug_z = opti.debug.value(z)
                print("debug z:",debug_z)
                debug_u = opti.debug.value(u)
                print("debug u :",debug_u)

                print("OPTIMIZED SOLVER FAILED")
                traceback.print_exc(file=stdout_old)
                print("Solve optimal problem fails")
                return tuple((False,tuple((DM(),DM()))))
    

    def get_ref_xy(self,current_state):
        """
        update self.ref_xy based on given current state
        """
        s0 = 0.0
        st = 0.0

        x0 = ca.DM([current_state[0],current_state[1],
                    current_state[2],current_state[3],
                    current_state[4],current_state[5]]) 
        # x, y, phi, vx, vy, omega
        tau0 = self.path_ptr_.xy_to_tau(x0[:2])
        # print("tau 0 :",tau0)
        if tau0<0.00001:
            tau0 = 0.1

        s0 = self.path_ptr_.tau_to_s_lookup(tau0)

        v0 = x0[3]

        if self.initialize:
            s0 -= v0*self.dt

        if v0 + self.dt*self.max_acc*self.horizon <= self.max_vx:
            self.v_arr = ca.linspace(v0,v0+self.dt*self.max_acc*self.horizon,self.horizon+1)
        else:
            self.v_arr = ca.linspace(v0,v0+self.dt*self.max_acc*self.horizon,self.horizon+1)
            self.v_arr = ca.fmin(self.v_arr,self.max_vx)
        ds_arr = self.v_arr*self.dt

        self.s_arr = ca.DM.zeros(1,self.horizon+1)
        self.s_arr[0,0] = s0
        for i in range(self.horizon):
            self.s_arr[0,i+1] = self.s_arr[0,i] + ds_arr[i]

        st = self.s_arr[0,-1]
        
        # print("st :",st)
        if st>=self.s_max:

            return False
            # print("smax :",self.s_max)
            # st=self.s_max
            # self.horizon = int((st-s0)/(self.max_vx*self.dt))
            # self.s_arr = self.s_arr[0,:self.horizon+1]
            # if self.horizon<=1:
                # return False
        # self.s_arr = ca.linspace(s0,st,self.horizon+1)
        # print("self sarr :",self.s_arr)

        self.tau_arr = self.path_ptr_.s_to_tau_lookup(self.s_arr)

        if self.initialize:
            s00 = s0 - v0*self.dt
            s00_arr = self.s_arr
            s00_arr[0,1:] = s00_arr[0,:-1]
            s00_arr[0,0] = s00
            tau_arr = self.path_ptr_.s_to_tau_lookup(s00_arr)
            n_arr = ca.DM.zeros(1,self.horizon+1)
            ref_pre_xy = self.path_ptr_.f_taun_to_xy(tau_arr,n_arr)
            self.ref_pre_x = ref_pre_xy[0,:]
            self.ref_pre_y = ref_pre_xy[1,:]
            self.ref_pre_phi = self.path_ptr_.f_phi(tau_arr)
            self.initialize = False


        n_arr = ca.DM.zeros(1,self.horizon+1)
        # define reference xy
        # print("tau arr:",self.tau_arr)
        # print("tau max is:",self.path_ptr_.tau_max)
        self.ref_xy = self.path_ptr_.f_taun_to_xy(self.tau_arr,n_arr)
        self.reference_phi = self.path_ptr_.f_phi(self.tau_arr)
        self.reference_x = self.ref_xy[0,:]
        self.reference_y = self.ref_xy[1,:]

        return True

    
    def get_abc12(self,up_a,up_b,up_c,low_a,low_b,low_c):
        lhs = (np.array(up_a*self.ref_xy[0,:]+up_b*self.ref_xy[1,:]-up_c)<=0).astype(int)

        a10 = up_a*lhs
        b10 = up_b*lhs
        c10 = up_c*lhs
        a20 = low_a*lhs
        b20 = low_b*lhs
        c20 = low_c*lhs

        lhs = (np.array(up_a*self.ref_xy[0,:]+up_b*self.ref_xy[1,:]-up_c)>0).astype(int)

        a11 = low_a*lhs
        b11 = low_b*lhs
        c11 = low_c*lhs
        a21 = up_a*lhs
        b21 = up_b*lhs
        c21 = up_c*lhs

        a1 = a10 + a11
        b1 = b10 + b11
        c1 = c10 + c11

        a2 = a20 + a21
        b2 = b20 + b21
        c2 = c20 + c21

        return a1,b1,c1,a2,b2,c2
    
    def get_ref_xyk(self):
        """
        calculate reference states x_ref, y_ref, phi_ref, vx_ref, vy_ref, omega_ref, alphaf_ref, alphar_ref
        """
        self.reference_omega = (self.reference_phi-self.ref_pre_phi[0,:self.horizon+1])/self.dt
    
    def get_reference_path(self):
        return self.ref_xy
    
    def get_reference_phi(self):
        return self.reference_phi
    
    def get_obs_atau(self):
        return self.obs_cons_atau
    
    def get_obs_an(self):
        return self.obs_cons_an
    
    def get_obs_btau(self):
        return self.obs_cons_btau
    
    def get_obs_bn(self):
        return self.obs_cons_bn
    
    def get_obs_tau0(self):
        return self.obs_cons_tau0
    
    def get_obs_tau1(self):
        return self.obs_cons_tau1
    
    def casadi_unwrap(self,op_phi):
        
        op_phi_np = np.unwrap(np.array(op_phi))

        # print("phi diff:",op_phi_np)

        return op_phi_np
    
    def get_safe_line_equation(self,tau0,n0,n1,phi_ref):

        # print("phi ref:",phi_ref)

        # print("tau 0 :",tau0)
        # print("n 0 :",n0)
        # print("n 1 :",n1)
        xy1 = self.path_ptr_.f_taun_to_xy(tau0,n0)
        xy2 = self.path_ptr_.f_taun_to_xy(tau0,n1)

        phi = self.path_ptr_.f_phi(tau0)

        phi_cart = phi + phi_ref
        # print("phi cart :",phi_cart)
        x1 = xy1[0,:]
        y1 = xy1[1,:]
        x2 = xy2[0,:]
        y2 = xy2[1,:]
        # Calculate coefficients
        a = ca.sin(phi_cart)
        b = -ca.cos(phi_cart)
        c = ca.sin(phi_cart)*x1 - ca.cos(phi_cart)*y1

        # the line will show as ax + by - c = 0

        test_func_val = a * x2 + b * y2 - c

        test_func_val = np.array(test_func_val)
        # print("test function value:",test_func_val)

        negative_index = np.where(test_func_val < 0)[1]
        # print("negative value :",negative_index)
        
        if negative_index.shape[0] != 0: 
            a[negative_index] = -1*a[negative_index]
            b[negative_index] = -1*b[negative_index]
            c[negative_index] = -1*c[negative_index]

        return a, b, c
    
    def update_safe_reference_track(self, current_idx, new_n, phi_ref):
        current_tau = self.tau_arr[0,current_idx]
        new_xy = self.path_ptr_.f_taun_to_xy(current_tau,new_n)
        # print("new xy :",new_xy)
        self.reference_x[0,current_idx] = new_xy[0,0]
        self.reference_y[0,current_idx] = new_xy[1,0]
        self.reference_phi[0,current_idx] += phi_ref
    
    def get_corridor_func(self):
        taus = self.center_ptr_.xy_to_tau(self.ref_xy)

        ns = ca.DM.zeros(1,taus.columns())
        xy = self.center_ptr_.f_taun_to_xy(taus,ns)

        x0 = xy[0,:] - self.track_width_
        x1 = xy[0,:] + self.track_width_

        y0 = xy[1,:] - self.track_width_
        y1 = xy[1,:] + self.track_width_

        # all should satisfies ax + by - c <= 0

        a11 = -1 * ca.DM.ones(1,x0.shape[1])
        b11 = ca.DM.zeros(1,x0.shape[1])
        c11 = -x0

        a12 = ca.DM.ones(1,x0.shape[1])
        b12 = ca.DM.zeros(1,x0.shape[1])
        c12 = x1

        a13 = ca.DM.zeros(1,x0.shape[1])
        b13 = -1 * ca.DM.ones(1,x0.shape[1])
        c13 = -y0

        a14 = ca.DM.zeros(1,x0.shape[1])
        b14 = ca.DM.ones(1,x0.shape[1])
        c14 = y1

        return a11.T, b11.T, c11.T, a12.T, b12.T, c12.T, a13.T, b13.T, c13.T, a14.T, b14.T, c14.T
    
    def get_old_p_param(self):
        return self.past_pvx,self.past_pvy,self.past_pphi,self.past_pdelta
    
    def update_model_noise(self,noise):
        self.noise_arr = self.model_noise.update_noise_matrix(noise)

    def update_new_p_param(self,new_pvx, new_pvy, new_pphi):
        self.p_vx = ca.DM(new_pvx).T
        self.p_vy = ca.DM(new_pvy).T
        self.p_phi = ca.DM(new_pphi).T
        # self.p_delta = ca.DM(new_pdelta).T

        # print(self.p_vx,self.p_vy,self.p_phi,self.p_delta)

    def save_fixed_noise(self):
        self.model_noise.save_noise_arr()

    def get_tau0_value(self):
        return self.tau0
    
    def get_obs_detect(self):
        return self.obs_detect

    def dynamic_model(self, x, u, dxdt):
        """	write dynamics as first order ODE: dxdt = f(x(t))
            x is a 6x1 vector: [x, y, psi, vx, vy, omega]^T
            u is a 2x1 vector: [acc/pwm, steer]^T
            dxdt is a casadi.SX variable
        """
        steer = u[0]
        acc = u[1]
        psi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        n = x[1]

        vmin = 0.05
        vy = ca.if_else(vx<vmin, 0, vy)
        omega = ca.if_else(vx<vmin, 0, omega)
        steer = ca.if_else(vx<vmin, 0, steer)
        vx = ca.if_else(vx<vmin, vmin, vx)

        Frx = acc*self.mass
        alphaf = steer - ca.atan2((self.lf*omega + vy), vx)
        alphar = ca.atan2((self.lr*omega - vy), vx)
        Ffy = self.Df * ca.sin(self.Cf * ca.arctan(self.Bf * alphaf))
        Fry = self.Dr * ca.sin(self.Cr * ca.arctan(self.Br * alphar))

        dxdt[0] = vx*ca.cos(psi) - vy*ca.sin(psi)
        dxdt[1] = vx*ca.sin(psi) + vy*ca.cos(psi)
        dxdt[2] = omega
        dxdt[3] = 1/self.mass * (Frx - Ffy*ca.sin(steer)) + vy*omega
        dxdt[4] = 1/self.mass * (Fry + Ffy*ca.cos(steer)) - vx*omega
        dxdt[5] = 1/self.Iz * (Ffy*self.lf*ca.cos(steer) - Fry*self.lr)
        return dxdt