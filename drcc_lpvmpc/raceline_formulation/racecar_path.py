import casadi as ca
from casadi import DM
from casadi import Slice
from casadi import MX
import numpy as np
from drcc_lpvmpc.raceline_formulation.shapely_time_tree import RTree


class RacecarPath():
    def __init__(self, pts, direction, upperx=None,uppery=None,lowerx=None,lowery=None) -> None:
        
        # print("Initialize Path\nDirection: ", direction)
        self.resolution = 100

        self.waypoints_ = DM(pts)
        # print(self.waypoints_.shape)
        self.upperx = upperx
        self.uppery = uppery
        self.lowerx = lowerx
        self.lowery = lowery


        # self.stage = stage

        # self.total_stage = total_stage
        # self.safe_range = safe_range

        # self.global_map_.get_buffer_map(self.stage,total_stage=self.total_stage,safe_range=self.safe_range)


        # self.end_phi = ca.atan2((self.waypoints_[-5,0]-self.waypoints_[-1,0]),(self.waypoints_[-5,1]-self.waypoints_[-1,1]))
        # print("end phi is:",self.end_phi)

        # self.edge_margin_ = edge_margin
        # print(self.waypoints_.T)
        if(self.waypoints_.columns() != 2):
            assert(pts.rows()==2)
            self.waypoints_ = pts.T
        else:
            self.waypoints_ = pts
        if not (direction.is_empty()):
            # print("direction is:",direction)
            self.direction_ = ca.reshape(direction,2,1)
            self.direction_ = self.direction_/ca.norm_2(self.direction_)
        print("success initialization")

        # self.p1_nature = DM([1])
        # self.p2_nature = DM([1])

        spline_params = self.natural_spline()

        self.p1_nature = spline_params[0]
        self.p2_nature = spline_params[1]

        self.initialized = False

        if(direction.is_empty()):
            self.f_xy = self.parametric_function(p1=self.p1_nature,p2=self.p2_nature)
            self.p1_ = self.p1_nature
            self.p2_ = self.p2_nature
            self.initialized = True
        else:
            beta = 100.0
            while(beta<20000):
                return_value = self.direct_spline(direct=self.direction_,beta=beta)
                if(return_value[0]):
                    p1_ = return_value[1][0]
                    p2_ = return_value[1][0]
                    self.f_xy = self.parametric_function(p1=p1_,p2=p2_)
                    dp1 = p1_-self.p1_nature
                    print("dp1 is:",dp1)
                   
                    dp2 = p2_-self.p2_nature
                    print("dp2 is:",dp2)
                    obj2 = (ca.dot(dp1[0,:],dp1[0,:])+
                            ca.dot(dp1[1,:],dp1[1,:])+
                            ca.dot(dp2[0,:],dp2[0,:])+
                            ca.dot(dp2[1,:],dp2[1,:]))
                    
                    print("Obj: ",obj2)
                    if(obj2<1e2):
                        self.initialized=True
                        break

                beta*=2

            if not self.initialized:
                print("parametric function construction fails")
                return
            
        print("parametric function is constructed")

        # print(f_xy(1.0))

        self.tau_max = self.waypoints_.rows()-1.000001

        taus = ca.linspace(0, self.tau_max, int(self.resolution * self.tau_max))
        taus_vec = np.array(taus).reshape((-1,))

        tau = ca.MX.sym("tau")
        n = ca.MX.sym("n")
        s = ca.MX.sym("s")
        pt_t_mx = self.f_xy(tau)

        # print("pt_t_mx is:",pt_t_mx)

        # first derivative, jacobian of the function
        jac = ca.jacobian(pt_t_mx,tau)
        # print("jacobian is:",jac)
        # second derivative, hessian matrix

        hes= ca.jacobian(jac,tau)
        # print("hessian is: ",hes.shape)

        self.f_tangent_vec = ca.Function("vec",[tau],[jac])

        

        kappa = (jac[0]*hes[1]-jac[1]*hes[0])/ca.power(ca.norm_2(jac),3)

        self.f_kappa = ca.Function("kappa",[tau],[kappa])
        self.f_phi = ca.Function("phi",[tau],[ca.arctan2(jac[1],jac[0])])
        # test_tau = [10,20,30,40]
        # test_tangent = self.f_phi(test_tau)
        # print("test tangent vector is: ",test_tangent)

        theta = np.pi/2
        rot_mat_vector = np.array(((np.cos(theta),-np.sin(theta)),(np.sin(theta),np.cos(theta)))).tolist()

        rot_mat = DM(rot_mat_vector)

        dae = {"x":s,"t":tau,"ode":ca.norm_2(self.f_tangent_vec(tau))}


        # integ = ca.integrator("inte","cvodes",dae,{"grid":taus_vec})
        # integ = ca.integrator("inte","cvodes",dae,0,taus_vec[1:])
        # integ = ca.integrator("inte","cvodes",dae,{'grid':taus_vec[1:]})
        integ = ca.integrator("inte","cvodes",dae,taus_vec[0],taus_vec[1:])




        

        # args = {"x0":0}

        s_inte = integ(x0 = 0)

        s_value = s_inte["xf"]

        s_value_vec = np.array(s_value).reshape((-1,))

        # s_value_vec.insert(0,0)
        # s_value_vec = np.insert(s_value_vec,0,0)

        self.s_max = s_value_vec[-1]
        print("maximum length is:",self.s_max)

        # print("s value vec :",s_value_vec)

        self.s_to_tau_lookup = ca.interpolant("s_to_t","linear",[s_value_vec],taus_vec[:-1])

        self.tau_to_s_lookup = ca.interpolant("t_to_s","linear",[taus_vec[:-1]],s_value_vec)

        # tau_w = ca.linspace(0,self.waypoints_.rows()-1,self.waypoints_.rows())

        # tau_w_vec = np.array(tau_w).reshape((-1,)).tolist()

        # self.f_tau_to_w = ca.interpolant("tau_to_w","linear",[tau_w_vec],self.edge_margin_)

        xy = pt_t_mx + ca.mtimes(rot_mat,jac/ca.norm_2(jac))*n

        self.f_taun_to_xy = ca.Function("tn_to_xy", [tau,n],[xy])


        phi = self.f_phi(tau)

        dm_xy = ca.MX.sym("dm_xy",2,1)

        dm_n = -ca.sin(phi)*(dm_xy-pt_t_mx)[0]+ca.cos(phi)*(dm_xy-pt_t_mx)[1]

        self.f_xy_to_taun = ca.Function("xy_to_taun",[dm_xy,tau],[dm_n])

        ns_zeros = ca.DM.zeros(taus.rows(),taus.columns())
        center_line = self.f_taun_to_xy(taus.T,ns_zeros.T)

        dm_x = center_line[0,Slice()]
        dm_y = center_line[1,Slice()]

        vec_x = np.array(dm_x.T).reshape((-1,)).tolist()
        vec_y = np.array(dm_y.T).reshape((-1,)).tolist()
        vec_taus = np.array(taus.T).reshape((-1,)).tolist()

        # leave part for search tree
        self.search_tree = RTree(vec_x,vec_y,vec_taus)


        # below is used to get the margin data
        if self.upperx is not None:
            upperxy = ca.DM.zeros(2,self.upperx.shape[0])
            upperxy[0,:] = self.upperx
            upperxy[1,:] = self.uppery

            lowerxy = ca.DM.zeros(2,self.lowerx.shape[0])
            lowerxy[0,:] = self.lowerx
            lowerxy[1,:] = self.lowery
            # print("upper xy :",upperxy)
            # upper_tau_start = self.xy_to_tau(upperxy[:,0])
            # upper_tau_end = self.xy_to_tau(upperxy[:,-1])
            # upper_tau = ca.linspace(upper_tau_start,upper_tau_end,self.upperx.shape[0])

            lower_tau_start = self.xy_to_tau(lowerxy[:,0])
            lower_tau_end = self.xy_to_tau(lowerxy[:,-1])

            # print(lower_tau_end)
            lower_tau = ca.linspace(lower_tau_start,lower_tau_end,self.lowerx.shape[0])

            upper_margin_pts_x=self.upperx
            upper_margin_pts_y=self.uppery

            lower_margin_pts_x=self.lowerx
            lower_margin_pts_y=self.lowery

            low_tau_input_vec = np.array(lower_tau).reshape((-1,)).tolist()
            # low_tau_input_vec = np.array(lower_tau).reshape((-1,)).tolist()

            self.f_tau_to_up_ptx = ca.interpolant("tau_to_upper_ptsx","linear",[low_tau_input_vec],upper_margin_pts_x)
            self.f_tau_to_up_pty = ca.interpolant("tau_to_upper_ptsy","linear",[low_tau_input_vec],upper_margin_pts_y)

            self.f_tau_to_low_ptx = ca.interpolant("tau_to_lower_ptsx","linear",[low_tau_input_vec],lower_margin_pts_x)
            self.f_tau_to_low_pty = ca.interpolant("tau_to_lower_ptsy","linear",[low_tau_input_vec],lower_margin_pts_y)

        # print("margin is:",margin_)


    def natural_spline(self):
        dm_waypoints = self.waypoints_
        if dm_waypoints.rows() != 2:
            dm_waypoints = dm_waypoints.T
        assert(dm_waypoints.rows()==2)

        n = dm_waypoints.columns()-1

        opti = ca.Opti()
        P1 = opti.variable(2,n)
        P2 = opti.variable(2,n)

        #To satisfy the C2, start point second diffirentiate should be zero 
        opti.subject_to(dm_waypoints[Slice(),0]-2*P1[Slice(),0]+P2[Slice(),0]==0)
        # end point's second differentiate should be zero
        opti.subject_to(P1[Slice(),n-1]-2*P2[Slice(),n-1]+dm_waypoints[Slice(),n]==0)

        for i in range(1,int(n)):
            opti.subject_to(P1[Slice(),i]+P2[Slice(),i-1]==2*dm_waypoints[Slice(),i])
            opti.subject_to(P1[Slice(),i-1]+2*P1[Slice(),i]==2*P2[Slice(),i-1]+P2[Slice(),i])

        casadi_option = {"print_time":0}
        ipopt_option = {"print_level":0,'sb':'yes'}

        opti.solver("ipopt",casadi_option,ipopt_option)

        # print("Calculating nature spline params\n")
        # modified
        sol = opti.solve()
        dm_p1 = sol.value(P1)
        dm_p2 = sol.value(P2)

        print("Get nature spline params")

        return tuple((dm_p1,dm_p2))
    
    def parametric_function(self,p1,p2):
        dm_waypoints = self.waypoints_
        if(dm_waypoints.rows()!=2):
            dm_waypoints = dm_waypoints.T
        
        assert(dm_waypoints.rows() == 2)

        n = dm_waypoints.columns()

        mx_waypoints = MX(dm_waypoints)

        mx_p1 = MX(p1)
        mx_p2 = MX(p2)
        
        t = MX.sym("t")
        # remainder
        # tau = MX.fmod(t,n)
        tau = ca.fmod(t,n)
        # round down to nearest int
        i = MX.floor(tau)
        a = mx_waypoints[Slice(),i]
        b = mx_p1[Slice(),i]
        c = mx_p2[Slice(),i]
        # i1 = MX.fmod(i + 1, n)
        i1 = ca.fmod(i + 1, n)

        d = mx_waypoints[Slice(),i1]

        g = (ca.power(1 - (tau - i),3) * a + 3*ca.power(1 - (tau - i), 2) * (tau -i) * b + 
             3 * (1 - (tau - i)) * ca.power(tau - i, 2) * c + ca.power(tau - i, 3) * d)

        return ca.Function("f_xy",[t],[g])
    
    def direct_spline(self,direct,beta):

        # print("Constructing Directional Spline With Beta = ",beta)
        dm_waypoints = self.waypoints_
        if(dm_waypoints.rows()!=2):
            dm_waypoints = dm_waypoints.T
        
        assert(dm_waypoints.rows()==2)

        n = dm_waypoints.columns()-1


        opti = ca.Opti()
        P1 = opti.variable(2,n)
        P2 = opti.variable(2,n)

        dm_direct = ca.reshape(direct,2,1)
        dm_direct = dm_direct/ca.norm_2(dm_direct)

        opti.subject_to((P1[Slice(),0]-dm_waypoints[Slice(),0])/ca.norm_2(P1[Slice(),0]-dm_waypoints[Slice(),0])==dm_direct)
        opti.subject_to(P1[Slice(),n-1]-2*P2[Slice(),n-1]+dm_waypoints[Slice(),n]==0)

        for i in range(1,int(n)):
            opti.subject_to(P1[Slice(),i]+P2[Slice(),i-1]==2*dm_waypoints[Slice(),i])
            opti.subject_to(P1[Slice(),i-1]+2*P1[Slice(),i]==2*P2[Slice(),i-1]+P2[Slice(),i])

        dp1 = P1-self.p1_nature
        dp2 = P2-self.p2_nature
        obj2 = (ca.dot(dp1[0,Slice()],dp1[0,Slice()])+ca.dot(dp1[1,Slice()],dp1[1,Slice()])+
                ca.dot(dp2[0,Slice()],dp2[0,Slice()])+ca.dot(dp2[1,Slice()],dp2[1,Slice()]))
        
        opti.minimize(ca.dot(dm_waypoints[Slice(),0]-2*P1[Slice(),0]+P2[Slice(),0],
                             dm_waypoints[Slice(),0]-2*P1[Slice(),0]+P2[Slice(),0])+beta*obj2)

        opti.set_initial(P1,self.p1_nature)
        opti.set_initial(P2,self.p2_nature)

        casadi_option = {"print_time":0}
        ipopt_option = {"print_level":0,'sb':'yes'}

        opti.solver("ipopt",casadi_option,ipopt_option)

        # print("Calculating directional spline params")

        try:
            sol = opti.solve()
            # print("Get directional spline params")
            dm_p1 = sol.value(P1)
            dm_p2 = sol.value(P2)
            return tuple((True,tuple((dm_p1,dm_p2))))
        
        except Exception as e:
            print("Get Directional spline fails")
            print(e)
            return tuple((False,tuple((DM(),DM()))))
        

    def xy_to_tau(self, xy, refined=True):
        # print("xy is : ",xy)
        # print("xy is vector: ",xy.is_vector())
        dm_x = DM()
        dm_y = DM()
        xy_ = DM(xy)
        if(xy_.is_vector()):
            dm_x = xy_[0]
            dm_y = xy_[1]
        else:
            dm_x = xy_[0,Slice()]
            dm_y = xy_[1,Slice()]

        # print("dm x and y is :",dm_x)
        # print(dm_y)

        vec_x = np.array(dm_x.T).reshape((-1,)).tolist()
        vec_y = np.array(dm_y.T).reshape((-1,)).tolist()

        # print("vector x and y is: ",vec_x)
        # print(vec_y)

        taus_vec = self.search_tree.findNearest(vec_x=vec_x,vec_y=vec_y)

        taus_vec = np.maximum(taus_vec,0.0001)


        if taus_vec[0]==self.tau_max:
            return DM(self.tau_max)

        # print("tau max is",self.tau_max)
        # print("taus_vec vector is : ",taus_vec)

        if not refined:
            return DM(taus_vec)
        
        tau_real = DM.zeros(1,len(vec_x))
        # print("tau real shape:",tau_real.shape)
        for i in range(len(vec_x)):
            tau = ca.MX.sym("tau")
            tan_vec = self.f_tangent_vec(tau)
            xy_on_line = self.f_xy(tau)

            norm_vec = xy_[Slice(),i] - xy_on_line
            dot_prod = norm_vec[0] * tan_vec[0] + norm_vec[1] * tan_vec[1]
            f = ca.Function("f", [tau], [dot_prod])
            rf = ca.rootfinder("rf", "newton", f,{'error_on_fail':False})
            # print("dot prod is :",f(DM(taus_vec[i])))
            # print("xy on line is:",self.f_xy(taus_vec[i]),xy_[Slice(),i])
            tau_real_i = rf(DM(taus_vec[i]))
            tau_real_i = ca.fmin(ca.fmax(tau_real_i,0.0001), self.tau_max)

            tau_real[0,i] = tau_real_i[0]
            # print("tau real 0 :",tau_real)

        # print("tau real is :",tau_real)
        
        return tau_real
        
    def get_path_params(self):
        return tuple((self.p1_,self.p2_))
        
    def get_max_length(self):
        return self.s_max
    
    def get_max_tau(self):
        return self.tau_max
    
    def get_end_point_phi(self):
        return self.f_phi(self.tau_max-1.00001)[0]
    

    def bound_linear_func(self,tau):
        upperx = self.f_tau_to_up_ptx(tau)
        uppery = self.f_tau_to_up_pty(tau)
        lowerx = self.f_tau_to_low_ptx(tau)
        lowery = self.f_tau_to_low_pty(tau)

        upperxy = ca.vertcat(upperx,uppery)
        lowerxy = ca.vertcat(lowerx,lowery)

        return upperxy,lowerxy
    
    def xy_to_up_bound_func(self,ref_xy,width):
        taus = self.xy_to_tau(ref_xy)

        ns = ca.DM.ones(1,taus.columns())*width
        xy = self.f_taun_to_xy(taus,ns)
        phis = self.f_phi(taus)
        a = ca.tan(phis)
        b = -1
        c = -xy[1,:] + a * xy[0,:]
        slid = np.array((np.abs(np.tan(np.array(phis)))<10).astype(int)).tolist()
        slid = ca.DM(slid)

        a = a*slid
        b = b*slid
        c = c*slid

        a2 = 1
        b2 = 0
        c2 = xy[0,:]

        slid = np.array((np.abs(np.tan(np.array(phis)))>=10).astype(int)).tolist()
        slid = ca.DM(slid)

        a2 *= slid
        b2 *= slid
        c2 *= slid

        a += a2
        b += b2
        c += c2

        # for i in range(phis.columns()):
        #     if ca.fabs(ca.tan(phis))[i]<10:
        #         a = ca.tan(phis)[i]
        #         b = -1
        #         c = -xy[1,:] + a * xy[0,:]
        #     else:
        #         a = 1
        #         b = 0
        #         c = xy[0,:]

        return a,b,c
    
    def xy_to_low_bound_func(self,ref_xy):
        taus = self.xy_to_tau(ref_xy)

        ns = ca.DM.zeros(1,taus.columns())
        xy = self.f_taun_to_xy(taus,ns)
        phis = self.f_phi(taus)
        a = ca.tan(phis)
        b = -1
        c = -xy[1,:] + a * xy[0,:]
        slid = np.array((np.abs(np.tan(np.array(phis)))<10).astype(int)).tolist()
        slid = ca.DM(slid)

        a = a*slid
        b = b*slid
        c = c*slid

        a2 = 1
        b2 = 0
        c2 = xy[0,:]

        slid = np.array((np.abs(np.tan(np.array(phis)))>=10).astype(int)).tolist()
        slid = ca.DM(slid)

        a2 *= slid
        b2 *= slid
        c2 *= slid

        a += a2
        b += b2
        c += c2

        # for i in range(phis.columns()):
        #     if ca.fabs(ca.tan(phis))[i]<10:
        #         a = ca.tan(phis)[i]
        #         b = -1
        #         c = -xy[1,:] + a * xy[0,:]
        #     else:
        #         a = 1
        #         b = 0
        #         c = xy[0,:]

        return a,b,c






