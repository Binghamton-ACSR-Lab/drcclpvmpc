import shapely
import matplotlib.pyplot as plt
import casadi as ca
import numpy as np
from casadi import Slice

def plot_path(path_ptr,type,labels=None,linew = 1.0):
    """ type = 1: plot racecar path
        type = 2: plot virtual running path
        type = 3: plot optimized path"""
    
    if type==1:
        ptr_=path_ptr
        tau_span = ca.linspace(0,ptr_.tau_max,2000)
        n_span = ca.DM.zeros(tau_span.rows(),tau_span.columns())
        ptr_xy = ptr_.f_taun_to_xy(tau_span.T,n_span.T)

        ptr_x = np.array(ptr_xy[0,Slice()]).reshape((-1))
        ptr_y = np.array(ptr_xy[1,Slice()]).reshape((-1))

        plt.plot(ptr_x,ptr_y,label=labels, color="green",linewidth=linew,linestyle='-.',zorder=0)
        plt.legend()

    elif type==2:
        virtual_path = path_ptr
        # print(path_arr)
        path = []
        for iter in virtual_path:
            
            path.append(np.array(iter).squeeze().tolist())
        path = np.array(path).transpose().tolist()

        plt.plot(path[0],path[1],label=labels)
        plt.legend()

    elif type==3:
        optimized_path = path_ptr
        optimized_px = np.array(optimized_path[0,:]).squeeze().tolist()
        optimized_py = np.array(optimized_path[1,:]).squeeze().tolist()

        plt.plot(optimized_px,optimized_py,label=labels)
        plt.legend()

def ax_plot(path_ptr,type,ax,labels=None):
    """ type = 1: plot racecar path
        type = 2: plot virtual running path
        type = 3: plot optimized path"""
    
    if type==1:
        ptr_=path_ptr
        tau_span = ca.linspace(0,ptr_.tau_max,2000)
        n_span = ca.DM.zeros(tau_span.rows(),tau_span.columns())
        ptr_xy = ptr_.f_taun_to_xy(tau_span.T,n_span.T)

        ptr_x = np.array(ptr_xy[0,Slice()]).reshape((-1))
        ptr_y = np.array(ptr_xy[1,Slice()]).reshape((-1))

        ax.plot(ptr_x,ptr_y,label=labels, color="red",linewidth=0.5)
        ax.legend()

    elif type==2:
        virtual_path = path_ptr
        # print(path_arr)
        path = []
        for iter in virtual_path:
            
            path.append(np.array(iter).squeeze().tolist())
        path = np.array(path).transpose().tolist()

        ax.plot(path[0],path[1],label=labels)
        ax.legend()

    elif type==3:
        optimized_path = path_ptr
        optimized_px = np.array(optimized_path[0,:]).squeeze().tolist()
        optimized_py = np.array(optimized_path[1,:]).squeeze().tolist()

        ax.plot(optimized_px,optimized_py,label=labels)
        ax.legend()


class OperationWindow:
    def __init__(self,horizon,batch) -> None:
        self.horizon_=horizon
        self.batch_ = batch
        self.operate_window = ca.DM.zeros(batch+horizon-1,horizon)
        self.operate_window[0:horizon,:] = ca.DM(np.identity(horizon))
        # print("operate windows is:",operate_window)
        for i in range(1,batch):
            step_window = ca.DM.zeros(batch+horizon-1,horizon)
            step_window[i:i+horizon,:] = ca.DM(np.identity(horizon))
            self.operate_window = ca.horzcat(self.operate_window,step_window)
        # print("operation window is:",self.operate_window)

        constrain_idx = self.get_Kinematics_constrain_list()
        self.last_idx = constrain_idx[1]
        self.next_idx = constrain_idx[0]
        



    def get_Kinematics_batch_data(self,state_data):
        if state_data.rows()==2:
            state_batchs = ca.mtimes(state_data,self.operate_window)
            state_batchs= ca.reshape(state_batchs,(2,-1))
            # print("state_batchs is:",ca.reshape(state_batchs[0,:],(-1,self.batch_)).T)
            state_batch = ca.vertcat(ca.reshape(state_batchs[0,:],(-1,self.batch_)).T,ca.reshape(state_batchs[1,:],(-1,self.batch_)).T)
        else:
            state_batchs = ca.mtimes(state_data,self.operate_window)
            # print("state_batchs is:",ca.reshape(state_batchs[0,:],(-1,self.batch_)).T)
            state_batch = ca.reshape(state_batchs[0,:],(-1,self.batch_)).T
        # print("state batch is:",state_batch)
        
        return state_batch
    
    def get_Kinematics_constrain_list(self):
        indx = np.array(range(0,3*self.batch_))
        
        initial_idx = np.array(range(0,3*self.batch_,self.batch_)).tolist()
        
        mask = np.ones(len(indx), dtype=bool)
        mask[initial_idx] = False

        result_inx = indx[mask,...]

        # print("result inx is:",result_inx)
        return tuple((result_inx,result_inx-1))
    

class BicycleDynamicsOperationWindow:
    def __init__(self,horizon,batch) -> None:
        self.horizon_=horizon
        self.batch_ = batch
        self.operate_window = ca.DM.zeros(batch+horizon-1,horizon)
        self.operate_window[0:horizon,:] = ca.DM(np.identity(horizon))
        # print("operate windows is:",operate_window)
        for i in range(1,batch):
            step_window = ca.DM.zeros(batch+horizon-1,horizon)
            step_window[i:i+horizon,:] = ca.DM(np.identity(horizon))
            self.operate_window = ca.horzcat(self.operate_window,step_window)
        # print("operation window is:",self.operate_window)

        constrain_idx = self.get_Dynamics_constrain_list()
        self.last_idx = constrain_idx[1]
        self.next_idx = constrain_idx[0]
        



    def get_Dynamics_batch_data(self,state_data):
        if state_data.rows()==2:
            state_batchs = ca.mtimes(state_data,self.operate_window)
            state_batchs= ca.reshape(state_batchs,(2,-1))
            # print("state_batchs is:",ca.reshape(state_batchs[0,:],(-1,self.batch_)).T)
            state_batch = ca.vertcat(ca.reshape(state_batchs[0,:],(-1,self.batch_)).T,ca.reshape(state_batchs[1,:],(-1,self.batch_)).T)
        else:
            state_batchs = ca.mtimes(state_data,self.operate_window)
            # print("state_batchs is:",ca.reshape(state_batchs[0,:],(-1,self.batch_)).T)
            state_batch = ca.reshape(state_batchs[0,:],(-1,self.batch_)).T
        # print("state batch is:",state_batch)
        
        return state_batch
    
    def get_Dynamics_constrain_list(self):
        indx = np.array(range(0,5*self.batch_))
        
        initial_idx = np.array(range(0,5*self.batch_,self.batch_)).tolist()
        
        mask = np.ones(len(indx), dtype=bool)
        mask[initial_idx] = False

        result_inx = indx[mask,...]

        # print("result inx is:",result_inx)
        return tuple((result_inx,result_inx-1))
    
def plotDirection(path_ptr,type,labels=None):
    """Currently only support type 2:virtual running path"""
    virtual_path = path_ptr
        # print(path_arr)
    path = []
    for iter in virtual_path:
        
        path.append(np.array(iter).squeeze().tolist())
    path = np.array(path).transpose().tolist()
    

    plt.quiver(path[0],path[1],np.cos(path[2]),np.sin(path[2]),label=labels)
    plt.legend()

def plotDynamicsVy(path_ptr,type,labels=None):
    """Currently only support type 2:virtual running path"""
    virtual_path = path_ptr
    virtual_path = np.array(virtual_path).transpose().tolist()
        # print(path_arr)
    # path = []
    # for iter in virtual_path:
        
    #     path.append(np.array(iter).squeeze().tolist())
    # path = np.array(path).transpose().tolist()

    step = range(len(virtual_path[3]))

    plt.plot(step,virtual_path[3],label=labels)
    plt.legend()

# def plotDynamicsAlpha_f(path_ptr,control_arr,type,labels=None):
#     """Currently only support type 2:virtual running path"""
#     dynamics_model = convert_dynamics_params(use_unity=True)
#     virtual_path = path_ptr
#     control_arr_ = control_arr
#     # print("virtual path:",virtual_path)
#     path = []
#     for iter in virtual_path:
#         path.append(np.array(iter).squeeze().tolist())
#     path = np.array(path).transpose()
#     step = range(len(path[3]))
#     alpha_f = (path[3,:-1]+dynamics_model.Lf*path[4,:-1])/control_arr_[1,:]-control_arr_[0,:]

#     alpha_f = np.array(alpha_f).squeeze().tolist()
#     plt.plot(step[:-1],alpha_f,label=labels)
#     plt.legend()

        
        

