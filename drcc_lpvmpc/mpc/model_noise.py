import numpy as np
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

class Model_noise:
    def __init__(self,horizon,fix_noise) -> None:
        self.n_states = 6
        self.n_inputs = 2
        self.sampled_data = 200

        self.history_size = 5

        self.horizon_ = horizon
        self.dir_name = os.path.dirname(__file__)
        # print("dir name is:",dir_name)

        n_file_name = "model_error.csv"
        z_state_file = "z_states.csv"
        fixnoise_state_file = "noise_arr.npy"


        n_file_name = os.path.join(self.dir_name,"data",n_file_name)
        z_state_file = os.path.join(self.dir_name,"data",z_state_file)

        self.fix_noise_data = os.path.join(self.dir_name,"data",fixnoise_state_file)

        # print("file path :",file_name)

        self.init_noise = np.genfromtxt(n_file_name,delimiter=',') #this helps generate noise from file, rows as x, y, phi, vx, vy, omega
        self.z_his_states = np.genfromtxt(z_state_file,delimiter=',') #this helps generate noise from file, rows as x, y, phi, vx, vy, omega

        self.num_his = self.init_noise.shape[1]
        # print(self.init_noise.shape)
        # print(self.z_his_states.shape)

        self.ep_states = np.zeros((self.num_his,self.n_states,self.horizon_,))
        self.z_states = np.zeros((self.num_his,self.n_states+self.horizon_*self.n_inputs,))
        for i in range(self.num_his):
            # print("noise arr: ",self.init_noise[:,i])
            # print("ep states shape :",self.ep_states[i].shape)
            self.ep_states[i] = self.init_noise[:,i].reshape((self.n_states,self.horizon_),order='F')
            # self.z_states[i] = self.z_his_states[:,i]

        # print("noise shape:",self.init_noise.shape)

        for i in range(self.n_states):
            self.x = self.ep_states[:,0,:].flatten()
            self.y = self.ep_states[:,1,:].flatten()
            self.phi = self.ep_states[:,2,:].flatten()
            self.vx = self.ep_states[:,3,:].flatten()
            self.vy = self.ep_states[:,4,:].flatten()
            self.omega = self.ep_states[:,5,:].flatten()

        self.get_noise_distribution()

        if fix_noise:
            self.selected_noise = np.load(self.fix_noise_data)


        # random_rows_indices = np.random.choice(self.num_his,2,replace=False)

        # self.selected_noise = self.ep_states[random_rows_indices]
        # print("ep state :",self.ep_states)
        # self.selected_z = self.z_states[random_rows_indices]
        # print("selected noise :",self.selected_noise)

    def update_noise_matrix(self,noise):
        # noise_ = noise.reshape(-1,order='F')
        # noise_ = noise_.transpose()

        # self.selected_noise[:-1] = self.selected_noise[1:]
        # self.selected_noise[-1] = noise_.reshape((self.n_states,self.horizon_),order='F')
        self.get_noise_distribution(noise)

        return self.selected_noise
    
    def random_update_noise_matrix(self):
        random_rows_indices = np.random.choice(self.num_his,self.horizon_,replace=False)
        self.selected_noise = self.ep_states[random_rows_indices]
        self.selected_z = self.z_states[random_rows_indices]

    def get_noise_distribution(self,new_noise=None):
        if new_noise is not None:
            noise_ = new_noise
            noise_ += np.random.uniform(-0.01,0.01,(6,self.horizon_))
            updated_noise = np.reshape(noise_,(self.n_states*self.horizon_,1),order='F')
            self.selected_noise[-1] = updated_noise

            # print("update noise")
            return


        d_x = gaussian_kde(self.x, bw_method='scott')
        d_y = gaussian_kde(self.y, bw_method='scott')
        d_phi = gaussian_kde(self.phi, bw_method='scott')
        d_vx = gaussian_kde(self.vx, bw_method='scott')
        d_vy = gaussian_kde(self.vy, bw_method='scott')
        d_omega = gaussian_kde(self.omega, bw_method='scott')

        self.x = d_x.resample(self.sampled_data)[0]
        self.y = d_y.resample(self.sampled_data)[0]
        self.phi = d_phi.resample(self.sampled_data)[0]
        self.vx = d_vx.resample(self.sampled_data)[0]
        self.vy = d_vy.resample(self.sampled_data)[0]
        self.omega = d_omega.resample(self.sampled_data)[0]

        # self.noise_dataset = np.stack((self.x,self.y,self.phi,self.vx,self.vy,self.omega))
        # print("noise dataset shape:",self.noise_dataset.shape)

        # plt.hist(self.x, bins=20, density= True, alpha=0.5, label="x")
        # plt.hist(self.y, bins=20, density= True, alpha=0.5, label="y")
        # plt.hist(self.phi, bins=20, density= True, alpha=0.5, label="phi")
        # plt.hist(self.vx, bins=20, density= True, alpha=0.5, label="vx")
        # plt.hist(self.vy, bins=20, density= True, alpha=0.5, label="vy")
        # plt.hist(self.omega, bins=20, density= True, alpha=0.5, label="omega")
        # plt.legend()
        # plt.show()

        self.selected_noise = np.zeros((self.history_size,self.n_states*self.horizon_,1))
        bound_states_x = np.partition(self.x,-self.horizon_*self.history_size)
        bound_states_y = np.partition(self.y,-self.horizon_*self.history_size)
        bound_states_phi = np.partition(self.phi,-self.horizon_*self.history_size)
        bound_states_vx = np.partition(self.vx,-self.horizon_*self.history_size)
        bound_states_vy = np.partition(self.vy,-self.horizon_*self.history_size)
        bound_states_omega = np.partition(self.omega,-self.horizon_*self.history_size)


        for i in range(self.history_size):
            for j in range(self.horizon_-1,-1,-1):
                random_state = np.zeros((self.n_states))
                # random_state[0] = np.random.choice(self.x)
                # random_state[1] = np.random.choice(self.y)
                # random_state[2] = np.random.choice(self.phi)
                # random_state[3] = np.random.choice(self.vx)
                # random_state[4] = np.random.choice(self.vy)
                # random_state[5] = np.random.choice(self.omega)
                random_state[0] = bound_states_x[-j*self.history_size] 
                random_state[1] = bound_states_y[-j*self.history_size] 
                random_state[2] = bound_states_phi[-j*self.history_size] 
                random_state[3] = bound_states_vx[-j*self.history_size]
                random_state[4] = bound_states_vy[-j*self.history_size]
                random_state[5] = bound_states_omega[-j*self.history_size]

                random_state += np.random.uniform(-0.01,0.01)

                self.selected_noise[i,j*self.n_states:(j+1)*self.n_states,0] = random_state

        # print("noise arr :",self.selected_noise)

    def get_noise_arr(self):
        return self.selected_noise
    
    def save_noise_arr(self):
        np.save(self.fix_noise_data,self.selected_noise)

if __name__=='__main__':
    horizon = 6
    model_noise = Model_noise(horizon)
    # new_noise = np.array([0,0,0,0,0,0])
    # model_noise.random_update_noise_matrix()
    # print("updated_noise :",model_noise.get_noise_arr())


    # z_states, m_noise = model_noise.get_noise_arr()
    # print(z_states[0].shape)
    # print(m_noise[0].shape)
    
