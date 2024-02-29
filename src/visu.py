
import torch
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, params,path):
        self.params = params
        # self.agent = agent
        

    def pendulum_discrete_dyn(self, X1_k, X2_k, U_k):
        """_summary_

        Args:
            x (_type_): _description_
            u (_type_): _description_
        """
        m =1
        l=1
        g=10
        dt = self.params["optimizer"]["dt"]
        X1_kp1 = X1_k + X2_k*dt 
        X2_kp1 = X2_k - g*np.sin(X1_k)*dt/l + U_k*dt/(l*l)
        return X1_kp1, X2_kp1
    
    def propagate_true_dynamics(self, x_init, U):
        x1_list = []
        x2_list = []
        X1_k = x_init[0]
        X2_k = x_init[1]   
        x1_list.append(X1_k.item())
        x2_list.append(X2_k.item())
        for ele in range(U.shape[0]):
            X1_kp1, X2_kp1 = self.pendulum_discrete_dyn(X1_k, X2_k, U[ele])
            x1_list.append(X1_kp1.item())
            x2_list.append(X2_kp1.item())
            X1_k = X1_kp1.copy()
            X2_k = X2_kp1.copy()
        return x1_list, x2_list
    
    def propagate_mean_dyn(self, x_init, U):
        x1_list = []
        x2_list = []
        X1_k = x_init[0]
        X2_k = x_init[1]   
        x1_list.append(X1_k.item())
        x2_list.append(X2_k.item())
        for ele in range(U.shape[0]):
            y1 = self.Dyn_gp_model['y1'](torch.Tensor([[X1_k, X2_k, U[ele]]])).mean.detach()[0]
            y2 = self.Dyn_gp_model['y2'](torch.Tensor([[X1_k, X2_k, U[ele]]])).mean.detach()[0]
            X1_kp1, X2_kp1 = y1[0], y2[0]
            x1_list.append(X1_kp1.item())
            x2_list.append(X2_kp1.item())
            X1_k = X1_kp1.clone()
            X2_k = X2_kp1.clone()
        return x1_list, x2_list

    def plot_pendulum_traj(self, X, U):
        plt.close()
        plt.plot(X[:,0::2],X[:,1::2])#, label = [i for i in range(self.params["agent"]["num_dyn_samples"])])
        # plt.legend([i for i in range(self.params["agent"]["num_dyn_samples"])])
        plt.xlabel('theta')
        plt.ylabel('theta_dot')
        x1_true, x2_true = self.propagate_true_dynamics(X[0,0:2], U)
        plt.plot(x1_true, x2_true, color='black', label='true', linestyle='--')
        x1_mean, x2_mean = self.propagate_mean_dyn(X[0,0:2], U)
        print("x1_mean", x1_mean, x2_mean)
        plt.plot(x1_mean, x2_mean, color='black', label='mean', linestyle='-.')
        plt.legend()
        plt.grid()
        plt.savefig('pendulum.png')


        
