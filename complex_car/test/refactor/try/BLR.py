import numpy as np
from scipy.linalg import cholesky
# class responsible for generating, training and sampling 
class BLR:
    def __init__(self, model, num_trajectories = 10, trajectory_length = 1000, noise_var = 1e-2):
        self.model = model
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length 
        self.noise_var = noise_var
        self.backend = np 
    def generate_training_data(self):
        dt = self.model.dt
        Df, Dr = self.model.Df, self.model.Dr
        features_matrix = []                
        final_labels = []               

        for _ in range(self.num_trajectories):
            x = np.array([
                np.random.uniform(0, 70),        
                np.random.uniform(0.0, 6),       
                np.random.uniform(-1.2, 1.2),  
                np.random.uniform(1.5,  8.0), 
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(-2.0, 2.0)
            ]) 
            for _ in range(self.trajectory_length):
                u = np.array([
                    np.random.uniform(-0.45, 0.45),     
                    np.random.uniform(-0.02,   0.02)     
                ]) 
                x_next = self.model.step(x, u, Df, Dr, self.backend)
                features = self.model.features(x, u, self.backend)
                f_known = self.model.f_known(x, u, self.backend)
                current_labels = (x_next - x) / dt - f_known + np.array([0.0, 0.0, 0.0, np.random.normal(0, np.sqrt(self.noise_var)), np.random.normal(0, np.sqrt(self.noise_var)), np.random.normal(0, np.sqrt(self.noise_var))])
                print(x_next)
                features_matrix.extend(features[3:])
                final_labels.extend(current_labels[3:])
                x = x_next.copy()

        features_matrix = np.vstack(features_matrix)          
        final_labels = np.array(final_labels)            
        return features_matrix, final_labels
    
    def train_gp(self, Phi: np.ndarray,
                y:   np.ndarray,
                prior_var: float = 1.0):
        prior_mat= (1.0 / prior_var) * np.eye(2)       
        prec = prior_mat + (1.0 / self.noise_var) * (Phi.T @ Phi)
        self.weights_posterior_cov = np.linalg.inv(prec)                            
        self.weights_posterior_mean = self.weights_posterior_cov @ ((1.0 / self.noise_var) * Phi.T @ y)            
        eigvals = np.linalg.eigvalsh(self.weights_posterior_cov)

        print("=" * 80)
        print(f"prec: {prec}")
        print("=" * 80)
        print(f"posterior cov matrix: {self.weights_posterior_cov}")
        print("=" * 80)
        print(f"posterior mean: {self.weights_posterior_mean}")
        print("=" * 80)
        print(f"posterior eigenvalues:  {eigvals.min():.3e}  …  {eigvals.max():.3e}")
        return self.weights_posterior_mean, self.weights_posterior_cov

    def sample_weights(self, num_samples):

        weight_samples = []
        
        L = cholesky(self.weights_posterior_cov, lower=True)
        for _ in range(num_samples):
            z = np.random.normal(0, 1, 2)
            w_sample = self.weights_posterior_mean + L @ z
            weight_samples.append(w_sample)
        obj = np.asarray(weight_samples, dtype=object)      
        np.save("weights.npy", obj)      
        print("saved weights to file") 
        return np.asarray(weight_samples)

