from scipy.linalg import cholesky
import numpy as np, time, casadi as ca
import CarDynamics 
np.random.seed(42)

# This class is almost the same as the class in the BLR.py file, only difference being that deterministic inputs and initial states are used here
class BLR:
    def __init__(self, model, num_trajectories = 10, trajectory_length = 1000, noise_var = 1e-2):
        self.model = model
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length 
        self.noise_var = noise_var

    def generate_training_data(self):
        dt = self.model.dt
        Df, Dr = self.model.Df, self.model.Dr
        features_matrix = []                
        final_labels = []               

        for _ in range(self.num_trajectories):
            x = np.array([
                0.0,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0
            ]) 
            for _ in range(self.trajectory_length):
                u = np.array([0.00,4])
                x_next = self.model.step(x, u, Df, Dr, np)
                features = self.model.features(x, u, np)
                f_known = self.model.f_known(x, u, np)
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

        np.savetxt("y_full.txt", y, fmt="%.8e")   
        print("saved y (labels) to text file")
        np.savetxt("Phi_full.txt", Phi, fmt="%.8e")   
        print("saved Phi to text file")
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

    



def test_drawn_samples():
    dyn_true = CarDynamics.CarDynamicsComplex(dt=1, Df=0.65, Dr=1.00)   
    blr = BLR(dyn_true, num_trajectories=1, trajectory_length=10, noise_var=1e-2)
    feature_matrix, final_labels = blr.generate_training_data() # generate training data
    _ ,_ = blr.train_gp(feature_matrix, final_labels,  prior_var=1.0) # this calculates the posterior mean and posterior covariance 
    D_samp = blr.sample_weights(num_samples= 5 ) # sample pairs (Df,Dr)
    print("=" * 80)
    print(f" five weight pairs: {D_samp}")
if __name__ == "__main__":
    test_drawn_samples()