import numpy as np, time, casadi as ca 
import CarDynamics
import plot
import BLR
import MPCC
np.random.seed(42)


# This function runs offline learning MPCC with N function samples drawn from the posterior distribution of the params Df and Dr, works quite well for horizon length 10
# but breaks for horizon length 20 due to Error_in_Step_Computation ?  
def run_simulation_offline_learning():
    Df, Dr = 0.65, 1.00
    dyn_true = CarDynamics.CarDynamicsComplex(Df=Df, Dr=Dr, dt = 0.06)   
    blr = BLR.BLR(dyn_true, num_trajectories=1, trajectory_length=10, noise_var=1e-2)
    feature_matrix, final_labels = blr.generate_training_data() # generate training data
    _ ,_ = blr.train_gp(feature_matrix, final_labels,  prior_var=1.0) # this calculates the posterior mean and posterior covariance 
    num_samples = 5
    D_samp = blr.sample_weights(num_samples) 
    print(f"sampled functions: {D_samp}")
    horizon = 10
    lambda_slack = 10
    mpc = MPCC.MPCC(dyn_true, num_samples, D_samp, horizon, lambda_slack)      
    x = np.array([0.0, 1.95, 0.0, 3.0, 0.0, 0.0])
    theta = x[0]
    # histogram of true trajectory
    hist_x, hist_u, hist_v, t_solve = [x.copy()], [], [], []
    sol = None
    warm = None
    print(f"{'k':>4} {'x':>6} {'y':>6} {'ψ':>7} {'vx':>6} {'vy':>6} {'ω':>6} "
      f"{'δ':>6} {'τ':>6} {'vθ':>6}   ms")
    for k in range(1000):
        tic = time.time()
        sol = mpc.solve(x, theta, warm)
        if isinstance(sol, tuple) and sol[0] is None:
            print("IPOPT failed.")
            break
        Xopt, Uopt, Vopt, Thopt, Sopt = sol
        t_solve.append((time.time()-tic)*1e3)

        u = Uopt[:,0]
        v_prog = float(Vopt[0]) if Vopt.ndim==1 else float(Vopt[0,0])
        print(f"{k:4d} {x[0]:6.2f} {x[1]:6.2f} {x[2]:7.2f} {x[3]:6.2f} "
      f"{x[4]:6.2f} {x[5]:6.2f} {u[0]:6.2f} {u[1]:6.2f} "
      f"{v_prog:6.2f} {t_solve[-1]:5.1f}")

        x_next_true = dyn_true.step(x, u, Df, Dr, np)
        theta += v_prog*dyn_true.dt

        hist_x.append(x.copy())
        hist_u.append(u.copy())
        hist_v.append(v_prog)
        x = x_next_true.copy()

        warm = (Xopt,Uopt,Vopt,Thopt,Sopt)
        if theta >= 71.0: break
    plot.plot_trajectory(hist_x, hist_u, hist_v, t_solve, dyn_true, mpc) 


if __name__ == '__main__':
    run_simulation_offline_learning()

