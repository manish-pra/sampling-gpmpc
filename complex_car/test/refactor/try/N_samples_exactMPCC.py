import numpy as np, time, casadi as ca
import CarDynamics 
import plot
import MPCC
np.random.seed(42)


# This function runs exact MPCC with N function samples that have the same exact dynamics (This is for sanity check, the number of samples shouldn't change the result mathematically
# but there could be very little numerical differences)
# If there is only one function sample, the code works for H = 10, 20, 40, 80, but when num_samples is more than 1 and horizon length is 20/40/80, often Error_in_Step_Computation is raised??
def run_simulation_exact():
    Df = 0.65
    Dr = 1.00
    dyn_true = CarDynamics.CarDynamicsComplex(Df=Df, Dr=Dr, dt = 0.06)
    num_samples = 2
    horizon = 10
    lambda_slack = 10
    mpc = MPCC.MPCC(dyn_true, num_samples, weight_samples=None, horizon=horizon, lambda_slack=lambda_slack)      
    x = np.array([0.0, 1.95, 0.0, 3.0, 0.0, 0.0])
    theta = x[0]

    hist_x, hist_u, hist_v, t_solve = [x.copy()], [], [], []
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

        x = dyn_true.step(x, u, Df, Dr, np)
        theta += v_prog*dyn_true.dt

        hist_x.append(x.copy())
        hist_u.append(u.copy())
        hist_v.append(v_prog)
        warm = (Xopt,Uopt,Vopt,Thopt,Sopt)
        if theta >= 71.0: break
    plot.plot_trajectory(hist_x, hist_u, hist_v, t_solve, dyn_true, mpc)
    


if __name__ == '__main__':
    run_simulation_exact()

