import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import CarDynamics
def plot_trajectory(hist_x, hist_u, hist_v, t_solve, dyn_true, mpc):
    hist_x = np.array(hist_x)
    hist_u = np.array(hist_u) 
    hist_v = np.array(hist_v)
    t = np.arange(len(hist_x))*dyn_true.dt

    plt.figure(figsize=(15,12))

    plt.subplot(331)
    # plot position
    plt.plot(hist_x[:,0], hist_x[:,1], 'b')
    SAFE_DIST, ELL_X = 5.67, 3.0

    for ox,oy in mpc.obstacles:
        w = 2*ELL_X*np.sqrt(SAFE_DIST); h = 2*np.sqrt(SAFE_DIST)
        plt.gca().add_patch(patches.Ellipse((ox,oy), w,h, alpha=0.3,color='red'))

    plt.axvline(70,color='r',ls='--'); plt.axhline(CarDynamics.StraightTrack.center_y,color='g',ls='--')
    plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.title('Position (x,y)')
    plt.axis('equal'); plt.grid()

    # psi 
    plt.subplot(332); plt.plot(t, hist_x[:,2]); plt.ylabel('ψ [rad]'); plt.title('Heading'); plt.grid()
    # vx
    plt.subplot(333); plt.plot(t, hist_x[:,3]); plt.ylabel('vx [m/s]'); plt.title('Long. speed'); plt.grid()
    # vy
    plt.subplot(334); plt.plot(t, hist_x[:,4]); plt.ylabel('vy [m/s]'); plt.title('Lat. speed'); plt.grid()
    # omega 
    plt.subplot(335); plt.plot(t, hist_x[:,5]); plt.ylabel('ω [rad/s]'); plt.title('Yaw rate'); plt.grid()

    # controls + progress
    if len(hist_u):
        plt.subplot(336); plt.plot(t[:-1], hist_u[:,0]); plt.ylabel('δ [rad]'); plt.title('Steering'); plt.grid()
        plt.subplot(337); plt.plot(t[:-1], hist_u[:,1]); plt.ylabel('τ [Nm]');  plt.title('Torque');   plt.grid()
        plt.subplot(338); plt.plot(t[:-1], hist_v);      plt.ylabel('vθ [m/s]');plt.title('Progress'); plt.grid()

    # solver time
    plt.subplot(339); plt.plot(t_solve); plt.ylabel('ms'); plt.title('Solve time'); plt.grid()

    plt.tight_layout(); plt.show() 
