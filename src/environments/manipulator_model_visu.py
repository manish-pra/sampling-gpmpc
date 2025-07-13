from urdf2casadi import urdfparser as u2c

parser = u2c.URDFparser()
# urdf_path = "/home/manish/work/MPC_Dyn/sampling-gpmpc/src/environments/urdf/franka/franka.urdf"
urdf_path = "/home/manish/work/MPC_Dyn/sampling-gpmpc/src/environments/urdf/2dof_robot.urdf"
parser.from_file(urdf_path)
# fk = parser.get_forward_kinematics("panda_link0", "panda_ee")["T_fk"]
# print(fk([0.0, 0.0]))
fk = parser.get_forward_kinematics("base_link", "ee_link")["T_fk"]
print(fk([0.1, 0.2]))

# import casadi as ca

# # Define symbolic variables
# theta = ca.SX.sym("theta", 2)
# theta_dot = ca.SX.sym("theta_dot", 2)
# tau = ca.SX.sym("tau", 2)

# theta_ddot_expr = parser.get_forward_dynamics_aba("base_link", "ee_link")(theta, theta_dot, tau)
# # Wrap in CasADi function
# forward_dyn = ca.Function("forward_dynamics", [theta, theta_dot, tau], [theta_ddot_expr])

# theta_ddot = forward_dyn(theta, theta_dot, tau)  # CasADi symbolic




# # Evaluate with test inputs
# print(forward_dyn([0.1, 0.2], [0, 0], [0.0, 0.0]))  # → returns q_ddot



import pybullet as p
import pybullet_data
import time
import numpy as np
import os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
# robot_id = p.loadURDF("dobot/dobot.urdf", basePosition=[0, 0, 0], useFixedBase=True)

p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=00,
    cameraPitch=-45,
    cameraTargetPosition=[0.2, 0, 0.2]
)

# -------------------------------
# Get revolute joint indices
# -------------------------------
joint_indices = [
    i for i in range(p.getNumJoints(robot_id))
    if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
]
print("Revolute joints:", joint_indices)

# -------------------------------
# Disable default motor control
# -------------------------------
for j in joint_indices:
    p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

# -------------------------------
# Animate joints using sin(t)
# -------------------------------
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

print("Animating robot...")
t0 = time.time()
while p.isConnected():
    t = time.time() - t0
    q = [0.5 * np.sin(t), -0.4 * np.cos(0.7*t), 0.6 * np.sin(1.5*t),0.5 * np.sin(t), -0.4 * np.cos(0.7*t), 0.6 * np.sin(1.5*t), 0.6 * np.sin(1.5*t)]
    # q = [np.sin(t), np.cos(t)]
    for j, qj in zip(joint_indices, q):
        p.setJointMotorControl2(
            robot_id,
            j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=qj,
            force=5.0,
        )
    p.stepSimulation()
    time.sleep(1/240.0)


# p.connect(p.GUI)
# robot_id = p.loadURDF("one_dof_robot.urdf", useFixedBase=True)

# # Disable default motor
# p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, force=0)

# # Move it
# t0 = time.time()
# while p.isConnected():
#     t = time.time() - t0
#     p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=0.5 * p.sin(t), force=5)
#     p.stepSimulation()
#     time.sleep(1/240.)