#!/usr/bin/env python3
"""
fk_and_visualise.py
-------------------

• Loads a URDF with urdf2casadi for FK computation.
• Opens PyBullet to visualise the same robot.
• Press the SPACE bar to toggle between two joint configurations:
  – all-zeros (home)
  – the sample q you used in the FK call.
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from urdf2casadi import urdfparser as u2c


# -------------------------------------------------------------------
# 1.  URDF-to-CasADi: compute FK of a sample configuration
# -------------------------------------------------------------------
urdf_path = "/home/manish/work/MPC_Dyn/sampling-gpmpc/src/environments/urdf/planar_robot.urdf"
root_link, end_link = "base_link", "tool0"

robot_parser = u2c.URDFparser()
robot_parser.from_file(urdf_path)

fk_dict = robot_parser.get_forward_kinematics(root_link, end_link)
forward_kinematics = fk_dict["T_fk"]

q_sample = np.array([0.3, 0.3, 0.3, 0.0, 0.3, 0.7])
T_sample = forward_kinematics(q_sample)
print("FK pose of tool0 for q =", q_sample)
print(np.array(T_sample))

# -------------------------------------------------------------------
# 2.  PyBullet visualisation
# -------------------------------------------------------------------
def set_joint_positions(body_id, joint_angles):
    """Helper: feed a vector of joint angles to Bullet."""
    for j, q in enumerate(joint_angles):
        p.resetJointState(body_id, j, q)

def main_visualise():
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    n_joints = p.getNumJoints(robot_id)
    print("Loaded robot with", n_joints, "joints (first 6 are actuated).")

    print("Robot ID:", robot_id)
    print("Joints:", p.getNumJoints(robot_id))

    # Two poses to toggle with SPACE
    home = np.zeros(6)
    poses = [home, q_sample]
    pose_idx = 0
    set_joint_positions(robot_id, poses[pose_idx])

    print("Press SPACE in the Bullet window to toggle between poses.")
    p.setRealTimeSimulation(1)

    try:
        while p.isConnected():
            keys = p.getKeyboardEvents()
            if keys.get(p.B3G_SPACE, 0) & p.KEY_WAS_TRIGGERED:
                pose_idx = 1 - pose_idx
                set_joint_positions(robot_id, poses[pose_idx])
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()


if __name__ == "__main__":
    main_visualise()