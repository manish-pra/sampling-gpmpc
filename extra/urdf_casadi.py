from urdf2casadi import urdfparser as u2c
import os
current_file_path = os.path.abspath(__file__)
print(current_file_path)
urdf_path = "/home/manish/work/MPC_Dyn/urdf2casadi/examples/urdf/ur5_mod.urdf"
root_link = "base_link"
end_link = "tool0"
robot_parser = u2c.URDFparser()
robot_parser.from_file(urdf_path)
# Also supports .from_server for ros parameter server, or .from_string if you have the URDF as a string.
fk_dict = robot_parser.get_forward_kinematics(root_link, end_link)
print(fk_dict.keys())
# should give ['q', 'upper', 'lower', 'dual_quaternion_fk', 'joint_names', 'T_fk', 'joint_list', 'quaternion_fk']
forward_kinematics = fk_dict["T_fk"]
print(forward_kinematics([0.3, 0.3, 0.3, 0., 0.3, 0.7]))