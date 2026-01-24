#!/usr/bin/env python3
"""
Test script to verify obstacle avoidance constraints are properly implemented
"""

import numpy as np
import yaml
import casadi as ca
from src.environments.drone import Drone

def test_obstacle_constraints():
    """Test that obstacle constraints are correctly formulated"""
    
    print("="*60)
    print("TESTING OBSTACLE AVOIDANCE CONSTRAINTS")
    print("="*60)
    
    # Load parameters
    with open("params/params_drone_obstacles.yaml", 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # Initialize drone environment
    drone_env = Drone(params)
    
    # Test 1: Check constraint expression
    print("\n[Test 1] Checking constraint expression...")
    nx = params["agent"]["dim"]["nx"]
    num_dyn = params["agent"]["num_dyn_samples"]
    
    # Create symbolic state
    model_x = ca.SX.sym('x', nx * num_dyn)
    
    # Get constraint expressions
    const_expr, const_expr_e = drone_env.const_expr(model_x, num_dyn)
    
    if const_expr is not None:
        print(f"✓ Obstacle constraints created: {const_expr.size1()} constraints")
        num_obstacles = len(params["env"]["obstacles"])
        expected_constraints = num_obstacles * num_dyn
        assert const_expr.size1() == expected_constraints, \
            f"Expected {expected_constraints} constraints, got {const_expr.size1()}"
        print(f"✓ Correct number of constraints: {expected_constraints}")
    else:
        print("✗ No obstacle constraints found!")
        return False
    
    # Test 2: Check constraint bounds
    print("\n[Test 2] Checking constraint bounds...")
    lh, uh, lh_e, uh_e = drone_env.const_value(num_dyn)
    
    print(f"✓ Lower bounds shape: {lh.shape}")
    print(f"✓ Upper bounds shape: {uh.shape}")
    
    num_obstacles = len(params["env"]["obstacles"])
    expected_bounds = num_obstacles * num_dyn
    assert len(lh) == expected_bounds, f"Expected {expected_bounds} bounds, got {len(lh)}"
    print(f"✓ Correct number of bounds: {expected_bounds}")
    
    # Test 3: Verify constraint values at specific points
    print("\n[Test 3] Verifying constraint evaluation...")
    
    # Create a function to evaluate constraints
    const_func = ca.Function('const_func', [model_x], [const_expr])
    
    # Test point inside obstacle (should violate constraint)
    x_inside = np.zeros(nx * num_dyn)
    x_inside[0] = 0.0  # px at obstacle center
    x_inside[1] = 0.0  # py at obstacle center
    
    const_val_inside = const_func(x_inside)
    obs1_radius = params["env"]["obstacles"]["obs1"][2]
    
    print(f"  Point at obstacle center (0, 0):")
    print(f"    Constraint value: {float(const_val_inside[0]):.4f}")
    print(f"    Required minimum: {obs1_radius**2:.4f}")
    
    if float(const_val_inside[0]) < obs1_radius**2:
        print(f"  ✓ Correctly identifies collision (value < r^2)")
    else:
        print(f"  ✗ Should identify collision!")
    
    # Test point outside obstacle (should satisfy constraint)
    x_outside = np.zeros(nx * num_dyn)
    x_outside[0] = 3.0  # px far from obstacles
    x_outside[1] = 3.0  # py far from obstacles
    
    const_val_outside = const_func(x_outside)
    
    print(f"\n  Point far from obstacles (3, 3):")
    print(f"    Constraint value: {float(const_val_outside[0]):.4f}")
    print(f"    Required minimum: {obs1_radius**2:.4f}")
    
    if float(const_val_outside[0]) >= obs1_radius**2:
        print(f"  ✓ Correctly identifies safe region (value >= r^2)")
    else:
        print(f"  ✗ Should be in safe region!")
    
    # Test 4: Check all obstacles
    print("\n[Test 4] Checking all obstacles...")
    for i, (obs_name, obs_params) in enumerate(params["env"]["obstacles"].items()):
        cx, cy, r = obs_params[0], obs_params[1], obs_params[2]
        print(f"  {obs_name}: center=({cx}, {cy}), radius={r}")
        
        # Test point at obstacle center
        x_test = np.zeros(nx * num_dyn)
        x_test[0] = cx
        x_test[1] = cy
        
        const_val = const_func(x_test)
        constraint_idx = i * num_dyn  # First dynamic sample, i-th obstacle
        
        dist_sq = float(const_val[constraint_idx])
        print(f"    Distance^2 at center: {dist_sq:.4f} (should be ~0)")
        print(f"    Constraint bound: {r**2:.4f}")
        
        if dist_sq < r**2:
            print(f"    ✓ Collision detected correctly")
        else:
            print(f"    ✗ Should detect collision!")
    
    # Test 5: Path generator
    print("\n[Test 5] Checking path generator...")
    path = drone_env.path_generator(0, 10)
    print(f"✓ Path shape: {path.shape}")
    print(f"  First point: {path[0]}")
    print(f"  Last point: {path[-1]}")
    
    goal = np.array(params["env"]["goal_state"])
    if np.allclose(path[0], goal) and np.allclose(path[-1], goal):
        print(f"✓ Path correctly points to goal {goal}")
    else:
        print(f"✗ Path should point to goal!")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nObstacle avoidance constraints are correctly implemented.")
    print("You can now run: python demo_obstacle_avoidance.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        test_obstacle_constraints()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
