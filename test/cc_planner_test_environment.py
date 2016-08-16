#!/usr/bin/env python

import pickle
import numpy as np
from openravepy import *
from IPython import embed

import sys
sys.path.append('../src/')
from utils import utils
from cc_planner import *

if __name__ == "__main__":
    # Generic configuration
    np.set_printoptions(precision=10, suppress=True)
    
    # Load OpenRAVE environment
    scene_file = '../xml/worlds/bimanual_ikea_assembly.env.xml'
    env = Environment()
    env.SetViewer('qtcoin')
    env.Load(scene_file)

    # Retrive robot and objects
    left_robot = env.GetRobot('denso_left')
    right_robot = env.GetRobot('denso_right')
    cage = env.ReadKinBodyXMLFile('../xml/objects/cage.kinbody.xml')
    env.Add(cage)

    velocity_scale = 0.5
    utils.scale_DOF_limits(left_robot, v=velocity_scale)
    utils.scale_DOF_limits(right_robot, v=velocity_scale)

    # Add collision checker
    collision_checker = RaveCreateCollisionChecker(env, 'ode')
    env.SetCollisionChecker(collision_checker)

    manip_name = 'denso_ft_sensor_gripper'
    left_manip = left_robot.SetActiveManipulator(manip_name)
    right_manip = right_robot.SetActiveManipulator(manip_name)

    utils.disable_gripper([left_robot, right_robot])
    utils.load_IK_model([left_robot, right_robot])
    
    # Load sample query info
    with open('query_info.pkl', 'rb') as f:
        obj_translation_limits, q_robots_start, q_robots_goal, T_obj_start, T_obj_goal = pickle.load(f)
    ccquery = CCQuery(obj_translation_limits, q_robots_start, q_robots_goal, T_obj_start, T_obj_goal, velocity_scale=velocity_scale) # Can omit T_obj_goal since not necessary
    
    ccplanner = CCPlanner(cage, [left_robot, right_robot], debug=True)
    res = ccplanner.solve(ccquery, timeout=20)

    embed()
    exit(0)
    
    ccplanner.visualize_cctraj(ccquery.cctraj, speed=1)
    ccplanner.shortcut(ccquery, maxiter=20)
    ccplanner.visualize_cctraj(ccquery.cctraj, speed=1)