#!/usr/bin/env python

import pickle
import numpy as np
import openravepy as orpy
from time import time
from IPython import embed
import ikea_openrave.utils as rave_utils

import sys
sys.path.append('../src/')
import cc_planner as ccp

if __name__ == "__main__":
  # Generic configuration
  np.set_printoptions(precision=10, suppress=True)
  
  # Load OpenRAVE environment
  scene_file = '../xml/worlds/bimanual_setup.env.xml'
  env = orpy.Environment()
  env.SetViewer('qtcoin')
  env.Load(scene_file)

  # Retrive robot and objects
  left_robot = env.GetRobot('denso_left')
  right_robot = env.GetRobot('denso_right')
  cage = env.ReadKinBodyXMLFile('../xml/objects/cage.kinbody.xml')
  env.Add(cage)

  velocity_scale = 0.2
  rave_utils.scale_DOF_limits(left_robot, v=velocity_scale)
  rave_utils.scale_DOF_limits(right_robot, v=velocity_scale)

  manip_name = 'denso_ft_sensor_gripper'
  left_manip = left_robot.SetActiveManipulator(manip_name)
  right_manip = right_robot.SetActiveManipulator(manip_name)
  left_basemanip = orpy.interfaces.BaseManipulation(left_robot)
  left_taskmanip = orpy.interfaces.TaskManipulation(left_robot)
  right_basemanip = orpy.interfaces.BaseManipulation(right_robot)
  right_taskmanip = orpy.interfaces.TaskManipulation(right_robot)

  rave_utils.disable_gripper([left_robot, right_robot])
  rave_utils.load_IK_model([left_robot, right_robot])

  
  # Load sample query info
  with open('../tests/query_info.pkl', 'rb') as f:
    (obj_translation_limits, q_robots_start, 
      q_robots_goal, q_robots_grasp, 
      T_obj_start, T_obj_goal) = pickle.load(f)

  ccplanner = ccp.CCPlanner(cage, [left_robot, right_robot], debug=False)

  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'ode'))
  t0 = time()
  for i in xrange(100):
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, T_obj_start, 
                          step_size=0.7, velocity_scale=velocity_scale, 
                          enable_bw=False)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=20)
    ccplanner.shortcut(ccquery, maxiter=20)
  print 'ode:', (time() - t0) / 100.0

  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'fcl_'))
  t0 = time()
  for i in xrange(100):
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, T_obj_start, 
                          step_size=0.7, velocity_scale=velocity_scale, 
                          enable_bw=False)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=20)
    ccplanner.shortcut(ccquery, maxiter=20)
  print 'fcl:', (time() - t0) / 100.0
