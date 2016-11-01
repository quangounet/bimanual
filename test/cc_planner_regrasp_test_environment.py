#!/usr/bin/env python
import pickle
import numpy as np
import openravepy as orpy
import ikea_openrave.utils as rave_utils
from IPython import embed

import sys
sys.path.append('../src/')
from utils import utils

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
  Lshape = env.ReadKinBodyXMLFile('../xml/objects/Lshape.kinbody.xml')
  env.Add(Lshape)
  # Correct robots transformation
  T_left_robot = np.array([[ 1.   ,  0.   ,  0.   ,  0    ],
                          [ 0.   ,  1.   ,  0.   , -0.536],
                          [ 0.   ,  0.   ,  1.   ,  0.005],
                          [ 0.   ,  0.   ,  0.   ,  1.   ]])
  T_right_robot = np.array([[ 1.   ,  0.   ,  0.   , 0.012],
                            [ 0.   ,  1.   ,  0.   , 0.536],
                            [ 0.   ,  0.   ,  1.   , 0],
                            [ 0.   ,  0.   ,  0.   , 1.   ]])
  T_Lshape = np.array([[ 0.   ,  1.   ,  0.   ,  0.26 ],
                      [ 1.   , -0.   ,  0.   , -0.175],
                      [ 0.   , -0.   , -1.   ,  0.176],
                      [ 0.   ,  0.   ,  0.   ,  1.   ]])
  with env:
    left_robot.SetTransform(T_left_robot)
    right_robot.SetTransform(T_right_robot)
    Lshape.SetTransform(T_Lshape)
    env.Remove(env.GetKinBody('table'))

  # Add collision checker
  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'ode'))
  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'fcl_'))

  manip_name = 'denso_ft_sensor_gripper'
  left_manip = left_robot.SetActiveManipulator(manip_name)
  right_manip = right_robot.SetActiveManipulator(manip_name)
  left_basemanip = orpy.interfaces.BaseManipulation(left_robot)
  left_taskmanip = orpy.interfaces.TaskManipulation(left_robot)
  right_basemanip = orpy.interfaces.BaseManipulation(right_robot)
  right_taskmanip = orpy.interfaces.TaskManipulation(right_robot)

  rave_utils.disable_gripper([left_robot, right_robot])
  rave_utils.load_IK_model([left_robot, right_robot])

  ############### Move to pre-grasp position ###############
  left_robot.SetDOFValues(
    [ 0.8041125559,  0.9405270996,  1.0073165866,  3.1415926536, 
     -1.1937489674,  2.3749088827,  0.4560000086])
  right_robot.SetDOFValues(
    [-0.9816436375,  0.7763915147,  1.2904895262, -3.1415926536, 
     -1.0747116127,  2.1599490161,  0.4560000086])

  ################## closed chain planning ###################

  obj_translation_limits =  [[0.9, 0.5, 1.2], [-0.5, -0.4, -0.4]]
  q_robots_start = [left_robot.GetActiveDOFValues(),
                    right_robot.GetActiveDOFValues()]
  q_robots_grasp = [left_robot.GetDOFValues()[-1],
                    right_robot.GetDOFValues()[-1]]
  T_obj_start = Lshape.GetTransform()
  T_obj_goal = np.array(
    [[ 0.          , -0.6233531712, -0.7819404223,  0.0683513284],
     [ 1.          ,  0.          ,  0.          , -0.1750002205],
     [ 0.          , -0.7819404223,  0.6233531712,  0.84509027  ],
     [ 0.          ,  0.          ,  0.          ,  1.          ]])

  q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], Lshape, q_robots_start,
                     q_robots_grasp, T_obj_start, T_obj_goal)

  embed()
  exit(0)
  import cc_planner_regrasp as ccp 

  ccplanner = ccp.CCPlanner(Lshape, [left_robot, right_robot], 
                            plan_regrasp=True, debug=False)
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                        q_robots_goal, q_robots_grasp, T_obj_start, nn=2, 
                        step_size=0.5, regrasp_limits=[0, 1])
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=100)

  rep = 100
  from time import time
  ccplanner = ccp.CCPlanner(Lshape, [left_robot, right_robot], debug=False)
  t = time() 
  i = 0
  while i < rep:
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, T_obj_start, nn=2, 
                          step_size=0.5, regrasp_limits=[1, 1])
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=20)
    if res:
      i += 1
  print (time()-t)/rep

  ccplanner.shortcut(ccquery, maxiter=20)
  ccplanner.visualize_cctraj(ccquery.cctraj, speed=2)
