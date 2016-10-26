#!/usr/bin/env python
import pickle
import numpy as np
import openravepy as orpy
import ikea_openrave.utils as rave_utils
from time import time
from IPython import embed

import sys
sys.path.append('../src/')

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

  velocity_scale = 0.5
  rave_utils.scale_DOF_limits(left_robot, v=velocity_scale)
  rave_utils.scale_DOF_limits(right_robot, v=velocity_scale)

  # Add collision checker
  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'ode'))

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
  with open('./query_info.pkl', 'rb') as f:
    (obj_translation_limits, q_robots_start, 
      q_robots_goal, q_robots_grasp, 
      T_obj_start, T_obj_goal) = pickle.load(f)

  embed()
  exit(0)

  rep_time = 20

  ################ Different planner variations #################
  import cc_planner as ccp
  ccplanner = ccp.CCPlanner(cage, [left_robot, right_robot], debug=False)
  t = time()
  for i in xrange(rep_time):
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, T_obj_start, nn=2,
                          step_size=0.5, velocity_scale=velocity_scale,
                          enable_bw=True)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=20)
  print (time()-t)/rep_time


  import cc_planner_connect as ccp
  ccplanner = ccp.CCPlanner(cage, [left_robot, right_robot], debug=False)
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, q_robots_goal,
                        q_robots_grasp, T_obj_start, nn=2, step_size=0.5,
                        velocity_scale=velocity_scale, enable_bw=True)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=20)


  import cc_planner_ms as ccp
  ccplanner = ccp.CCPlanner(cage, left_robot, right_robot, debug=False)
  ccquery = ccp.CCQuery(q_robots_start[0], q_robots_start[1], 
                        q_robots_goal[0], q_robots_goal[1],
                        q_robots_grasp[0], q_robots_grasp[1], 
                        T_obj_start, obj_translation_limits,
                        step_size=0.7, nn=2, velocity_scale=velocity_scale, 
                        enable_bw=True)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=20)



  import cc_planner_ms_connect as ccp
  ccplanner = ccp.CCPlanner(cage, left_robot, right_robot, debug=False)
  ccquery = ccp.CCQuery(q_robots_start[0], q_robots_start[1], 
                        q_robots_goal[0], q_robots_goal[1],
                        q_robots_grasp[0], q_robots_grasp[1], 
                        T_obj_start, obj_translation_limits,
                        step_size=0.7, nn=2, velocity_scale=velocity_scale, 
                        enable_bw=True)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=20)

  ###################### Visualization #######################
  ccplanner.visualize_cctraj(ccquery.cctraj, speed=5)
  ccplanner.shortcut(ccquery, maxiter=20)
  ccplanner.visualize_cctraj(ccquery.cctraj, speed=5)
