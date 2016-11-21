#!/usr/bin/env python
import pickle
import numpy as np
import openravepy as orpy
import ikea_openrave.utils as rave_utils
from bimanual.utils import utils
from bimanual.utils import placement_utils as putils
from IPython import embed
import pymanip.planningutils.utils as pymanip_utils
from pymanip.planningutils import myobject, intermediateplacement, staticequilibrium

import os.path as path
model_path = path.abspath(path.join(path.dirname(__file__), "../xml"))

if __name__ == "__main__":
  # Generic configuration
  np.set_printoptions(precision=10, suppress=True)
  
  # Load OpenRAVE environment
  scene_file = model_path + '/worlds/bimanual_setup_regrasp.env.xml'
  env = orpy.Environment()
  env.SetViewer('qtcoin')

  frame = env.ReadKinBodyXMLFile(model_path + '/objects/ikea_left_frame.kinbody.xml')
  env.Add(frame)
  
  env.Load(scene_file)

  # Retrive robot and objects
  left_robot = env.GetRobot('denso_left')
  right_robot = env.GetRobot('denso_right')
  table = env.GetKinBody('table')
  # Correct robots transformation
  T_left_robot = np.array([[ 1.   ,  0.   ,  0.   ,  0    ],
                          [ 0.   ,  1.   ,  0.   , -0.536],
                          [ 0.   ,  0.   ,  1.   ,  0.005],
                          [ 0.   ,  0.   ,  0.   ,  1.   ]])
  T_right_robot = np.array([[ 1.   ,  0.   ,  0.   , 0.012],
                            [ 0.   ,  1.   ,  0.   , 0.536],
                            [ 0.   ,  0.   ,  1.   , 0],
                            [ 0.   ,  0.   ,  0.   , 1.   ]])
  T_frame = np.array(
    [[-0.068746388 ,  0.9383021362,  0.3389144956,  0.1319232285],
     [ 0.0225705356, -0.3381684353,  0.9408149022, -0.1287491769],
     [ 0.9973788172,  0.0723271079,  0.0020698767,  0.027942704 ],
     [ 0.          ,  0.          ,  0.          ,  1.          ]])


  T_table = np.eye(4)
  with env:
    left_robot.SetTransform(T_left_robot)
    right_robot.SetTransform(T_right_robot)
    frame.SetTransform(T_frame)
    table.SetTransform(T_table)

  # Add collision checker
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
  qgrasp_left = [5, 3, 1, -0.1]
  T_left_gripper = pymanip_utils.ComputeTGripper2(
                    frame, qgrasp_left[0], qgrasp_left)
  left_robot.SetActiveDOFValues(left_manip.FindIKSolutions(T_left_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[0])

  qgrasp_right = [2, 3, 2, 0]
  T_right_gripper = pymanip_utils.ComputeTGripper2(
                      frame, qgrasp_right[0], qgrasp_right)
  right_robot.SetActiveDOFValues(right_manip.FindIKSolutions(T_right_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[0])

  left_taskmanip.CloseFingers()
  left_robot.WaitForController(0)
  right_taskmanip.CloseFingers()
  right_robot.WaitForController(0)
  ################## closed chain planning ###################

  embed()
  exit(0)

  obj_translation_limits =  [[0.7, 0.5, 0.5], [-0.7, -0.5, 0]]
  q_robots_start = [left_robot.GetActiveDOFValues(),
                    right_robot.GetActiveDOFValues()]
  q_robots_grasp = [left_robot.GetDOFValues()[-1],
                    right_robot.GetDOFValues()[-1]]
  qgrasps = [qgrasp_left, qgrasp_right]
  T_obj_start = frame.GetTransform()
  T_obj_goal = np.array(
    [[-0.0676201641,  0.9215304983,  0.3823729255, -0.4412593842],
       [ 0.025748682 , -0.3815106065,  0.9240057698,  0.0456922799],
       [ 0.9973788241,  0.0723270207,  0.002069614 ,  0.2413713932],
       [ 0.          ,  0.          ,  0.          ,  1.          ]])

  q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], frame, q_robots_start,
                     q_robots_grasp, T_obj_start, T_obj_goal)


  p_frame = putils.create_placement_object(frame, env, T_rest=T_table)


  import bimanual.planners.cc_planner_regrasp_placement as ccp 
  ccplanner = ccp.CCPlanner(frame, p_frame, [left_robot, right_robot], 
                            plan_regrasp=True, debug=False)
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                        q_robots_goal, q_robots_grasp, qgrasps,
                        T_obj_start, nn=2, step_size=0.5, 
                        fmax=100, mu=0.3, regrasp_limit=5)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=30)

  ccplanner.shortcut(ccquery, maxiters=[30, 10])
  ccplanner.visualize_cctraj(ccquery.cctraj, speed=1)

