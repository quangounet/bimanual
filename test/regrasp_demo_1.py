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
  scene_file = model_path + '/worlds/bimanual_setup_regrasp_snapshot.env.xml'
  env = orpy.Environment()
  # env.SetViewer('qtcoin')

  Lshape = env.ReadKinBodyXMLFile(model_path + '/objects/Lshape_regrasp.kinbody.xml')
  env.Add(Lshape)
  
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
  T_Lshape = np.array([[ 0.   ,  1.   ,  0.   ,  0.37 ],
                       [ 1.   , -0.   ,  0.   ,  0.005],
                       [ 0.   , -0.   , -1.   ,  0.159],
                       [ 0.   ,  0.   ,  0.   ,  1.   ]])
  T_table = np.eye(4)
  with env:
    left_robot.SetTransform(T_left_robot)
    right_robot.SetTransform(T_right_robot)
    Lshape.SetTransform(T_Lshape)
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
  qgrasp_left = [1, 2, 4, -0.013]
  qgrasp_right = [0, 2, 0, 0.13]
  T_left_gripper = pymanip_utils.ComputeTGripper2(
                      Lshape, qgrasp_left[0], qgrasp_left)
  T_right_gripper = pymanip_utils.ComputeTGripper2(
                      Lshape, qgrasp_right[0], qgrasp_right)

  left_robot.SetActiveDOFValues(left_manip.FindIKSolutions(T_left_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[1])

  right_robot.SetActiveDOFValues(right_manip.FindIKSolutions(T_right_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[2])

  left_taskmanip.CloseFingers()
  left_robot.WaitForController(0)
  right_taskmanip.CloseFingers()
  right_robot.WaitForController(0)
  ################## closed chain planning ###################

  obj_translation_limits =  [[0.7, 0.5, 1.2], [-0.5, -0.5, 0]]
  q_robots_start = [left_robot.GetActiveDOFValues(),
                    right_robot.GetActiveDOFValues()]
  q_robots_grasp = [left_robot.GetDOFValues()[-1],
                    right_robot.GetDOFValues()[-1]]
  qgrasps = [qgrasp_left, qgrasp_right]
  T_obj_start = Lshape.GetTransform()
  T_obj_goal = np.array([[ 0.   ,  1.   ,  0.   , -0.23],
                         [ 1.   , -0.   ,  0.   ,  0.005],
                         [ 0.   , -0.   , -1.   ,  0.159],
                         [ 0.   ,  0.   ,  0.   ,  1.   ]])
  q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], Lshape, q_robots_start,
                     q_robots_grasp, T_obj_start, T_obj_goal, seeds=[0,0])

  p_Lshape = putils.create_placement_object(Lshape, env, T_rest=T_table)

  embed()
  exit(0)

  t_total = []
  t_regrasp = []
  rep = 50
  from time import time
  import bimanual.planners.cc_planner_regrasp_placement_bm as ccp 
  ccplanner = ccp.CCPlanner(Lshape, p_Lshape, [left_robot, right_robot], 
                            plan_regrasp=True, debug=False)
  t = time() 
  i = 0
  while i < rep:
    print 'i: ', i
    # fastest stepsize is 0.3
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, qgrasps,
                          T_obj_start, nn=2, step_size=0.3, 
                          fmax=100, mu=0.5, regrasp_limit=1)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=30)
    if res:
      i += 1
    t_total.append(ccquery.running_time)
    t_regrasp.append(ccquery.regrasp_planning_time)

  t_total = np.array(t_total)
  t_regrasp = np.array(t_regrasp)

  print (time()-t)/rep

  ccplanner.shortcut(ccquery, maxiters=[30, 60])
  ccplanner.visualize_cctraj(ccquery.cctraj, speed=1)

