#!/usr/bin/env python
import pickle
import numpy as np
import openravepy as orpy
import ikea_openrave.utils as rave_utils
from bimanual.utils import utils
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

  Lshape = env.ReadKinBodyXMLFile(model_path + '/objects/Lshape_regrasp.kinbody.xml')
  env.Add(Lshape)
  myObject = myobject.MyObject(Lshape)
  
  env.Load(scene_file)

  # Retrive robot and objects
  left_robot = env.GetRobot('denso_left')
  right_robot = env.GetRobot('denso_right')
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
  qgrasp_left = [1, 2, 1, 0.013]
  qgrasp_right = [0, 2, 0, 0.13]

  left_robot.SetDOFValues(
    [ 0.6949541317,  1.3183457567,  0.2273780467,  3.1415926536,
      -1.5958688501,  2.2657504585,  0.4560000086])
  right_robot.SetDOFValues(
    [-0.8419911695,  0.9491865457,  0.9209619619, -3.1415926536,
     -1.2714441461,  2.2996014841,  0.4560000086])

  ################## closed chain planning ###################

  obj_translation_limits =  [[0.7, 0.5, 1.2], [-0.5, -0.5, 0]]
  q_robots_start = [left_robot.GetActiveDOFValues(),
                    right_robot.GetActiveDOFValues()]
  q_robots_grasp = [left_robot.GetDOFValues()[-1],
                    right_robot.GetDOFValues()[-1]]
  T_obj_start = Lshape.GetTransform()
  T_obj_goal = np.array([[ 0.   ,  1.   ,  0.   , -0.23],
                         [ 1.   , -0.   ,  0.   ,  0.005],
                         [ 0.   , -0.   , -1.   ,  0.159],
                         [ 0.   ,  0.   ,  0.   ,  1.   ]])

  q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], Lshape, q_robots_start,
                     q_robots_grasp, T_obj_start, T_obj_goal, seeds=[0,0])

  embed()
  exit(0)

  left_robot.SetDOFValues([0],[6])
  right_robot.SetDOFValues([0],[6])
  Lshape.SetTransform(np.array(
      [[ 0.2639622163,  0.8893508793,  0.373334919 ,  0.2613566816],
       [ 0.9586656352, -0.2845350591, -0.0000000021,  0.0416169688],
       [ 0.1062268713,  0.3579033579, -0.9276966305,  0.1387752742],
       [ 0.          ,  0.          ,  0.          ,  1.          ]]))


  T_left_gripper = pymanip_utils.ComputeTGripper2(
                      Lshape, qgrasp_left[0], qgrasp_left)
  T_right_gripper = pymanip_utils.ComputeTGripper2(
                      Lshape, qgrasp_right[0], qgrasp_right)
  left_robot.SetActiveDOFValues(left_manip.FindIKSolution(T_left_gripper, orpy.IkFilterOptions.CheckEnvCollisions))
  right_robot.SetActiveDOFValues(right_manip.FindIKSolution(T_right_gripper, orpy.IkFilterOptions.CheckEnvCollisions))

  fmax = 100
  mu = 0.5
  res = intermediateplacement.ComputeFeasibleClosePlacements([left_robot, right_robot], [qgrasp_left, qgrasp_right], Lshape, Lshape.GetTransform(), T_table, fmax, mu, placementType=2, myObject=myObject)


  import bimanual.planners.cc_planner_regrasp_transfer as ccp 
  ccplanner = ccp.CCPlanner(Lshape, [left_robot, right_robot], 
                            plan_regrasp=True, debug=False)
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                        q_robots_goal, q_robots_grasp, T_obj_start, nn=2, 
                        step_size=0.5, regrasp_limit=1)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=30)

  ccplanner.shortcut(ccquery, maxiters=[20, 50])
  ccplanner.visualize_cctraj(ccquery.cctraj, speed=2)

