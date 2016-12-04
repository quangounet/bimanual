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
  file_path = '/worlds/bimanual_setup_regrasp_snapshot.env.xml'
  # file_path = '/worlds/bimanual_setup_regrasp.env.xml'
  scene_file = model_path + file_path
  env = orpy.Environment()
  # env.SetViewer('qtcoin')

  frame = env.ReadKinBodyXMLFile(model_path + '/objects/ikea_right_frame.kinbody.xml')
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
  T_right_robot = np.array(
    [[ 0.9999599223, -0.0040494323,  0.0071425583, -0.0010716475],
     [ 0.0040038435,  0.9999752852,  0.0057140467,  0.53482426  ],
     [-0.0080076871, -0.00568522  ,  0.9999581659,  0.0010255922],
     [ 0.          ,  0.          ,  0.          ,  1.          ]])
  T_frame = np.array(
   [[ 0.0141874819, -0.1104851349,  0.9937765092, -0.3449324965],
    [-0.9969209075,  0.0750919977,  0.0225808787,  0.01489163  ],
    [-0.0771195148, -0.9910369453, -0.1090795746,  0.4210543382],
    [ 0.          ,  0.          ,  0.          ,  1.          ]])

  T_table = np.array([[ 1.,  0.,  0.,  -0.005],
                      [ 0.,  1.,  0.,  0.],
                      [ 0.,  0.,  1.,  0.005],
                      [ 0.,  0.,  0.,  1.]])

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
  qgrasp_left = [5, 3, 1, 0]
  T_left_gripper = pymanip_utils.ComputeTGripper2(
                    frame, qgrasp_left[0], qgrasp_left)
  left_robot.SetActiveDOFValues(left_manip.FindIKSolutions(T_left_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[0])


  qgrasp_right = [4, 0, 1, 0]
  T_right_gripper = pymanip_utils.ComputeTGripper2(
                      frame, qgrasp_right[0], qgrasp_right)
  right_robot.SetActiveDOFValues(right_manip.FindIKSolutions(T_right_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[0])

  left_taskmanip.CloseFingers()
  left_robot.WaitForController(0)
  right_taskmanip.CloseFingers()
  right_robot.WaitForController(0)

  ################## closed chain planning ###################

  obj_translation_limits =  [[0.8, 0.4, 1.0], [-0.8, -0.4, -0.2]]
  q_robots_start = [left_robot.GetActiveDOFValues(),
                    right_robot.GetActiveDOFValues()]
  q_robots_grasp = [left_robot.GetDOFValues()[-1],
                    right_robot.GetDOFValues()[-1]]
  qgrasps = [qgrasp_left, qgrasp_right]
  T_obj_start = frame.GetTransform()
  T_obj_goal = np.array(
   [[ 0.0038325298,  0.1402944345, -0.9901024105,  0.4820188093],
    [-0.9972415754,  0.0739288701,  0.0066153268, -0.142555654 ],
    [ 0.074125246 ,  0.9873459342,  0.1401907774,  0.0648671477],
    [ 0.          ,  0.          ,  0.          ,  1.          ]])
  q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], frame, q_robots_start,
                     q_robots_grasp, T_obj_start, T_obj_goal)

  p_frame = putils.create_placement_object(frame, env, T_rest=T_table)

  embed()
  exit(0)

  # import bimanual.planners.cc_planner_regrasp_placement_multi_trial as ccp 
  import bimanual.planners.cc_planner_regrasp_placement_bm as ccp 
  ccplanner = ccp.CCPlanner(frame, p_frame, [left_robot, right_robot], 
                            plan_regrasp=True, debug=False)
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                        q_robots_goal, q_robots_grasp, qgrasps,
                        T_obj_start, nn=2, step_size=0.4, 
                        fmax=100, mu=0.3, regrasp_limit=3)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=100)

  t_total = []
  t_regrasp = []
  rep = 10
  from time import time
  import bimanual.planners.cc_planner_regrasp_placement_bm as ccp 
  ccplanner = ccp.CCPlanner(frame, p_frame, [left_robot, right_robot], 
                            plan_regrasp=True, debug=False)
  t = time() 
  i = 0
  while i < rep:
    print 'i: ', i
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, qgrasps,
                          T_obj_start, nn=2, step_size=0.3, 
                          fmax=500, mu=0.8, regrasp_limit=3, black_ratio=10)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=15)
    if res:
      i += 1
    t_total.append(ccquery.running_time)
    t_regrasp.append(ccquery.regrasp_planning_time)

  t_total = np.array(t_total)
  t_regrasp = np.array(t_regrasp)

  print (time()-t)/rep



  # 0.4 
  #     20  16.54
  #     10  21.47
  #     5   18.96
  # 0.3
  #     20  19.98  15.94
  #     10  13.48  17.45
            # timeout 10  18.66
            #         15  12.97
            #         30  13.25
  #     5   17.61  18.22
  #     2   16.88  17.34

  t1 20 18
  t2 20 15
  t3 20 15
  t4 10 15