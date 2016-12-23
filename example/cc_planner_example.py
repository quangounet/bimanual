#!/usr/bin/env python
import pickle
import numpy as np
import openravepy as orpy
from bimanual.utils import utils
from bimanual.utils.loggers import TextColors
from time import time
from IPython import embed

import os.path as path
model_path = path.abspath(path.join(path.dirname(__file__), "../xml"))

if __name__ == "__main__":
  # Generic configuration
  np.set_printoptions(precision=10, suppress=True)
  
  # Load OpenRAVE environment
  scene_file = model_path + '/worlds/bimanual_setup.env.xml'
  env = orpy.Environment()
  env.SetViewer('qtcoin')
  env.Load(scene_file)

  # Retrive robot and objects
  left_robot = env.GetRobot('denso_left')
  right_robot = env.GetRobot('denso_right')
  cage = env.ReadKinBodyXMLFile(model_path + '/objects/cage.kinbody.xml')
  env.Add(cage)

  velocity_scale = 0.5
  utils.scale_DOF_limits(left_robot, v=velocity_scale)
  utils.scale_DOF_limits(right_robot, v=velocity_scale)

  # Add collision checker
  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'ode'))

  manip_name = 'denso_ft_sensor_gripper'
  left_manip = left_robot.SetActiveManipulator(manip_name)
  right_manip = right_robot.SetActiveManipulator(manip_name)
  left_basemanip = orpy.interfaces.BaseManipulation(left_robot)
  left_taskmanip = orpy.interfaces.TaskManipulation(left_robot)
  right_basemanip = orpy.interfaces.BaseManipulation(right_robot)
  right_taskmanip = orpy.interfaces.TaskManipulation(right_robot)

  utils.disable_gripper([left_robot, right_robot])
  utils.load_IK_model([left_robot, right_robot])

  cage.SetTransform(np.array(
      [[ 0.9932,  0.    ,  0.1168,  0.3328],
       [-0.1168, -0.    ,  0.9932, -0.1473],
       [ 0.    , -1.    , -0.    ,  0.5182],
       [ 0.    ,  0.    ,  0.    ,  1.    ]]))

  left_robot.SetActiveDOFValues(
    [ 1.4198645611,  0.3863062134,  2.5424642416, 
     -1.6038764249,  1.5636383162,  1.3580719875])
  right_robot.SetActiveDOFValues(
    [-1.4300510265,  0.7726813928,  0.9824060939, 
     -1.5225318994, -1.317566809 ,  1.3802711535])

  left_taskmanip.CloseFingers()
  left_robot.WaitForController(0)
  right_taskmanip.CloseFingers()
  right_robot.WaitForController(0)

  ################## closed chain planning ###################
  obj_translation_limits = [[0.6, 0.25, 0.91], [0.28, -0.35, 0.136]]
  q_robots_start = [left_robot.GetActiveDOFValues(),
                    right_robot.GetActiveDOFValues()]
  q_robots_grasp = [left_robot.GetDOFValues()[-1],
                    right_robot.GetDOFValues()[-1]]
  T_obj_start = cage.GetTransform()
  T_obj_goal = np.array(
    [[ 1.    ,  0.    ,  0.    ,  0.37  ],
     [ 0.    , -1.    , -0.    ,  0.18  ],
     [ 0.    ,  0.    , -1.    ,  0.3882],
     [ 0.    ,  0.    ,  0.    ,  1.    ]])

  q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], cage, q_robots_start,
                     q_robots_grasp, T_obj_start, T_obj_goal)
  embed()
  exit(0)

  logger = TextColors(TextColors.INFO)
  import bimanual.planners.cc_planner_connect as ccp
  ccplanner = ccp.CCPlanner(cage, [left_robot, right_robot], logger=logger, 
                            planner_type='BiRRT')
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                        q_robots_goal, q_robots_grasp, T_obj_start, nn=2,
                        step_size=0.5, velocity_scale=velocity_scale,
                        enable_bw=True)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=20)

  ###################### Visualization #######################
  ccplanner.shortcut(ccquery, maxiter=20)
  ccplanner.visualize_cctraj(ccquery.cctraj, speed=1)
