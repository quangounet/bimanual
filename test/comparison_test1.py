#!/usr/bin/env python
import os
import numpy as np
import openravepy as orpy
import ikea_openrave.utils as rave_utils
from time import time
from bimanual.utils import utils
from IPython import embed

import os.path as path
model_path = path.abspath(path.join(path.dirname(__file__), "../xml"))

class Comparison(object):
  def __init__(self):
    np.set_printoptions(precision=10, suppress=True)

    # Load OpenRAVE environment
    scene_file = model_path + '/worlds/bimanual_setup.env.xml'
    env = orpy.Environment()
    env.SetViewer('qtcoin')
    env.Load(scene_file)

    # Retrive robot and objects
    left_robot = env.GetRobot('denso_left')
    right_robot = env.GetRobot('denso_right')
    stick = env.ReadKinBodyXMLFile(model_path + '/objects/ikea_stick.kinbody.xml')
    # Correct robots transformation
    T_left_robot = np.array([[ 1.   ,  0.   ,  0.   ,  0    ],
                            [ 0.   ,  1.   ,  0.   , -0.536],
                            [ 0.   ,  0.   ,  1.   ,  0.005],
                            [ 0.   ,  0.   ,  0.   ,  1.   ]])
    T_right_robot = np.array([[ 1.   ,  0.   ,  0.   , 0.012],
                              [ 0.   ,  1.   ,  0.   , 0.536],
                              [ 0.   ,  0.   ,  1.   , 0],
                              [ 0.   ,  0.   ,  0.   , 1.   ]])
    T_stick = np.array(
      [[-1.          ,  0.          ,  0.          ,  0.],
       [ 0.          ,  1.          ,  0.          ,  0.],
       [ 0.          ,  0.          , -1.          ,  0.],
       [ 0.          ,  0.          ,  0.          ,  1.]])
    T_obs_original = np.array(
      [[-1.          ,  0.          ,  0.          ,  -0.6],
       [ 0.          ,  1.          ,  0.          ,  0.],
       [ 0.          ,  0.          , -1.          ,  0.],
       [ 0.          ,  0.          ,  0.          ,  1.]])
    with env:
      env.Remove(env.GetKinBody('table'))
      env.Remove(env.GetKinBody('reference_frame'))
      env.Add(stick)
      left_robot.SetTransform(T_left_robot)
      right_robot.SetTransform(T_right_robot)
      stick.SetTransform(T_stick)
      obs = []
      for i in xrange(3):
        obs.append(env.ReadKinBodyXMLFile(
          '../xml/objects/ikea_stick.kinbody.xml'))
        env.Add(obs[i], True)
      obs[0].SetTransform(np.array(
        [[-1.          ,  0.0000001187, -0.0000001192, -0.039773237 ],
         [-0.0000001187,  0.0043285555,  0.9999906318,  0.008154206 ],
         [ 0.0000001192,  0.9999906318, -0.0043285555,  0.0952612907],
         [ 0.          ,  0.          ,  0.          ,  1.          ]]))
      obs[1].SetTransform(np.array(
        [[-1.          ,  0.0000001189, -0.0000001192,  0.0754337907],
         [-0.0000001189,  0.0025676682,  0.9999967035, -0.0017708684],
         [ 0.0000001192,  0.9999967035, -0.0025676682,  0.0948928073],
         [ 0.          ,  0.          ,  0.          ,  1.          ]]))
      obs[2].SetTransform(np.array(
        [[ 0.0330897657, -0.9994523838,  0.0000001191,  0.0684351176],
         [-0.9994523838, -0.0330897657,  0.0000001232,  0.0085579157],
         [-0.0000001191, -0.0000001232, -1.          , -0.0601185635],
         [ 0.          ,  0.          ,  0.          ,  1.          ]]))

    # Add collision checker
    env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'ode'))

    manip_name = 'denso_ft_sensor_gripper'
    left_manip = left_robot.SetActiveManipulator(manip_name)
    right_manip = right_robot.SetActiveManipulator(manip_name)

    rave_utils.disable_gripper([left_robot, right_robot])
    rave_utils.load_IK_model([left_robot, right_robot])
    
    # Create manipulation tasks  
    left_basemanip = orpy.interfaces.BaseManipulation(left_robot)
    left_taskmanip = orpy.interfaces.TaskManipulation(left_robot)
    right_basemanip = orpy.interfaces.BaseManipulation(right_robot)
    right_taskmanip = orpy.interfaces.TaskManipulation(right_robot)

    orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)

    ############### Move to pre-grasp position ###############
    left_robot.SetActiveDOFValues(
      [ 1.5707963268,  1.0063337055,  1.5325008337,  0.          ,
        0.6027581144,  1.5707963268])
    right_robot.SetActiveDOFValues(
      [-1.5983123161,  0.9934847953,  1.5358623814, -0.          ,
        0.6122454769,  1.5432803375])
    left_taskmanip.CloseFingers()
    right_taskmanip.CloseFingers()

    ################## closed chain planning ###################
    
    rep_time = 20
    obj_translation_limits =  [[0.8, 0.8, 0.8], [-0.3, -0.3, -0.3]]
    q_robots_start = [left_robot.GetActiveDOFValues(),
                      right_robot.GetActiveDOFValues()]
    q_robots_grasp = [left_robot.GetDOFValues()[-1],
                      right_robot.GetDOFValues()[-1]]
    T_obj_start = stick.GetTransform()

    T_obj_goal = np.array([[-1. ,  0. ,  0. ,  0.15 ],
                           [ 0. ,  1. ,  0. ,  0. ],
                           [ 0. ,  0. , -1. ,  0.1],
                           [ 0. ,  0. ,  0. ,  1. ]])

    q_robots_goal = utils.compute_bimanual_goal_configs(
                      [left_robot, right_robot], stick, q_robots_start,
                       q_robots_grasp, T_obj_start, T_obj_goal)

    embed()
    exit(0)

    rep_time = 20

    from bimanual.planners.cc_planner_connect import CCPlanner, CCQuery
    ccplanner = CCPlanner(stick, [left_robot, right_robot], debug=False)
    t = time()
    for i in xrange(rep_time):
      ccquery = CCQuery(obj_translation_limits, q_robots_start, 
        q_robots_goal, q_robots_grasp, T_obj_start, nn=2,
        step_size=0.2, enable_bw=False)
      ccplanner.set_query(ccquery)
      res = ccplanner.solve(timeout=20)
    print (time()-t)/rep_time

    from bimanual.planners.cc_planner_ms_connect import CCPlanner, CCQuery
    ccplanner = CCPlanner(stick, left_robot, right_robot, debug=False)
    t = time()
    for i in xrange(rep_time):
      ccquery = CCQuery(q_robots_start[0], q_robots_start[1], 
                          q_robots_goal[0], q_robots_goal[1],
                          q_robots_grasp[0], q_robots_grasp[1], 
                          T_obj_start, obj_translation_limits,
                          nn=2, step_size=0.2, enable_bw=False)
      ccplanner.set_query(ccquery)
      res = ccplanner.solve(timeout=20)
    print (time()-t)/rep_time

    ccplanner.visualize_cctraj(ccquery.cctraj)


if __name__ == "__main__":
  demo = Comparison()
