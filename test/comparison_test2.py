#!/usr/bin/env python
import rospy
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
    Lshape = env.ReadKinBodyXMLFile(model_path + '/objects/Lshape.kinbody.xml')
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

    ############### Move to pre-grasp position ###############
    left_robot.SetDOFValues(
      [ 0.8041125559,  0.9405270996,  1.0073165866,  3.1415926536, 
       -1.1937489674,  2.3749088827,  0.4560000086])
    right_robot.SetDOFValues(
      [-0.9816436375,  0.7763915147,  1.2904895262, -3.1415926536, 
       -1.0747116127,  2.1599490161,  0.4560000086])
    ################## closed chain planning ###################

    obj_translation_limits =  [[0.7, 0.25, 1.2], [-0.1, -0.5, 0]]
    q_robots_start = [left_robot.GetActiveDOFValues(),
                      right_robot.GetActiveDOFValues()]
    q_robots_grasp = [left_robot.GetDOFValues()[-1],
                      right_robot.GetDOFValues()[-1]]
    T_obj_start = Lshape.GetTransform()

    T_obj_goal = np.array(
      [[ 0.0582528963, -0.9964414827,  0.0609177455,  0.1181300282],
       [ 0.9983001464,  0.0582572434, -0.00170625  , -0.0844197497],
       [-0.0018487217,  0.0609135882,  0.9981413312,  0.9244756699],
       [ 0.          ,  0.          ,  0.          ,  1.          ]])

    q_robots_goal = utils.compute_bimanual_goal_configs(
                      [left_robot, right_robot], Lshape, q_robots_start,
                       q_robots_grasp, T_obj_start, T_obj_goal)

    embed()
    exit(0)

    rep_time = 20

    from bimanual.planners.cc_planner_connect import CCPlanner, CCQuery
    ccplanner = CCPlanner(Lshape, [left_robot, right_robot], debug=False)
    t = time()
    for i in xrange(rep_time):
      ccquery = CCQuery(obj_translation_limits, q_robots_start, 
        q_robots_goal, q_robots_grasp, T_obj_start, nn=2,
        step_size=0.5, enable_bw=True)
      ccplanner.set_query(ccquery)
      res = ccplanner.solve(timeout=200)  
    print (time()-t)/rep_time

    from bimanual.planners.cc_planner_ms_connect import CCPlanner, CCQuery
    ccplanner = CCPlanner(Lshape, left_robot, right_robot, debug=False)
    t = time()
    for i in xrange(rep_time):
      ccquery = CCQuery(q_robots_start[0], q_robots_start[1], 
                          q_robots_goal[0], q_robots_goal[1],
                          q_robots_grasp[0], q_robots_grasp[1], 
                          T_obj_start, obj_translation_limits,
                          nn=2, step_size=0.5, enable_bw=True)
      ccplanner.set_query(ccquery)
      res = ccplanner.solve(timeout=200) 
    print (time()-t)/rep_time

    ccplanner.visualize_cctraj(ccquery.cctraj)


if __name__ == "__main__":
  demo = Comparison()
