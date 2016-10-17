#!/usr/bin/env python
import os
import numpy as np
import openravepy as orpy
from time import time
from IPython import embed

import sys
sys.path.append('../src/')
from utils import utils

class Benchmark(object):
  def __init__(self):
    np.set_printoptions(precision=10, suppress=True)
    
    # Read configuration parameters
    left_robot_name = 'left'
    right_robot_name = 'right'

    # Load OpenRAVE environment
    scene_file = '../xml/worlds/bimanual_setup.env.xml'
    env = orpy.Environment()
    # env.SetViewer('qtcoin')
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

    # Add collision checker
    collision_checker = orpy.RaveCreateCollisionChecker(env, 'ode')
    env.SetCollisionChecker(collision_checker)

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


    velocity_scale = 0.2
    rave_utils.scale_DOF_limits(left_robot, v=velocity_scale)
    rave_utils.scale_DOF_limits(right_robot, v=velocity_scale)

    orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)

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
    q_left_start, q_right_start = q_robots_start
    q_left_grasp, q_right_grasp = q_robots_grasp
    T_obj_start = Lshape.GetTransform()

    # # Simple
    # T_obj_goal = np.array(
    #   [[ 0.2418989753, -0.5117459199,  0.8243791598,  0.336411953 ],
    #    [ 0.9174005952,  0.3973257056, -0.0225484294, -0.1316559166],
    #    [-0.3160079646,  0.7617403738,  0.5655886926,  0.6580717564],
    #    [ 0.          ,  0.          ,  0.          ,  1.          ]])
    # Complex
    T_obj_goal = np.array(
      [[ 0.0582528963, -0.9964414827,  0.0609177455,  0.1181300282],
       [ 0.9983001464,  0.0582572434, -0.00170625  , -0.0844197497],
       [-0.0018487217,  0.0609135882,  0.9981413312,  0.9244756699],
       [ 0.          ,  0.          ,  0.          ,  1.          ]])

    # # Middle
    # T_obj_goal = np.array(
    #   [[ 0.347057442 , -0.7710350045,  0.5339065029,  0.191773653 ],
    #    [ 0.917400635 ,  0.3973256127, -0.0225484463, -0.1438052654],
    #    [-0.194749087 ,  0.4976317709,  0.8452428135,  0.8325030804],
    #    [ 0.          ,  0.          ,  0.          ,  1.          ]])

    q_robots_goal = utils.compute_bimanual_goal_configs(
                      [left_robot, right_robot], Lshape, q_robots_start,
                       q_robots_grasp, T_obj_start, T_obj_goal)

    # Start benchmarking

    rrt_type = {True:'Bi-RRT', False:'Single-RRT'}

    print '================================================================'
    print '                           original                             '
    print '================================================================'
    from cc_planner_original import CCPlanner, CCQuery
    ccplanner = CCPlanner(Lshape, [left_robot, right_robot], debug=False)
    for predict in (False, True):
      print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
      for enable_bw in (False, True):
        print '************************************************************'
        for nn in (-1, 1, 2):
          print '----------------------------------------------------------'
          for step_size in np.arange(0.3, 0.9, 0.1):
            ts=[]
            for i in xrange(100):
              t = time()
              ccquery = CCQuery(obj_translation_limits, q_robots_start, 
                q_robots_goal, q_robots_grasp, T_obj_start, nn=nn,
                step_size=step_size, predict=predict, 
                velocity_scale=velocity_scale, enable_bw=enable_bw)
              ccplanner.set_query(ccquery)
              res = ccplanner.solve(timeout=200)
              ts.append(time()-t)
            ts = np.array(ts)
            print "predict: {0}, rrt_type: {1}, nn: {2}, step_size: {3}, avg: [{4:.2f}], std2: {5:.2f}, std3:{6:.2f}, min:{7:.2f}, max:{8:.2f}".format(predict, rrt_type[enable_bw], nn, step_size,
              np.average(ts), np.std(ts), np.power(np.mean(np.abs(ts - ts.mean())**3), 1.0/3.0),
              np.min(ts), np.max(ts))

    print '================================================================'
    print '             restart:     restart_time = 3                      '
    print '================================================================'
    from cc_planner_restart import CCPlanner, CCQuery
    ccplanner = CCPlanner(Lshape, [left_robot, right_robot],
                          restart_time = 3, debug=False)
    for predict in (False, True):
      print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
      for enable_bw in (False, True):
        print '************************************************************'
        for nn in (-1, 1, 2):
          print '----------------------------------------------------------'
          for step_size in np.arange(0.3, 0.9, 0.1):
            ts=[]
            for i in xrange(100):
              t = time()
              ccquery = CCQuery(obj_translation_limits, q_robots_start, 
                q_robots_goal, q_robots_grasp, T_obj_start, nn=nn,
                step_size=step_size, predict=predict, 
                velocity_scale=velocity_scale, enable_bw=enable_bw)
              ccplanner.set_query(ccquery)
              res = ccplanner.solve(timeout=200)
              ts.append(time()-t)
            ts = np.array(ts)
            print "predict: {0}, rrt_type: {1}, nn: {2}, step_size: {3}, avg: [{4:.2f}], std2: {5:.2f}, std3:{6:.2f}, min:{7:.2f}, max:{8:.2f}".format(predict, rrt_type[enable_bw], nn, step_size,
              np.average(ts), np.std(ts), np.power(np.mean(np.abs(ts - ts.mean())**3), 1.0/3.0),
              np.min(ts), np.max(ts))

    print '================================================================'
    print '                         random                                 '
    print '================================================================'
    from cc_planner_random import CCPlanner, CCQuery
    ccplanner = CCPlanner(Lshape, [left_robot, right_robot], debug=False)
    for predict in (False, True):
      print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
      for enable_bw in (False, True):
        print '************************************************************'
        for nn in (-1, 1, 2):
          print '----------------------------------------------------------'
          for step_size in np.arange(0.3, 0.9, 0.1):
            ts=[]
            for i in xrange(100):
              t = time()
              ccquery = CCQuery(obj_translation_limits, q_robots_start,
                q_robots_goal, q_robots_grasp, T_obj_start, nn=nn, nr=3,
                step_size=step_size, predict=predict,
                velocity_scale=velocity_scale, enable_bw=enable_bw)
              ccplanner.set_query(ccquery)
              res = ccplanner.solve(timeout=200)
              ts.append(time()-t)
            ts = np.array(ts)
            print "predict: {0}, rrt_type: {1}, nn: {2}, step_size: {3}, avg: [{4:.2f}], std2: {5:.2f}, std3:{6:.2f}, min:{7:.2f}, max:{8:.2f}".format(predict, rrt_type[enable_bw], nn, step_size,
              np.average(ts), np.std(ts), np.power(np.mean(np.abs(ts - ts.mean())**3), 1.0/3.0),
              np.min(ts), np.max(ts))
        
    print '================================================================'
    print '                        multiextension                          '
    print '================================================================'
    from cc_planner_multiextension import CCPlanner, CCQuery
    ccplanner = CCPlanner(Lshape, [left_robot, right_robot], debug=False)
    for predict in (False, True):
      print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
      for enable_bw in (False, True):
        print '************************************************************'
        for nn in (-1, 1, 2):
          print '----------------------------------------------------------'
          for step_size in np.arange(0.3, 0.9, 0.1):
            ts=[]
            for i in xrange(100):
              t = time()
              ccquery = CCQuery(obj_translation_limits, q_robots_start, 
                q_robots_goal, q_robots_grasp, T_obj_start, nn=nn, ne=3, 
                step_size=step_size, predict=predict, 
                velocity_scale=velocity_scale, enable_bw=enable_bw)
              ccplanner.set_query(ccquery)
              res = ccplanner.solve(timeout=200)
              ts.append(time()-t)
            ts = np.array(ts)
            print "predict: {0}, rrt_type: {1}, nn: {2}, step_size: {3}, avg: [{4:.2f}], std2: {5:.2f}, std3:{6:.2f}, min:{7:.2f}, max:{8:.2f}".format(predict, rrt_type[enable_bw], nn, step_size,
              np.average(ts), np.std(ts), np.power(np.mean(np.abs(ts - ts.mean())**3), 1.0/3.0),
              np.min(ts), np.max(ts))

    print '================================================================'
    print '                          rrt-connect                           '
    print '================================================================'
    from cc_planner_connect import CCPlanner, CCQuery
    ccplanner = CCPlanner(Lshape, [left_robot, right_robot], debug=False)
    for nn in (-1, 1, 2):
      print '----------------------------------------------------------'
      for step_size in np.arange(0.2, 0.9, 0.1):
        ts=[]
        for i in xrange(100):
          t = time()
          ccquery = CCQuery(obj_translation_limits, q_robots_start, 
            q_robots_goal, q_robots_grasp, T_obj_start, nn=nn,
            step_size=step_size, velocity_scale=velocity_scale, 
            enable_bw=True)
          ccplanner.set_query(ccquery)
          res = ccplanner.solve(timeout=200)
          ts.append(time()-t)
        ts = np.array(ts)
        print "predict: {0}, rrt_type: {1}, nn: {2}, step_size: {3}, avg: [{4:.2f}], std2: {5:.2f}, std3:{6:.2f}, min:{7:.2f}, max:{8:.2f}".format(predict, rrt_type[enable_bw], nn, step_size,
          np.average(ts), np.std(ts), np.power(np.mean(np.abs(ts - ts.mean())**3), 1.0/3.0),
          np.min(ts), np.max(ts))

    print '================================================================'
    print '                         cc_planner_ms                          '
    print '================================================================'
    q_left_goal, q_right_goal = utils.compute_bimanual_goal_configs(
                                [left_robot, right_robot], Lshape, 
                                [q_left_start, q_right_start], 
                                [q_left_grasp, q_right_grasp],
                                T_obj_start, T_obj_goal)
    from cc_planner_ms import CCPlanner, CCQuery
    ccplanner = CCPlanner(Lshape, left_robot, right_robot, debug=False)
    for nn in (1, 2, 5):
      print '----------------------------------------------------------'
      for step_size in np.arange(0.3, 0.9, 0.1):
        ts=[]
        for i in xrange(100):
          t = time()
          ccquery = CCQuery(q_left_start, q_right_start, 
            q_left_goal, q_right_goal, q_left_grasp, q_right_grasp,
            T_obj_start, obj_translation_limits=obj_translation_limits,
            nn=nn, step_size=step_size,
            velocity_scale=velocity_scale, enable_bw=True)
          ccplanner.set_query(ccquery)
          res = ccplanner.solve(timeout=200)
          ts.append(time()-t)
        ts = np.array(ts)
        print('rrt_type: {0}, nn: {1}, step_size: {2}, avg: [{3:.2f}], '
              'std2: {4:.2f}, std3:{5:.2f}, min:{6:.2f}, max:{7:.2f}'
              .format(rrt_type[True], nn, step_size, np.average(ts), 
              np.std(ts), np.power(np.mean(np.abs(ts - ts.mean())**3), 
              1.0/3.0), np.min(ts), np.max(ts)))


if __name__ == "__main__":
  demo = Benchmark()
