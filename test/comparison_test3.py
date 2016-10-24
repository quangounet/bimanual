#!/usr/bin/env python
'''
This is a very interesting case I found occasionally.
When planning use cc_planner, using seeds[1,2] can find solution in a few seconds, but without seeds it can never find solution.
When planning use cc_planner_ms, if take the left robot as master, it is like crap. But if take the right robot as master, using seeds[1,2] solves in a few seconds, and without seeds is even much faster.

This narrow passage problem can also be solved using regrasp planner.
It is a typical example showing that when start and goal closed-chain configurations are within the same class, it can still take ages to solve without regrasping if only sampling in SE3 space.
'''
#!/usr/bin/env python
import rospy
import os
import numpy as np
import openravepy as orpy
# Planner
import sys
sys.path.append('../src/')
from utils import utils
# Utils
import ikea_openrave.utils as rave_utils
import ikea_manipulation.utils as manip_utils

from IPython import embed

class ClosedChainMotion(object):
  def __init__(self):
    # Generic configuration
    np.set_printoptions(precision=10, suppress=True)
    
    # Load OpenRAVE environment
    scene_file = '../xml/worlds/bimanual_ikea_assembly_2.env.xml'
    env = orpy.Environment()
    env.SetViewer('qtcoin')
    env.Load(scene_file)

    # Retrive robot and objects
    left_robot = env.GetRobot('denso_left')
    right_robot = env.GetRobot('denso_right')
    upper_frame = env.ReadKinBodyXMLFile('../xml/objects/right_frame.kinbody.xml')
    lower_frame = env.ReadKinBodyXMLFile('../xml/objects/left_frame.kinbody.xml')
    long_stick = env.ReadKinBodyXMLFile('../xml/objects/ikea_long_stick.kinbody.xml')
    stick = env.ReadKinBodyXMLFile('../xml/objects/ikea_stick.kinbody.xml')
    back = env.ReadKinBodyXMLFile('../xml/objects/ikea_back.kinbody.xml')
    # Correct robots transformation
    T_left_robot = np.array([[ 1.   ,  0.   ,  0.   ,  0    ],
                             [ 0.   ,  1.   ,  0.   , -0.536],
                             [ 0.   ,  0.   ,  1.   ,  0.005],
                             [ 0.   ,  0.   ,  0.   ,  1.   ]])
    T_right_robot = np.array([[ 1.   ,  0.   ,  0.   , 0.012],
                              [ 0.   ,  1.   ,  0.   , 0.536],
                              [ 0.   ,  0.   ,  1.   , 0    ],
                              [ 0.   ,  0.   ,  0.   , 1.   ]])
    T_upper_frame = np.array(
      [[-0.5819262528,  0.0512289085, -0.8116264136,  0.083908692 ],
       [-0.8130147964, -0.013080617 ,  0.5820960731,  0.1336675584],
       [ 0.0192035722,  0.99860127  ,  0.049261815 ,  0.0164849497],
       [ 0.          ,  0.          ,  0.          ,  1.          ]])
    T_lower_frame = np.array(
      [[ 0.  ,  1.  ,  0.  , -0.19],
       [-1.  ,  0.  ,  0.  ,  0.3],
       [ 0.  ,  0.  ,  1.  ,  0.02 ],
       [ 0.  ,  0.  ,  0.  ,  1.  ]])
    T_back = np.array(
      [[ 0.9783646386,  0.2068635107, -0.0031814933,  0.3711627996],
       [ 0.2050614999, -0.9716515188, -0.1176567351,  0.2907784802],
       [-0.0274301881,  0.1144587873, -0.9930492288,  0.3069383067],
       [ 0.          ,  0.          ,  0.          ,  1.          ]])   
    T_long_stick = np.array(
      [[-0.0430562132,  0.0000002264,  0.9990726513,  0.2171998447],
       [ 0.996804047 ,  0.0673518022,  0.0429584298, -0.1145144928],
       [-0.0672893339,  0.9977292893, -0.0029001393,  0.2079720313],
       [ 0.          ,  0.          ,  0.          ,  1.          ]])
    T_stick = np.array(
      [[ 0.0567546182, -0.0054029418,  0.9983735381,  0.2342538661],
       [ 0.9952152233,  0.0799668653, -0.0561423171,  0.2574529421],
       [-0.0795334686,  0.9967828794,  0.0099155788,  0.1714758214],
       [ 0.          ,  0.          ,  0.          ,  1.          ]])

    q_left_start = np.array([ 1.0892011176,  0.3679566286,  2.0465749564, -4.5061957951, 1.4560078247,  0.7342313871])
    q_right_start = np.array([-0.7233691649,  0.1028194532,  1.2388610257,  1.598951087 , -1.4506669078, -0.3         ])


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

    with env:
      env.Add(upper_frame)
      env.Add(lower_frame)
      env.Add(back)
      env.Add(stick)
      env.Add(long_stick)
      left_robot.SetTransform(T_left_robot)
      right_robot.SetTransform(T_right_robot)
      upper_frame.SetTransform(T_upper_frame)
      lower_frame.SetTransform(T_lower_frame)
      back.SetTransform(T_back)
      stick.SetTransform(T_stick)
      long_stick.SetTransform(T_long_stick)
      left_robot.SetActiveDOFValues(q_left_start)
      right_robot.SetActiveDOFValues(q_right_start)

    left_gmodel = orpy.databases.grasping.GraspingModel(left_robot, upper_frame)
    if not left_gmodel.load():
      gmodel.autogenerate()

    embed()
    exit(0)

    ################## closed chain planning ###################
    orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)

    obj_translation_limits =  [[2, 2, 2], [-2, -2, -1]]
    q_robots_grasp = [left_robot.GetDOFValues()[-1],
                      right_robot.GetDOFValues()[-1]]
    #################### motion 1 #####################
    q_robots_start = [left_robot.GetActiveDOFValues(),
                      right_robot.GetActiveDOFValues()]
    T_obj_start = upper_frame.GetTransform()

    T_obj_goal = np.array(
      [[-0.3250499015,  0.0528300586, -0.9442200731, -0.1957464218],
       [-0.9455018687,  0.0021198501,  0.3256097703,  0.1550759375],
       [ 0.0192035882,  0.9986012673,  0.049261862 ,  0.0164838415],
       [ 0.          ,  0.          ,  0.          ,  1.          ]])

    q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], upper_frame, q_robots_start,
                    q_robots_grasp, T_obj_start, T_obj_goal,seeds=[1,2])

    q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], upper_frame, q_robots_start,
                    q_robots_grasp, T_obj_start, T_obj_goal)

    import ikea_planner.cc_planner_connect as ccp    
    ccplanner = ccp.CCPlanner(upper_frame, [left_robot, right_robot], debug=False)
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start,
                          q_robots_goal, q_robots_grasp, T_obj_start, nn=2,
                          step_size=0.2, enable_bw=True)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=10)


    from ikea_planner.cc_planner_ms_connect import CCPlanner, CCQuery
    ccplanner = CCPlanner(upper_frame, right_robot, left_robot, debug=False)
    ccquery = CCQuery(q_robots_start[1], q_robots_start[0], 
                        q_robots_goal[1], q_robots_goal[0],
                        q_robots_grasp[1], q_robots_grasp[0], 
                        T_obj_start, obj_translation_limits,
                        nn=2, step_size=0.2, enable_bw=True)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=20)


    import cc_planner_regrasp as ccp
    ccplanner = ccp.CCPlanner(upper_frame, [left_robot, right_robot], debug=False)
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, T_obj_start, nn=2, 
                          step_size=0.2, enable_bw=True)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=20)




    ccplanner.shortcut(ccquery, maxiter=60)
    ccplanner.visualize_cctraj(ccquery.cctraj, speed=1)


if __name__ == "__main__":
  demo = ClosedChainMotion()
