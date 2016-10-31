#!/usr/bin/env python
import pickle
import numpy as np
import openravepy as orpy
import ikea_openrave.utils as rave_utils
from IPython import embed

import sys
sys.path.append('../src/')
from utils import utils

if __name__ == "__main__":
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
                            [ 0.   ,  0.   ,  1.   , 0],
                            [ 0.   ,  0.   ,  0.   , 1.   ]])
  T_upper_frame = np.array(
    [[-1.          ,  0.          ,  0.          , -0.2375028133],
     [-0.          ,  1.          , -0.          , -0.3633791208],
     [-0.          , -0.          , -1.          ,  0.0336048044],
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
  q_left_start = np.array(
    [ 2.6211708721,  1.1921830071,  0.948171676 , -3.2192204259,
      -1.0747315268,  1.0970359988])
  q_right_start = np.array(
    [ 0.7080560322, -0.7356266017, -1.8234738306, -0.1000137583,
      -0.6611108429,  1.0098724261])

  manip_name = 'denso_ft_sensor_gripper'
  left_manip = left_robot.SetActiveManipulator(manip_name)
  right_manip = right_robot.SetActiveManipulator(manip_name)
  left_basemanip = orpy.interfaces.BaseManipulation(left_robot)
  left_taskmanip = orpy.interfaces.TaskManipulation(left_robot)
  right_basemanip = orpy.interfaces.BaseManipulation(right_robot)
  right_taskmanip = orpy.interfaces.TaskManipulation(right_robot)

  rave_utils.disable_gripper([left_robot, right_robot])
  rave_utils.load_IK_model([left_robot, right_robot])

  # Add collision checker
  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'ode'))

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

  velocity_scale = 0.5
  rave_utils.scale_DOF_limits(left_robot, v=velocity_scale)
  rave_utils.scale_DOF_limits(right_robot, v=velocity_scale)


  obj_translation_limits =  [[2, 2, 2], [-2, -2, -1]]
  q_robots_grasp = [left_robot.GetDOFValues()[-1],
                    right_robot.GetDOFValues()[-1]]
  #################### motion 1 #####################
  q_robots_start = [left_robot.GetActiveDOFValues(),
                    right_robot.GetActiveDOFValues()]
  T_obj_start = upper_frame.GetTransform()
  T_obj_goal = np.array(
    [[ 0.0040964853,  0.0491919345, -0.9987809431, -0.2837391496],
     [-0.9998072013,  0.01938201  , -0.0031460922,  0.1217863202],
     [ 0.0192036199,  0.9986012674,  0.0492618485,  0.0164831764],
     [ 0.          ,  0.          ,  0.          ,  1.          ]])
  q_robots_goal = utils.compute_bimanual_goal_configs(
                  [left_robot, right_robot], upper_frame, q_robots_start,
                  q_robots_grasp, T_obj_start, T_obj_goal)

  embed()
  exit(0)

  import cc_planner as ccp    
  ccplanner = ccp.CCPlanner(upper_frame, [left_robot, right_robot], debug=False)
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start,
                        q_robots_goal, q_robots_grasp, T_obj_start, nn=3,
                        step_size=0.1, velocity_scale=velocity_scale, 
                        enable_bw=True)
  ccplanner.set_query(ccquery)
  with open('trajs.pkl', 'rb') as f:
    traj = pickle.load(f)[0]
  ccquery.cctraj = traj
  ccplanner.shortcut(ccquery)