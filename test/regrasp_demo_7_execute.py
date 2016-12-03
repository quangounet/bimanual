#!/usr/bin/env python
import rospy
import os
import numpy as np
import openravepy as orpy
import cPickle as pickle
# Planner
from bimanual.utils import utils
from bimanual.utils import placement_utils as putils
import pymanip.planningutils.utils as pymanip_utils
from pymanip.planningutils import myobject, intermediateplacement, staticequilibrium
# Utils
from criros.utils import read_parameter
import ikea_openrave.utils as rave_utils
import ikea_manipulation.utils as manip_utils
# Controllers
from denso_control.controllers import JointPositionController
from netft_control.controller import FTSensor
from robotiq_control.controller import Robotiq

import os.path as path
model_path = path.abspath(path.join(path.dirname(__file__), "../xml"))

from IPython import embed


class ClosedChainMotion(object):
  def __init__(self):
    # Generic configuration
    np.set_printoptions(precision=10, suppress=True)
    
    # Read configuration parameters
    left_robot_name = 'left'
    right_robot_name = 'right'

    # Load OpenRAVE environment
    scene_file = model_path + '/worlds/bimanual_setup_regrasp.env.xml'
    env = orpy.Environment()
    env.SetViewer('qtcoin')
    
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
    T_table_t = np.array([[ 1.,  0.,  0.,  -0.005],
                        [ 0.,  1.,  0.,  0.],
                        [ 0.,  0.,  1.,  0.012],
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

    rave_utils.disable_gripper([left_robot, right_robot])
    rave_utils.load_IK_model([left_robot, right_robot])
    
    # Create manipulation tasks  
    left_basemanip = orpy.interfaces.BaseManipulation(left_robot)
    left_taskmanip = orpy.interfaces.TaskManipulation(left_robot)
    right_basemanip = orpy.interfaces.BaseManipulation(right_robot)
    right_taskmanip = orpy.interfaces.TaskManipulation(right_robot)

    js_rate = read_parameter('/%s/joint_state_controller/publish_rate'
                             % right_robot_name, 125.0)
    T = 1. / js_rate

    left_posController = JointPositionController(left_robot_name)
    right_posController = JointPositionController(right_robot_name)

    # Update robot states in openrave
    manip_utils.update_rave_robots([left_robot, right_robot],
                                   [left_posController, right_posController])

    left_gripper = Robotiq(left_robot_name)
    right_gripper = Robotiq(right_robot_name)
    left_gripper.open()
    right_gripper.open()
    right_ft_sensor = FTSensor(right_robot_name)

    rave_utils.scale_DOF_limits(left_robot, v=0.15)
    rave_utils.scale_DOF_limits(right_robot, v=0.15)

    ############### Move to pre-grasp position ###############
    qgrasp_left = [5, 3, 1, 0]
    T_left_gripper = pymanip_utils.ComputeTGripper2(
                      frame, qgrasp_left[0], qgrasp_left)
    q_left_start = left_manip.FindIKSolutions(T_left_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[0]

    qgrasp_right = [4, 0, 1, 0]
    T_right_gripper = pymanip_utils.ComputeTGripper2(
                        frame, qgrasp_right[0], qgrasp_right)
    q_right_start = right_manip.FindIKSolutions(T_right_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[0]

    # left_robot.SetActiveDOFValues(q_left_start)
    # right_robot.SetActiveDOFValues(q_right_start)

    left_traj = left_basemanip.MoveActiveJoints(goal=q_left_start, 
                                                outputtrajobj=True)
    manip_utils.execute_openrave_traj(left_robot, left_posController,
                                      left_traj, T)

    right_traj = right_basemanip.MoveActiveJoints(goal=q_right_start,
                                                  outputtrajobj=True)
    manip_utils.execute_openrave_traj(right_robot, right_posController,
                                      right_traj, T)

    embed()
    exit(0)

    rospy.sleep(18)

    rospy.sleep(0.4)
    left_gripper.close()
    left_taskmanip.CloseFingers()
    left_robot.WaitForController(0) 

    rospy.sleep(0.4)
    wrench_offset = right_ft_sensor.get_compensated_wrench()
    right_gripper.close()
    right_taskmanip.CloseFingers()
    right_robot.WaitForController(0)


    # manip_utils.update_rave_robots([left_robot, right_robot],
    #                                [left_posController, right_posController])

    ################### closed chain planning ###################
    velocity_scale = 0.5
    rave_utils.scale_DOF_limits(left_robot, v=velocity_scale)
    rave_utils.scale_DOF_limits(right_robot, v=velocity_scale)

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


    import bimanual.planners.cc_planner_regrasp_placement as ccp 
    ccplanner = ccp.CCPlanner(frame, p_frame, [left_robot, right_robot], 
                              plan_regrasp=True, debug=False)
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, qgrasps,
                          T_obj_start, nn=2, step_size=0.4, 
                          velocity_scale=velocity_scale, fmax=100, mu=0.3, 
                          regrasp_limit=3)
    ccplanner.set_query(ccquery)
    # res = ccplanner.solve(timeout=50)

    # ccquery.serialize()
    # with open('demo7_2.pkl', 'wb') as f:
    #   pickle.dump(ccquery, f)
    with open('demo7_2.pkl', 'rb') as f:
      ccquery = pickle.load(f)
      ccquery.deserialize(env)

    ccplanner.shortcut(ccquery, maxiters=[300, 200])
    ccplanner.visualize_cctraj(ccquery.cctraj, speed=10)

    manip_utils.execute_ccregrasptraj_compliance_transition(
                    [left_robot, right_robot],
                    [left_posController, right_posController],
                    [left_gripper, right_gripper],
                    frame, ccquery, right_ft_sensor, wrench_offset, 
                    [left_taskmanip, right_taskmanip], force_threshold=8.0)

if __name__ == "__main__":
  # Initialize the node
  node_name = os.path.splitext(os.path.basename(__file__))[0]
  rospy.init_node(node_name)
  rospy.loginfo('Starting [%s] node' % node_name)
  demo = ClosedChainMotion()
  rospy.loginfo('Shuting down [%s] node' % node_name)