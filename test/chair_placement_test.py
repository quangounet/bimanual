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
  scene_file = model_path + '/worlds/bimanual_setup2.env.xml'
  env = orpy.Environment()
  env.SetViewer('qtcoin')
  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'fcl_'))
  
  table = orpy.RaveCreateKinBody(env, '')
  table.InitFromBoxes(np.array([[0.0, 0.0, -0.02001, 0.5, 1.0, 0.02]]))
  table.SetName('table')
  table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor([1, 0.8, 0.6])
  T_table = np.array([[-1.0,  0.0,  0.0,  0.6385794],
                     [ 0.0, -1.0,  0.0,  0.0],
                     [ 0.0,  0.0,  1.0,  0.1362327],
                     [ 0.0,  0.0,  0.0,  1.0]])
  env.Add(table)
  table.SetTransform(T_table)

  chair = env.ReadKinBodyXMLFile('/home/zhouxian/git/pymanip/pymanip/models/chair.xml')
  T_chair = np.array(
    [[ -2.77555756e-17,  -1.00000000e+00,   0.00000000e+00,   2.88579321e-01],
     [ -5.96515267e-01,  -2.77555756e-17,   8.02601730e-01,  -2.93648541e-02],
     [ -8.02601730e-01,   0.00000000e+00,  -5.96515267e-01,   3.80555246e-01],
     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
  env.Add(chair)
  chair.SetTransform(T_chair)

  myObject = putils.create_placement_object(chair, env, T_rest=T_table)

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

  with env:
    left_robot.SetTransform(T_left_robot)
    right_robot.SetTransform(T_right_robot)


  manip_name = 'denso_ft_sensor_gripper'
  left_manip = left_robot.SetActiveManipulator(manip_name)
  right_manip = right_robot.SetActiveManipulator(manip_name)
  left_basemanip = orpy.interfaces.BaseManipulation(left_robot)
  left_taskmanip = orpy.interfaces.TaskManipulation(left_robot)
  right_basemanip = orpy.interfaces.BaseManipulation(right_robot)
  right_taskmanip = orpy.interfaces.TaskManipulation(right_robot)

  rave_utils.disable_gripper([left_robot, right_robot])
  rave_utils.load_IK_model([left_robot, right_robot])

  embed()
  exit(0)
  # Choose one grasp for testing
  qgrasp_l = [14, 2, 0, -0.05429823617575867]
  qgrasp_r = [5, 0, 2, -0.10022448427655264]

  T_left_gripper = pymanip_utils.ComputeTGripper2(chair, qgrasp_l[0], qgrasp_l)
  T_right_gripper = pymanip_utils.ComputeTGripper2(chair, qgrasp_r[0], qgrasp_r)
  left_robot.SetActiveDOFValues(left_manip.FindIKSolutions(T_left_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[0])
  right_robot.SetActiveDOFValues(right_manip.FindIKSolutions(T_right_gripper, orpy.IkFilterOptions.CheckEnvCollisions)[0])
  # Compute a feasible placement on the floor
  fmax = 100
  mu = 0.5
  placementType = 2 # placing on a vertex
  res = intermediateplacement.ComputeFeasibleClosePlacements([left_robot, right_robot], [qgrasp_l, qgrasp_r], chair, T_chair, T_table, fmax, mu, placementType=placementType, myObject=myObject)


