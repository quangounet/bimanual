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
  scene_file = '../xml/worlds/bimanual_setup.env.xml'
  env = orpy.Environment()
  env.SetViewer('qtcoin')
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
    env.Remove(env.GetKinBody('table'))

  # Add collision checker
  env.SetCollisionChecker(orpy.RaveCreateCollisionChecker(env, 'ode'))
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
  left_robot.SetDOFValues(
    [ 0.8041125559,  0.9405270996,  1.0073165866,  3.1415926536, 
     -1.1937489674,  2.3749088827,  0.4560000086])
  right_robot.SetDOFValues(
    [-0.9816436375,  0.7763915147,  1.2904895262, -3.1415926536, 
     -1.0747116127,  2.1599490161,  0.4560000086])

  ################## closed chain planning ###################

  obj_translation_limits =  [[0.9, 0.5, 1.2], [-0.5, -0.4, -0.4]]
  q_robots_start = [left_robot.GetActiveDOFValues(),
                    right_robot.GetActiveDOFValues()]
  q_robots_grasp = [left_robot.GetDOFValues()[-1],
                    right_robot.GetDOFValues()[-1]]
  T_obj_start = Lshape.GetTransform()
  T_obj_goal = np.array(
    [[ 0.          , -0.6233531712, -0.7819404223,  0.0683513284],
     [ 1.          ,  0.          ,  0.          , -0.1750002205],
     [ 0.          , -0.7819404223,  0.6233531712,  0.84509027  ],
     [ 0.          ,  0.          ,  0.          ,  1.          ]])

  q_robots_goal = utils.compute_bimanual_goal_configs(
                    [left_robot, right_robot], Lshape, q_robots_start,
                     q_robots_grasp, T_obj_start, T_obj_goal)

  embed()
  exit(0)

  import cc_planner_regrasp as ccp 
  ccplanner = ccp.CCPlanner(Lshape, [left_robot, right_robot], 
                            plan_regrasp=True, debug=False)
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                        q_robots_goal, q_robots_grasp, T_obj_start, nn=2, 
                        step_size=0.5, regrasp_limit=2)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=100)

  import cc_planner_reform as ccp 
  ccplanner = ccp.CCPlanner(Lshape, [left_robot, right_robot], 
                            plan_regrasp=True, debug=False)
  ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                        q_robots_goal, q_robots_grasp, T_obj_start, nn=2, 
                        step_size=0.4, regrasp_limit=100)
  ccplanner.set_query(ccquery)
  res = ccplanner.solve(timeout=15)

  rep = 50
  from time import time
  import cc_planner_regrasp as ccp 
  ccplanner = ccp.CCPlanner(Lshape, [left_robot, right_robot], 
                            plan_regrasp=False, debug=False)
  t2 = time() 
  i = 0
  while i < rep:
    print 'i: ', i
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, T_obj_start, nn=2, 
                          step_size=0.5, regrasp_limit=2)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=30)
    if res:
      i += 1
  t2_end = time()

  import cc_planner_reform as ccp 
  ccplanner = ccp.CCPlanner(Lshape, [left_robot, right_robot], 
                            plan_regrasp=False, debug=False)
  t3 = time() 
  i = 0
  while i < rep:
    print 'i: ', i
    ccquery = ccp.CCQuery(obj_translation_limits, q_robots_start, 
                          q_robots_goal, q_robots_grasp, T_obj_start, nn=2, 
                          step_size=0.5, regrasp_limit=2)
    ccplanner.set_query(ccquery)
    res = ccplanner.solve(timeout=30)
    if res:
      i += 1
  t3_end = time()

  print (t2_end-t2)/rep
  print (t3_end-t3)/rep


  ccplanner.shortcut(ccquery, maxiters=[40, 50])
  ccplanner.visualize_cctraj(ccquery.cctraj, speed=5)



left_robot.SetActiveDOFValues(np.array([1.3286037675828692, 1.61926880978681, -0.78215209411198694, 4.0283431951051281, 0.93621987466452217, -1.0108507059372083]))

right_robot.SetActiveDOFValues([-0.89919974486785126,  0.87162093934540275,  1.2360258804531066,  -0.47815594937409694,  -1.5946827762340383,  0.090781819060121921])

q_robot = np.array([2.2423920808053284,  -2.0633626740460622,  1.2360199965182757,  -4.1702100454797009,  -0.56691154243021036,  1.0538235669215026])

basemanip = right_basemanip
robot = right_robot
c = robot.GetController()

Lshape.SetTransform(np.array(
  [[ 0.496487317 , -0.5973714785,  0.6297996989,  0.3379715483],
   [ 0.7947580193,  0.6046029621, -0.0530560928, -0.1244274658],
   [-0.3490845669,  0.5268800384,  0.7749434755,  0.6898874894],
   [ 0.          ,  0.          ,  0.          ,  1.          ]]))

left_taskmanip.ReleaseFingers()

right_taskmanip.ReleaseFingers()


from time import time
t = time()
traj = basemanip.MoveActiveJoints(goal=q_robot, execute=False, outputtrajobj=True)
print time()-t


t = time()
params = orpy.Planner.PlannerParameters()
params.SetRobotActiveJoints(robot)
params.SetGoalConfig(q_robot) # set goal to all ones
# params.SetExtraParameters("""<_postprocessing planner="linearsmoother">
#     <_nmaxiterations>1</_nmaxiterations>
# </_postprocessing>""")
params.SetExtraParameters("<_postprocessing></_postprocessing>")
planner=orpy.RaveCreatePlanner(env,'birrt')
planner.InitPlan(robot, params)
traj = orpy.RaveCreateTrajectory(env,'')
planner.PlanPath(traj)
print time()-t

c.SetPath(traj)
robot.SetActiveDOFValues([-0.89919974486785126,  0.87162093934540275,  1.2360258804531066,  -0.47815594937409694,  -1.5946827762340383,  0.090781819060121921])



###############################################


left_robot.SetActiveDOFValues(np.array([1.1179185740387636, 0.9386752793904507, 0.80032749713981743, -2.1946447274984533, 1.0806177078710155, 1.5873185270739358]))

right_robot.SetActiveDOFValues(np.array([-1.3452288403551229, 0.38678944640524932, 1.012264320407571, -1.1359528270330019, -1.476739947025685, 2.8216469080645283]))

q_robot = np.array([1.1179190426593697, 1.6992327458331573, -0.73367709448158691, 4.6567772140613615, 0.79973950741550781, 0.67549521798187973])

basemanip = left_basemanip
robot = left_robot
c = robot.GetController()

Lshape.SetTransform(np.array(
      [[-0.4660339977, -0.0091527126,  0.8847194702,  0.4715947041],
       [ 0.3627313889, -0.9140248958,  0.1816161591,  0.0891404323],
       [ 0.8069933411,  0.4055548269,  0.4292866523,  0.538749835 ],
       [ 0.          ,  0.          ,  0.          ,  1.          ]]))

left_taskmanip.ReleaseFingers()

right_taskmanip.ReleaseFingers()


from time import time
t = time()
try: traj = basemanip.MoveActiveJoints(goal=q_robot, execute=False, outputtrajobj=True)
except:
  print time()-t


t = time()
params = orpy.Planner.PlannerParameters()
params.SetRobotActiveJoints(robot)
params.SetGoalConfig(q_robot) # set goal to all ones
# forces parabolic planning with 40 iterations
# params.SetExtraParameters("""<_postprocessing planner="parabolicsmoother">
#     <_nmaxiterations>40</_nmaxiterations>
# </_postprocessing>""")
params.SetExtraParameters("""<_nmaxiterations>5000</_nmaxiterations><_postprocessing></_postprocessing>""")
planner=orpy.RaveCreatePlanner(env,'birrt')
planner.InitPlan(robot, params)
traj = orpy.RaveCreateTrajectory(env,'')
planner.PlanPath(traj)
print time()-t

c.SetPath(traj)
robot.SetActiveDOFValues(np.array([1.1179185740387636, 0.9386752793904507, 0.80032749713981743, -2.1946447274984533, 1.0806177078710155, 1.5873185270739358]))