import openravepy as orpy
import numpy as np
import logging
_loglevel = logging.DEBUG
logging.basicConfig(format='[%(levelname)s] [%(name)s: %(funcName)s] %(message)s', level=_loglevel)
_log = logging.getLogger(__name__)


def plan_transit_motion(robot, q_start, q_goal, pregrasp_start=True, pregrasp_goal=True, ntrials=1):
  """
  Plan a transit motion (non-grasping motion) for robot moving from q_start to q_goal.

  @type     robot: openravepy.Robot
  @param    robot: The robot to plan this motion for
  @type   q_start: numpy.ndarray
  @param  q_start: Start configuration of the robot
  @type    q_goal: numpy.ndarray
  @param   q_goal: Goal configuration of the robot
  @type   pregrasp_start: bool
  @param  pregrasp_start: Indicates if pregrasp motion at the beginning of the trajectory 
                          should be planned
  @type    pregrasp_goal: bool
  @param   pregrasp_goal: Indicates if pregrasp motion at the end of the trajectory should be planned

  @rtype: oepnravepy.Trajectory
  @return: An OpenRAVE trajectory for the robot moving from q_start to q_goal
  """

  # MoveHandStraight parameters
  minsteps = 1
  maxsteps = 10
  steplength = 0.005
  execute = False

  env = robot.GetEnv()
  manip = robot.GetActiveManipulator()
  basemanip = orpy.interfaces.BaseManipulation(robot)
  q_original = robot.GetActiveDOFValues()

  traj_pregrasp_start = None
  traj_pregrasp_goal = None
  with robot:
    # Plan pregrasp motion at the start configuration
    if pregrasp_start:
      robot.SetActiveDOFValues(q_start)
      eematrix = manip.GetTransform()
      direction = -eematrix[0:3, 2]
      try:
        traj_pregrasp_start = basemanip.MoveHandStraight(direction, minsteps=minsteps, maxsteps=maxsteps, steplength=steplength, starteematrix=eematrix, execute=execute, outputtrajobj=True)
      except:
        _log.info("Caught an exception in MoveHandStraight (pregrasp_start).")
        traj_pregrasp_start = None
        # return None
      if traj_pregrasp_start is None:
        _log.info("MoveHandStraight failed (pregrasp_start).")
        new_q_start = q_start
        pregrasp_start = False
      else:
        new_q_start = traj_pregrasp_start.GetWaypoint(traj_pregrasp_start.GetNumWaypoints() -  1, traj_pregrasp_start.GetConfigurationSpecification().GetGroupFromName("joint_values"))
        robot.SetActiveDOFValues(new_q_start)
        if robot.CheckSelfCollision():
          new_q_start = q_start
          pregrasp_start = False
        else:
          orpy.planningutils.RetimeActiveDOFTrajectory(traj_pregrasp_start, robot, hastimestamps=False, maxvelmult=0.9, maxaccelmult=0.81, plannername="parabolictrajectoryretimer")      
    else:
      new_q_start = q_start

    # Plan pregrasp motion at the goal configuration
    if pregrasp_goal:
      robot.SetActiveDOFValues(q_goal)
      eematrix = manip.GetTransform()
      direction = -eematrix[0:3, 2]
      try:
        traj_pregrasp_goal = basemanip.MoveHandStraight(direction, minsteps=minsteps, maxsteps=maxsteps, steplength=steplength, starteematrix=eematrix, execute=execute, outputtrajobj=True)
      except:
        _log.info("Caught an exception in MoveHandStraight (pregrasp_goal).")
        traj_pregrasp_goal = None
        # return None
      if traj_pregrasp_goal is None:
        _log.info("MoveHandStraight failed (pregrasp_goal).")
        new_q_goal = q_goal
        pregrasp_goal = False
      else:
        new_q_goal = traj_pregrasp_goal.GetWaypoint(traj_pregrasp_goal.GetNumWaypoints() -  1, traj_pregrasp_goal.GetConfigurationSpecification().GetGroupFromName("joint_values"))
        robot.SetActiveDOFValues(new_q_goal)
        if robot.CheckSelfCollision():
          new_q_goal = q_goal
          pregrasp_goal = False
        else:
          traj_pregrasp_goal_rev = orpy.planningutils.ReverseTrajectory(traj_pregrasp_goal)
          orpy.planningutils.RetimeActiveDOFTrajectory(traj_pregrasp_goal_rev, robot, hastimestamps=False, maxvelmult=0.9, maxaccelmult=0.81, plannername="parabolictrajectoryretimer")
          traj_pregrasp_goal = traj_pregrasp_goal_rev
    else:
      new_q_goal = q_goal

    _log.info("new_q_start = np.{0}".format(new_q_start.__repr__()))
    _log.info("new_q_goal = np.{0}".format(new_q_goal.__repr__()))

    # Plan a motion connecting new_q_start and new_q_goal
    try:
      # Using MoveActiveJoints sometimes has problems with jittering. As a fix, we plan using this birrt + parabolicsmoother
      for i in xrange(ntrials):
        planner = orpy.RaveCreatePlanner(env, 'birrt')
        params = orpy.Planner.PlannerParameters()
        params.SetRobotActiveJoints(robot) # robot's active DOFs must be correctly set before calling this function
        params.SetInitialConfig(new_q_start)
        params.SetGoalConfig(new_q_goal)
        params.SetMaxIterations(10000)
        params.SetRandomGeneratorSeed(i)
        extraparams = '<_postprocessing planner="parabolicsmoother2"><_nmaxiterations>100</_nmaxiterations></_postprocessing>'
        params.SetExtraParameters(extraparams)
        planner.InitPlan(robot, params)
        traj_connect = orpy.RaveCreateTrajectory(env, '')
        res = planner.PlanPath(traj_connect)
        if not (res == orpy.PlannerStatus.HasSolution):
          _log.info("Trial {0}: planning failed.".format(i + 1))
          continue
        else:
          break
    except Exception as e:
      _log.info("Caught an exception ({0}) in MoveActiveJoints.".format(e))
      return None
    if traj_connect is None:
      _log.info("Planner failed after {0} trials.".format(ntrials))
      return None

    trajs_list = [traj for traj in [traj_pregrasp_start, traj_connect, traj_pregrasp_goal] if traj is not None]
    final_traj = combine_openrave_trajectories(trajs_list)
    return final_traj

  
def combine_openrave_trajectories(trajs_list):
  """
  Combine trajectories in the given list into one single OpenRAVE trajectory.

  @type   trajs_list: list of openravepy.Trajectory
  @param  trajs_list: Trajectories to be combined

  @rtype: oepnravepy.Trajectory
  @return: An OpenRAVE trajectory which is a concatenation of trajectories in trajs_list.

  """
  # Get one configuration specification from the first trajectory and convert configspec
  # of everyone else to be the same.
  spec = trajs_list[0].GetConfigurationSpecification()
  for traj in trajs_list[1:]:
    orpy.planningutils.ConvertTrajectorySpecification(traj, spec)

  traj_all = orpy.RaveCreateTrajectory(trajs_list[0].GetEnv(), "")
  cloning_option = 0
  traj_all.Clone(trajs_list[0], cloning_option)
  for traj in trajs_list[1:]:
    for iwaypoint in xrange(traj.GetNumWaypoints()):
      waypoint = traj.GetWaypoint(iwaypoint)
      traj_all.Insert(traj_all.GetNumWaypoints(), waypoint)

  return traj_all

  
    
