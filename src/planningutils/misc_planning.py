import openravepy as orpy
import numpy as np
import logging
_loglevel = logging.DEBUG
logging.basicConfig(format='[%(levelname)s] [%(name)s: %(funcName)s] %(message)s', level=_loglevel)
_log = logging.getLogger(__name__)


def plan_transit_motion(robot, q_start, q_goal, pregrasp_start=True, pregrasp_goal=True):
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
  with env:
    # Plan pregrasp motion at the start configuration
    if pregrasp_start:
      robot.SetActiveDOFValues(q_start)
      eematrix = manip.GetTransform()
      direction = -eematrix[0:3, 2]
      traj_pregrasp_start = None
      try:
        traj_pregrasp_start = basemanip.MoveHandStraight(direction, minsteps=minsteps, maxsteps=maxsteps, steplength=steplength, starteematrix=eematrix, execute=execute, outputtrajobj=True)
      except:
        _log.info("Caught an exception in MoveHandStraight (pregrasp_start).")
        return None
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
      traj_pregrasp_goal = None
      try:
        traj_pregrasp_goal = basemanip.MoveHandStraight(direction, minsteps=minsteps, maxsteps=maxsteps, steplength=steplength, starteematrix=eematrix, execute=execute, outputtrajobj=True)
      except:
        _log.info("Caught an exception in MoveHandStraight (pregrasp_goal).")
        return None
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
    else:
      new_q_goal = q_goal

    _log.info("new_q_start = {0}".format(new_q_start.__repr__()))
    _log.info("new_q_goal = {0}".format(new_q_goal.__repr__()))

    # Plan a motion connecting new_q_start and new_q_goal
    try:
      # Using MoveActiveJoints sometimes has problems with jittering. As a fix, we plan using this birrt + parabolicsmoother
      planner = orpy.RaveCreatePlanner(env, 'birrt')
      params = orpy.Planner.PlannerParameters()
      params.SetRobotActiveJoints(robot) # robot's active DOFs must be correctly set before calling this function
      params.SetInitialConfig(new_q_start)
      params.SetGoalConfig(new_q_goal)
      params.SetMaxIterations(10000)
      extraparams = '<_postprocessing planner="parabolicsmoother2"><_nmaxiterations>100</_nmaxiterations></_postprocessing>'
      params.SetExtraParameters(extraparams)
      planner.InitPlan(robot, params)
      traj_connect = orpy.RaveCreateTrajectory(env, '')
      res = planner.PlanPath(traj_connect)
      if not (res == orpy.PlannerStatus.HasSolution):
        _log.info("Planner failed.")
        return None
    except Exception as e:
      _log.info("Caught an exception ({0}) in MoveActiveJoints.".format(e))
      return None
    if traj_connect is None:
      _log.info("MoveActiveJoints failed.")
      return None

    # Combine all trajectory segments into one. Note that although all trajectories are currently
    # parabolic, they might have different ConfigurationSpecification. We need to fix this first
    # before combining all trajectories into a single trajectory.
    spec = traj_connect.GetConfigurationSpecification()
    if pregrasp_start and pregrasp_goal:
      orpy.planningutils.ConvertTrajectorySpecification(traj_pregrasp_start, spec)
      orpy.planningutils.ConvertTrajectorySpecification(traj_pregrasp_goal_rev, spec)
      trajlist = [traj_pregrasp_start, traj_connect, traj_pregrasp_goal_rev]
    elif pregrasp_start:
      orpy.planningutils.ConvertTrajectorySpecification(traj_pregrasp_start, spec)
      trajlist = [traj_pregrasp_start, traj_connect]
    elif pregrasp_goal:
      orpy.planningutils.ConvertTrajectorySpecification(traj_pregrasp_goal_rev, spec)
      trajlist = [traj_connect, traj_pregrasp_goal_rev]
    else:
      trajlist = [traj_connect]

    if len(trajlist) == 1:
      final_traj = trajlist[0]
    else:
      final_traj = orpy.RaveCreateTrajectory(env, "")
      cloning_option = 0
      final_traj.Clone(trajlist[0], cloning_option)
      for traj in trajlist[1:]:
        for iwaypoint in xrange(traj.GetNumWaypoints()):
          waypoint = traj.GetWaypoint(iwaypoint)
          final_traj.Insert(final_traj.GetNumWaypoints(), waypoint)

    return final_traj

  
