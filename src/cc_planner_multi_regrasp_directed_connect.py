"""
Closed-chain motion planner with regrasping for bimaual setup.
"""

import openravepy as orpy
import numpy as np
import random
from time import time, sleep
import traceback
import TOPP
from utils.utils import colorize
from utils import utils, heap, lie
from IPython import embed

# Global parameters
FW          = 0
BW          = 1
REACHED     = 0
ADVANCED    = 1
TRAPPED     = 2
NEEDREGRASP = 3
FROMEXTEND  = 0
FROMCONNECT = 1

IK_CHECK_COLLISION = orpy.IkFilterOptions.CheckEnvCollisions
IK_IGNORE_COLLISION = orpy.IkFilterOptions.IgnoreSelfCollisions
TrajectoryFromStr = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString
_RNG = random.SystemRandom()

class SE3Config(object):
  """
  Class of configuration in SE(3) space. It stores both 
  quaternion/translation and transformation matrix.
  """

  def __init__(self, q, p, qd=None, pd=None):
    """
    SE3Config constructor.

    @type   q: numpy.ndarray
    @param  q: Quaternion, (s, vx, vy, vz).
    @type   p: numpy.ndarray
    @param  p: Translation 3-vector, (x, y, z).
    @type  qd: numop.ndarray
    @param qd: Time derivative of q.
    @type  pd: numpy.ndarray
    @param pd: Time derivative of p (translational velocity.
    """
    quat_length = np.linalg.norm(q)
    assert(quat_length > 0)
    self.q = q / quat_length
    if qd is None:
      self.qd = np.zeros(3)
    else:
      self.qd = np.array(qd)

    self.p = p
    if pd is None:
      self.pd = np.zeros(3)
    else:
      self.pd = np.array(pd)

    self.T = orpy.matrixFromPose(np.hstack([self.q, self.p]))

  @staticmethod
  def from_matrix(T):
    """
    Initialize an SE3Config object from a transformation matrix.
    T can be a None, since SE3Config of a goal vertex can be a None.

    @type  T: numpy.ndarray
    @param T: 4x4 transformation matrix
    @rtype:   SE3Config
    @return:  An SE3Config object initialized using T.
    """    
    if T is None:
      return None

    quat = orpy.quatFromRotationMatrix(T[0:3, 0:3])
    p = T[0:3, 3]
    return SE3Config(quat, p)        


class CCTrajectory(object):
  """
  Class of closed-chain trajectory, storing all information needed for 
  a trajectory in closed-chain motions.
  """

  def __init__(self, lie_traj, translation_traj, bimanual_trajs, timestamps):
    """
    CCTrajectory constructor.

    @type          lie_traj: lie.LieTraj
    @param         lie_traj: Trajectory of the manipulated object in SO(3) space.
    @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Translational trajectory of the manipulated object.
    @type     bimanual_wpts: list
    @param    bimanual_wpts: Trajectory of bimanual robots in form 
                             of waypoints list.
    @type        timestamps: list
    @param       timestamps: Timestamps for time parameterization 
                             of C{bimanual_wpts}.
    """
    self.lie_traj         = lie_traj
    self.translation_traj = translation_traj
    self.bimanual_trajs   = bimanual_trajs
    self.timestamps       = timestamps[:]

class CCConfig(object):
  """
  Configuration Class to contain configuration information of
  both robots and the object in closed-chain motion.
  """

  def __init__(self, q_robots, SE3_config):
    """
    CCConfig constructor.

    @type    q_robots: list
    @param   q_robots: List of onfigurations of both robots in the bimanual set-up.
    @type  SE3_config: SE3Config
    @param SE3_config: Configuration of the manipulated object.
    """
    self.q_robots         = np.array(q_robots)
    self.q_robots_nominal = np.array(q_robots)
    self.SE3_config       = SE3_config

  def set_q_robots(self, q_robots):
    self.q_robots = np.array(q_robots)

class CCVertex(object):  
  """
  Vertex of closed-chain motion. It stores all information required
  to generated a RRT tree (C{CCTree}) in this planner.
  """

  def __init__(self, config):
    """
    CCVertex constructor.

    @type  config: CCConfig
    @param config: Configuration of the bimanual set-up in this vertex.
    """
    self.config = config

    # These parameters are to be assigned when the vertex is added to the tree
    self.index                 = 0
    self.parent_index          = None
    self.rot_traj              = None # TOPP trajectory
    self.translation_traj      = None # TOPP trajectory
    self.bimanual_wpts         = []
    self.timestamps            = []
    self.level                 = 0
    self.contain_regrasp       = False
    self.bimanual_regrasp_traj = None

  def remove_regrasp(self):
    self.contain_regrasp = False
    self.config.set_q_robots(self.config.q_robots_nominal)
    self.bimanual_regrasp_traj = None

  def add_regrasp(self, bimanual_regrasp_traj, q_robots):
    self.contain_regrasp = True
    self.config.set_q_robots(q_robots)
    self.bimanual_regrasp_traj = bimanual_regrasp_traj   

class CCTree(object):  
  """
  An RRT tree class for planning closed-chain motion.
  """

  def __init__(self, v_root=None, treetype=FW):
    """
    CCTree constructor.

    @type    v_root: CCVertex
    @param   v_root: Root vertex to grow the tree from.
    @type  treetype: int
    @param treetype: The direction of the tree in the closed-chain motion.
                     It can be either forward(C{FW}) or backward(C{BW}).
    """
    self.vertices = []
    self.length = 0
    if v_root is not None:
      self.vertices.append(v_root)
      self.length = 1

    self.treetype = treetype

  def __len__(self):
    return len(self.vertices)

  def __getitem__(self, index):
    return self.vertices[index]        
  
  def add_vertex(self, v_new, parent_index, rot_traj, translation_traj, bimanual_wpts, timestamps):
    """
    Add a C{CCVertex} to the tree.

    @type             v_new: CCVertex
    @param            v_new: New vertex to be added.
    @type      parent_index: int
    @param     parent_index: Index of C{v_new}'s parent vertex in 
                             the tree's C{vertices} list.
    @type          rot_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param         rot_traj: Trajecoty of the manipulated object's rotational 
                             motion in SO(3) space.
    @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Trajecoty of the manipulated object's 
                             translational motion.
    @type     bimanual_wpts: list
    @param    bimanual_wpts: Trajectory of bimanual robots in form 
                             of waypoints list.
    @type        timestamps: list
    @param       timestamps: Timestamps for time parameterization 
                             of C{bimanual_wpts}.
    """
    v_new.parent_index     = parent_index
    v_new.rot_traj         = rot_traj
    v_new.translation_traj = translation_traj
    v_new.bimanual_wpts    = bimanual_wpts
    v_new.timestamps       = timestamps
    v_new.index            = self.length
    v_new.level            = self.vertices[parent_index].level + 1
    
    self.vertices.append(v_new)
    self.length += 1

  def generate_rot_traj_list(self, end_index=-1):
    """
    Return all C{rot_traj} of vertices 
    connecting the specified vertex and C{v_root}.

    @type  end_index: int
    @param end_index: Index of the vertex used as one end of the connection.
                      The other end is C{v_root}.

    @rtype:  list
    @return: A list containing all C{rot_traj} of all vertices, 
             starting from vertex with a earlier C{timestamp}.
    """
    rot_traj_list = []

    vertex = self.vertices[end_index]
    while vertex.parent_index is not None:
      parent = self.vertices[vertex.parent_index]
      rot_traj_list.append(vertex.rot_traj)
      vertex = parent

    if self.treetype == FW:
      rot_traj_list.reverse()

    return rot_traj_list

  def generate_rot_mat_list(self, end_index=-1):
    """
    Return all rotation matrices of vertices 
    connecting the specified vertex and C{v_root}.

    @type  end_index: int
    @param end_index: Index of the vertex used as one end of the connection.
                      The other end is C{v_root}.

    @rtype:  list
    @return: A list containing rotation matrices of all vertices, 
             starting from vertex with a earlier C{timestamp}.
    """
    rot_mat_list = []

    vertex = self.vertices[end_index]
    while vertex.parent_index is not None:
      parent = self.vertices[vertex.parent_index]
      rot_mat_list.append(vertex.config.SE3_config.T[0:3, 0:3])
      vertex = parent
    rot_mat_list.append(self.vertices[0].config.SE3_config.T[0:3, 0:3])

    if self.treetype == FW:
      rot_mat_list.reverse()

    return rot_mat_list

  def generate_translation_traj_list(self, end_index=-1):
    """
    Return all C{translation_traj} of vertices 
    connecting the specified vertex and C{v_root}.

    @type  end_index: int
    @param end_index: Index of the vertex used as one end of the connection.
                      The other end is C{v_root}.

    @rtype:  list
    @return: A list containing all C{translation_traj} of all vertices, 
             starting from vertex with a earlier C{timestamp}.
    """
    translation_traj_list = []
      
    vertex = self.vertices[end_index]
    while vertex.parent_index is not None:
      parent = self.vertices[vertex.parent_index]
      translation_traj_list.append(vertex.translation_traj)
      vertex = parent

    if self.treetype == FW:
      translation_traj_list.reverse()

    return translation_traj_list

  def generate_bimanual_trajs(self, end_index=-1):
    """
    Return all C{bimanual_wpts} of vertices 
    connecting the specified vertex and C{v_root}.

    @type  end_index: int
    @param end_index: Index of the vertex used as one end of the connection.
                      The other end is C{v_root}.

    @rtype:  list
    @return: A list containing all C{bimanual_wpts} of all vertices, 
             starting from vertex with a earlier C{timestamp}.
    """
    bimanual_trajs = []
      
    vertex = self.vertices[end_index]
    while (vertex.parent_index is not None):
      if vertex.contain_regrasp:
        bimanual_trajs.append(vertex.bimanual_regrasp_traj)
      bimanual_trajs.append(vertex.bimanual_wpts)
      vertex = self.vertices[vertex.parent_index]

    if (self.treetype == FW):
        bimanual_trajs.reverse()

    return bimanual_trajs

  def generate_timestamps_list(self, end_index=-1):
    """
    Return all C{timestamps} of vertices 
    connecting the specified vertex and C{v_root}.

    @type  end_index: int
    @param end_index: Index of the vertex used as one end of the connection.
                      The other end is C{v_root}.

    @rtype:  list
    @return: A list containing all C{timestamps} of all vertices, 
             starting from vertex with a earlier C{timestamp}.
    """
    timestamps_list = []
    
    vertex = self.vertices[end_index]
    while vertex.parent_index is not None:
      parent = self.vertices[vertex.parent_index]
      timestamps_list.append(vertex.timestamps)
      vertex = parent

    if self.treetype == FW:
      timestamps_list.reverse()

    return timestamps_list            


class CCQuery(object):
  """
  Class to store all information needed in a closed-chain query.
  """

  def __init__(self, obj_translation_limits, q_robots_start, q_robots_goal, 
               q_robots_grasp, T_obj_start, T_obj_goal=None, nn=-1, 
               step_size=0.7, velocity_scale=1, interpolation_duration=None, 
               discr_timestep=5e-3, discr_check_timestep=None, 
               enable_bw=False):
    """
    CCQuery constructor. It is independent of robots to be planned since robot
    info will be stored in planner itself.

    Default step_size (when robot has full velocity, i.e. velocity_scale = 1) 
    for each trajectory interpolation is 0.7, with interpolation_duration = 
    1.5s and discr_check_timestep = 0.03s. Defualt discr_timestep is 0.005s.
    These values are determined by experiments to make sure 
      - planning is not too slow
      - interpolated trajectory would result in joint velocity within limit 
        (specifically for denso robot)
      - generated trajectory is smooth
    When user specifies different velocity_scale or step_size, these value 
    are scaled accordingly to satisfy the abovementioned criteria.

    @type  obj_translation_limits: list
    @param obj_translation_limits: Cartesian workspace limits of the object
    @type          q_robots_start: list
    @param         q_robots_start: Start configurations of the two robots.
    @type           q_robots_goal: list
    @param          q_robots_goal: Goal configurations of the two robots.
    @type          q_robots_grasp: list
    @param         q_robots_grasp: Configurations of the two robots' grippers
                                   when grasping the object.
    @type             T_obj_start: numpy.ndarray
    @param            T_obj_start: Start transformation matrix of the object.
    @type              T_obj_goal: numpy.ndarray
    @param             T_obj_goal: Goal transformation matrix of the object.
                                   This is optional, since goal 
                                   transformation can be computed based on 
                                   robots' goal configurations.
    @type                      nn: int
    @param                     nn: Number of nearest vertices to consider for
                                   connection with the new one in extension
                                   and connection.
    @type               step_size: float
    @param              step_size: Size of each step for tree extension, >=0.1
    @type          velocity_scale: float
    @param         velocity_scale: Ratio of the robots' velocity limit to 
                                   their full velocity.
    @type  interpolation_duration: float
    @param interpolation_duration: Length of time used for interpolating 
                                   trajectory connecting vertices. This needs 
                                   to be multiple of C{discr_check_timestep}
    @type          discr_timestep: float
    @param         discr_timestep: Timestep between adjacent waypoints.
    @type    discr_check_timestep: float
    @param   discr_check_timestep: Timestep for 
                                   -  taking samples in feasibility checking
                                   -  generating waypoints by solving IK. 
                                      (other waypoints are interpolated)
                                   This needs to be multiple of 
                                   C{discr_timestep} for uniformity 
                                   uniformity in trajectory generated.
    @type               enable_bw: bool
    @param              enable_bw: B{True} to enable extension of C{tree_end}.
    """
    # Initialize v_start and v_goal
    SE3_config_start    = SE3Config.from_matrix(T_obj_start)
    SE3_config_goal     = SE3Config.from_matrix(T_obj_goal)
    cc_config_start     = CCConfig(q_robots_start, SE3_config_start)
    cc_config_goal      = CCConfig(q_robots_goal, SE3_config_goal)
    self.v_start        = CCVertex(cc_config_start)
    self.v_goal         = CCVertex(cc_config_goal)
    self.q_robots_grasp = q_robots_grasp

    # Initialize RRTs
    self.tree_start             = CCTree(self.v_start, FW)
    self.tree_end               = None # to be initialized when being passed 
                                       # to a planner (after grasping pose 
                                       # check is passed)
    self.nn                     = nn
    self.step_size              = step_size
    self.interpolation_duration = interpolation_duration
    self.discr_timestep         = discr_timestep
    self.discr_check_timestep   = discr_check_timestep
    self.enable_bw              = enable_bw

    if step_size < 0.1:
      raise CCPlannerException('step_size should not be less than 0.1')
      
    if self.discr_check_timestep is None:
      self.discr_check_timestep   = 0.03 / velocity_scale
      self.discr_check_timestep = int(self.discr_check_timestep 
        / self.discr_timestep) * self.discr_timestep

    if self.interpolation_duration is None:
      self.interpolation_duration = step_size * 1.5 / (0.7 * velocity_scale)
      self.interpolation_duration = int(self.interpolation_duration 
        / self.discr_check_timestep) * self.discr_check_timestep

    # traj information
    self.connecting_rot_traj         = None
    self.connecting_translation_traj = None
    self.connecting_bimanual_wpts    = None
    self.connecting_timestamps       = None
    self.rot_traj_list               = None
    self.rot_mat_list                = None
    self.lie_traj                    = None
    self.translation_traj_list       = None
    self.translation_traj            = None
    self.timestamps                  = None
    self.bimanual_trajs              = None

    # Statistics
    self.running_time    = 0.0
    self.iteration_count = 0
    self.solved          = False
    
    # Parameters    
    self.upper_limits = obj_translation_limits[0]
    self.lower_limits = obj_translation_limits[1]
    
  def generate_final_lie_traj(self):
    """
    Generate final lie trajectory of this query (if solved) and store it 
    in {self.lie_traj}. This trajectory is used for the manipulated object's
    motion in SO(3) space. 
    """
    if not self.solved:
      raise CCPlannerException('Query not solved.')
      return

    # Generate rot_traj_list
    self.rot_traj_list = self.tree_start.generate_rot_traj_list()
    if self.connecting_rot_traj is not None:
      self.rot_traj_list.append(self.connecting_rot_traj)
    self.rot_traj_list += self.tree_end.generate_rot_traj_list()

    # Generate rot_mat_list
    self.rot_mat_list = self.tree_start.generate_rot_mat_list()
    self.rot_mat_list += self.tree_end.generate_rot_mat_list()

    # Combine rot_traj_list and rot_mat_list to generate lie_traj
    self.lie_traj = lie.LieTraj(self.rot_mat_list, self.rot_traj_list)

  def generate_final_translation_traj(self):
    """
    Generate final translational trajectory of this query (if solved) and store it 
    in {self.translation_traj}. This trajectory is used for the manipulated
    object's translational motion.
    """
    if not self.solved:
      raise CCPlannerException('Query not solved.')
      return

    # Generate translation_traj_list
    self.translation_traj_list = self.tree_start.generate_translation_traj_list()
    if self.connecting_translation_traj is not None:
      self.translation_traj_list.append(self.connecting_translation_traj)
    self.translation_traj_list += self.tree_end.generate_translation_traj_list()

    # Convert translation_traj_list to translation_traj
    self.translation_traj = TrajectoryFromStr(utils.traj_str_from_traj_list(self.translation_traj_list))

  def generate_final_bimanual_trajs(self):
    """
    Generate final waypoints for both robots in the bimanual set-up of this 
    query (if solved) and store it in {self.bimanual_wpts}. 
    """
    if not self.solved:
      raise CCPlannerException('Query not solved.')
      return
    
    bimanual_trajs = self.tree_start.generate_bimanual_trajs()  
    bimanual_trajs.append(self.connecting_bimanual_wpts)        
    bimanual_trajs += self.tree_end.generate_bimanual_trajs()

    self.bimanual_trajs = utils.merge_bimanual_trajs_wpts_list(bimanual_trajs)

  def generate_final_timestamps(self):
    """
    Generate final timestamps of this query (if solved) and store it in
    {self.timestamps}. It is used as time discretization together with
    C{self.bimanual_wpts} for both robots' motion.
    """
    if (not self.solved):
      raise CCPlannerException('Query not solved.')
      return

    timestamps_list = self.tree_start.generate_timestamps_list()
    if (self.connecting_timestamps is not None):
      timestamps_list.append(self.connecting_timestamps)
    timestamps_list += self.tree_end.generate_timestamps_list()

    self.timestamps = utils.merge_timestamps_list(timestamps_list)

  def generate_final_cctraj(self):
    """
    Generate final closed-chain trajectory (C{CCTrajectory}) of this query 
    (if solved) and store it in {self.cctraj}. It combines all the components required for a closed-chain motion.
    """
    if (not self.solved):
      raise CCPlannerException('Query not solved.')
      return

    # Generate CCTrajectory components
    self.generate_final_lie_traj()
    self.generate_final_translation_traj()
    self.generate_final_timestamps()
    self.generate_final_bimanual_trajs()
    
    self.cctraj = CCTrajectory(self.lie_traj, self.translation_traj, 
                               self.bimanual_trajs, self.timestamps)

class CCPlanner(object):
  """
  A closed-chain motion planner for bimanual set-up.

  Requirements:
  - two identical robots
  """
  
  def __init__(self, manip_obj, robots, debug=False):
    """
    CCPlanner constructor. It requires infomation of the robots and the object
    being manipulated.

    @type  manip_obj: openravepy.KinBody
    @param manip_obj: Object to be manipulated in the closed-chain motion. It
                      connects the end-effectors of the two robots.
    @type     robots: list of openravepy.Robot
    @param    robots: List of robots for the closed-chain motion.
    @type      debug: bool
    @param     debug: B{True} if debug info is to be displayed.
    """
    self.obj = manip_obj
    self._debug = debug
    self.robots = robots
    self.manips = []
    self.basemanips = []
    self.taskmanips = []
    for (i, robot) in enumerate(self.robots):
      self.manips.append(robot.GetActiveManipulator())
      self.basemanips.append(orpy.interfaces.BaseManipulation(robot))
      self.taskmanips.append(orpy.interfaces.TaskManipulation(robot))
      robot.SetActiveDOFs(self.manips[i].GetArmIndices())

    self.bimanual_obj_tracker = BimanualObjectTracker(self.robots, manip_obj, debug=self._debug)

    self._active_dofs = self.manips[0].GetArmIndices()
    self._vmax = self.robots[0].GetDOFVelocityLimits()[self._active_dofs]
    self._amax = self.robots[0].GetDOFAccelerationLimits()[self._active_dofs]

    self.env = self.obj.GetEnv()

  def sample_SE3_config(self):
    """
    Return a random SE3 configuration C{SE3Config}.
    This function does not do any feasibility checking since when
    extending a vertex on a tree to this config, we do not use
    this config directly.

    @rtype:  SE3Config
    @return: A random SE3 Configuration.
    """ 
    q_rand = lie.RandomQuat()
    p_rand = np.asarray([_RNG.uniform(self._query.lower_limits[i], 
                                      self._query.upper_limits[i]) 
                        for i in xrange(3)])
    
    qd_rand = np.zeros(3)
    pd_rand = np.zeros(3)

    return SE3Config(q_rand, p_rand, qd_rand, pd_rand)
  
  def _check_grasping_pose(self):
    """
    Check if the start and goal grasping pose matches; if the check is passed,
    complete tree_end in C{self._query}. Meanwhile, initialize attribute
    C{self.bimanual_T_rel} to store relative pose of the robots' end-effectors
    w.r.t. to the object.
    """
    # Compute relative transformation from end-effectors to object
    self.bimanual_T_rel = []
    for i in xrange(2):
      self.bimanual_T_rel.append(np.dot(np.linalg.inv(self._query.v_start.config.SE3_config.T), utils.compute_endeffector_transform(self.manips[i], self._query.v_start.config.q_robots[i])))

    # Compute object SE3_config at goal if not specified
    if self._query.v_goal.config.SE3_config is None:
      T_left_robot_goal = utils.compute_endeffector_transform(self.manips[0], self._query.v_goal.config.q_robots[0])
      T_obj_goal = np.dot(T_left_robot_goal, np.linalg.inv(self.bimanual_T_rel[0]))
      self._query.v_goal.config.SE3_config = SE3Config.from_matrix(T_obj_goal)

    # Check start and goal grasping pose
    bimanual_goal_rel_T = []
    for i in xrange(2):
      bimanual_goal_rel_T.append(np.dot(np.linalg.inv(self._query.v_goal.config.SE3_config.T), utils.compute_endeffector_transform(self.manips[i], self._query.v_goal.config.q_robots[i])))

    if not np.isclose(self.bimanual_T_rel, bimanual_goal_rel_T, atol=1e-3).all():
      raise CCPlannerException('Start and goal grasping pose not matching.')

    # Complete tree_end in the query 
    self._query.tree_end = CCTree(self._query.v_goal, BW)


  def loose_gripper(self, query):
    """
    Open grippers of C{self.robots} by a small amount from C{q_robots_grasp}
    stored in C{query}. This is necessary to avoid collision between object
    and gripper in planning.

    @type  query: CCQuery
    @param query: Query used to extract the grippers' configuration when grasping
                  the object, which is taken as a reference for loosing the gripper.
    """
    for i in xrange(2):
      self.robots[i].SetDOFValues([query.q_robots_grasp[i]*0.7],
                                  [self.manips[i].GetArmDOF()])

  def set_query(self, query):
    """
    Set a C{CCQuery} object to the planner for planning. Then checks whether 
    the start and goal grasping pose matches.

    @type  query: CCQuery
    @param query: Query to be used for planning.
    """
    self._query = query
    self._check_grasping_pose()
    self.bimanual_obj_tracker.update_vmax()

  def solve(self, timeout=20):
    """
    Solve the query stored in the planner. 

    @type  timeout: float
    @param timeout: Time limit for solving the query.

    @rtype:  int
    @return: Whether the query is solved within given time limit.
    """
    if self._query.solved:
      self._output_info('This query has already been solved.', 'green')
      return True

    self.loose_gripper(self._query)

    t = 0.0
    prev_iter = self._query.iteration_count

    reextend = False
    reextend_type = None
    reextend_info = None
    
    t_begin = time()
    res = self._connect()
    if len(res) == 1:
      status = res[0]
      if status == REACHED:
        self._query.iteration_count += 1
        t_end = time()
        self._query.running_time += (t_end - t_begin)
        self._output_info('Path found. Iterations: {0}. Running time: {1}s.'.format(self._query.iteration_count, self._query.running_time), 'green')
        self._query.solved = True
        self._query.generate_final_cctraj()
        self.reset_config(self._query)
        return True
      reextend = False
    else:
      robot_index, q_new, q_robots_orig, T_obj_orig = res[1]
      nearest_index = res[2]
      reextend = True
      reextend_type = FROMCONNECT
      reextend_info = (robot_index, q_new, q_robots_orig, 
                       T_obj_orig, nearest_index)

    elasped_time = time() - t_begin
    t += elasped_time
    self._query.running_time += elasped_time

    while (t < timeout):
      t_begin = time()

      if reextend:
        SE3_config = SE3Config.from_matrix(reextend_info[3])
        if reextend_type == FROMCONNECT:
          self._query.iteration_count += 1
          self._output_debug(
            'Iteration no. {0} (try extension from interuptted connect)'.format(self._query.iteration_count), 'blue')
        elif reextend_type == FROMEXTEND:
          self._output_debug(
            'Iteration no. {0} (retry extension from interuptted extend)'.format(self._query.iteration_count), 'blue')
      else:
        self._query.iteration_count += 1
        self._output_debug('Iteration no. {0} (start new extension)'.format(
                            self._query.iteration_count), 'blue')
        SE3_config = self.sample_SE3_config()

      res = self._extend(SE3_config, reextend=reextend, 
                         reextend_info=reextend_info)
      if len(res) == 1:
        status = res[0]
        if status != TRAPPED:
          self._output_debug('Tree start : {0}; Tree end : {1}'.format(len(self._query.tree_start.vertices), len(self._query.tree_end.vertices)), 'green')

          res = self._connect()
          if len(res) == 1:
            status = res[0]
            if status == REACHED:
              t_end = time()
              self._query.running_time += (t_end - t_begin)
              self._output_info('Path found. Iterations: {0}. Running time: {1}s.'.format(self._query.iteration_count, self._query.running_time), 'green')
              self._query.solved = True
              self._query.generate_final_cctraj()
              self.reset_config(self._query)
              return True
            reextend = False
          else:
            robot_index, q_new, q_robots_orig, T_obj_orig = res[1]
            nearest_index = res[2]
            reextend = True
            reextend_type = FROMCONNECT
            reextend_info = (robot_index, q_new, q_robots_orig, 
                             T_obj_orig, nearest_index)
      else:
        robot_index, q_new, q_robots_orig, T_obj_orig = res[1]
        nearest_index = res[2] 
        reextend = True
        reextend_type = FROMEXTEND
        reextend_info = (robot_index, q_new, q_robots_orig, 
                         T_obj_orig, nearest_index)
        
      elasped_time = time() - t_begin
      t += elasped_time
      self._query.running_time += elasped_time

    self._output_info('Timeout {0}s reached after {1} iterations'.format(timeout, self._query.iteration_count - prev_iter), 'red')
    self.reset_config(self._query)
    return False

  def reset_config(self, query):
    """
    Reset everything to their starting configuration according to the query,
    including re-closing the grippers which were probably opened for planning.
    This is used after planning is done since in planning process robots and object
    will be moved for collision checking.

    @type  query: CCQuery
    @param query: Query to be used to extract starting configuration.
    """
    for i in xrange(len(self.robots)):
      self.robots[i].SetActiveDOFValues(query.v_start.config.q_robots[i])
      self.robots[i].SetDOFValues([query.q_robots_grasp[i]],
                                  [self.manips[i].GetArmDOF()])
    self.obj.SetTransform(query.v_start.config.SE3_config.T)

  def _extend(self, SE3_config, reextend=False, reextend_info=None):
    """
    Extend the tree(s) in C{self._query} towards the given SE3 config.

    @type  SE3_config: SE3Config
    @param SE3_config: Configuration towards which the tree will be extended.

    @rtype:  int
    @return: Result of this extension attempt. Possible values:
             -  B{TRAPPED}:  when the extension fails
             -  B{REACHED}:  when the extension reaches the given config
             -  B{ADVANCED}: when the tree is extended towards the given
                             config
    """
    if (self._query.iteration_count - 1) % 2 == FW or not self._query.enable_bw:
      return self._extend_fw(SE3_config, reextend=reextend,
                             reextend_info=reextend_info)
    else:
      return self._extend_bw(SE3_config, reextend=reextend,
                             reextend_info=reextend_info)

  def _extend_fw(self, SE3_config, reextend=False, reextend_info=None):
    """
    Extend C{tree_start} (rooted at v_start) towards the given SE3 config.

    @type  SE3_config: SE3Config
    @param SE3_config: Configuration towards which the tree will be extended.

    @rtype:  int
    @return: Result of this extension attempt. Possible values:
             -  B{TRAPPED}:  when the extension fails
             -  B{REACHED}:  when the extension reaches the given config
             -  B{ADVANCED}: when the tree is extended towards the given
                             config
    """
    status = TRAPPED
    nnindices = self._nearest_neighbor_indices(SE3_config, FW)
    if reextend:
      nnindices = (reextend_info[4],)
    for index in nnindices:
      self._output_debug('index:{0}, nnindices:{1}'.format(index, nnindices))
      v_near = self._query.tree_start[index]
      
      q_beg  = v_near.config.SE3_config.q
      qd_beg = v_near.config.SE3_config.qd
      p_beg  = v_near.config.SE3_config.p
      pd_beg = v_near.config.SE3_config.pd

      q_end = SE3_config.q
      p_end = SE3_config.p
      qd_end = SE3_config.qd
      pd_end = SE3_config.pd

      if reextend:
        status = REACHED
        new_SE3_config = SE3_config
      else:
        # Check if SE3_config is too far from v_near.SE3_config
        SE3_dist = utils.SE3_distance(SE3_config.T, 
                                      v_near.config.SE3_config.T,
                                      1.0 / np.pi, 1.0)
        if SE3_dist <= self._query.step_size:
          status = REACHED
          new_SE3_config = SE3_config
        else:
          if not utils._is_close_axis(q_beg, q_end):
            q_end = -q_end
          q_end = q_beg + self._query.step_size * (q_end - q_beg) / SE3_dist
          q_end /= np.sqrt(np.dot(q_end, q_end))

          p_end = p_beg + self._query.step_size * (p_end - p_beg) / SE3_dist

          new_SE3_config = SE3Config(q_end, p_end, qd_end, pd_end)
          status = ADVANCED

        # Check collision (SE3_config)
        res = self.is_collision_free_SE3_config(new_SE3_config)
        if not res:
          self._output_debug('TRAPPED : SE(3) config in collision',
                             bold=False)
          status = TRAPPED
          continue

        # Check reachability (SE3_config)
        res = self.check_SE3_config_reachability(new_SE3_config)
        if not res:
          self._output_debug('TRAPPED : SE(3) config not reachable',
                             bold=False)
          status = TRAPPED
          continue
      
      # Interpolate a SE3 trajectory for the object
      R_beg = orpy.rotationMatrixFromQuat(q_beg)
      R_end = orpy.rotationMatrixFromQuat(q_end)
      rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                    self._query.interpolation_duration)
      translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, self._query.interpolation_duration)

      # Check translational limit
      # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
      res = utils.check_translation_traj_str_limits(self._query.upper_limits, self._query.lower_limits, translation_traj_str)
      if not res:
        self._output_debug('TRAPPED : SE(3) trajectory exceeds translational'
                           ' limit', bold=False)
        status = TRAPPED
        continue

      translation_traj = TrajectoryFromStr(translation_traj_str)

      # Check collision (object trajectory)
      res = self.is_collision_free_SE3_traj(rot_traj, translation_traj, R_beg)
      if not res:
        self._output_debug('TRAPPED : SE(3) trajectory in collision', 
                           bold=False)
        status = TRAPPED
        continue
      
      # Check reachability (object trajectory)
      res = self.check_SE3_traj_reachability(lie.LieTraj([R_beg, R_end],
              [rot_traj]), translation_traj, v_near.config.q_robots)
      if res[0] is False:
        passed, bimanual_wpts, timestamps = res[1:]
        if not passed:
          self._output_debug('TRAPPED : SE(3) trajectory not reachable', 
                             bold=False)
          status = TRAPPED
          continue

        # Now this trajectory is alright.
        self._output_debug('Successful : new vertex generated', 
                           color='green', bold=False)
        new_q_robots = [wpts[-1] for wpts in bimanual_wpts] 
        new_config = CCConfig(new_q_robots, new_SE3_config)
        v_new = CCVertex(new_config)

        if reextend:
          q_robots_pr = np.array(new_q_robots) # post regrasp configs
          q_robots_pr[reextend_info[0]] = reextend_info[1]
          v_new.add_regrasp({0: None, 1: None}, q_robots_pr)
          self._output_info('Adding regrasping-----', 'yellow')

        self._query.tree_start.add_vertex(v_new, v_near.index, rot_traj, translation_traj, bimanual_wpts, timestamps)
        return status,
      else:
        self._output_debug('TRAPPED : SE(3) trajectory need regrasping', 
                           bold=False)
        status = NEEDREGRASP
        return status, res[1:], index
    return status,

  def _extend_bw(self, SE3_config, reextend=False, reextend_info=None):
    """
    Extend C{tree_end} (rooted at v_goal) towards the given SE3 config.

    @type  SE3_config: SE3Config
    @param SE3_config: Configuration towards which the tree will be extended.

    @rtype:  int
    @return: Result of this extension attempt. Possible values:
             -  B{TRAPPED}:  when the extension fails
             -  B{REACHED}:  when the extension reaches the given config
             -  B{ADVANCED}: when the tree is extended towards the given
                             config
    """
    status = TRAPPED
    nnindices = self._nearest_neighbor_indices(SE3_config, BW)
    if reextend:
      nnindices = (reextend_info[4],)
    for index in nnindices:
      self._output_debug('index:{0}, nnindices:{1}'.format(index, nnindices))
      v_near = self._query.tree_end[index]
      
      # quaternion
      q_end  = v_near.config.SE3_config.q
      qd_end = v_near.config.SE3_config.qd
      
      # translation
      p_end  = v_near.config.SE3_config.p
      pd_end = v_near.config.SE3_config.pd

      q_beg = SE3_config.q
      p_beg = SE3_config.p
      qd_beg = SE3_config.qd
      pd_beg = SE3_config.pd

      if reextend:
        status = REACHED
        new_SE3_config = SE3_config
      else:
        # Check if SE3_config is too far from v_near.SE3_config
        SE3_dist = utils.SE3_distance(SE3_config.T,
                                      v_near.config.SE3_config.T,
                                      1.0 / np.pi, 1.0)
        if SE3_dist <= self._query.step_size:
          status = REACHED
          new_SE3_config = SE3_config
        else:
          if not utils._is_close_axis(q_beg, q_end):
            q_beg = -q_beg
          q_beg = q_end + self._query.step_size * (q_beg - q_end) / SE3_dist
          q_beg /= np.sqrt(np.dot(q_beg, q_beg))

          p_beg = p_end + self._query.step_size * (p_beg - p_end) / SE3_dist

          new_SE3_config = SE3Config(q_beg, p_beg, qd_beg, pd_beg)
          status = ADVANCED

        # Check collision (SE3_config)
        res = self.is_collision_free_SE3_config(new_SE3_config)
        if not res:
          self._output_debug('TRAPPED : SE(3) config in collision',
                             bold=False)
          status = TRAPPED
          continue

        # Check reachability (SE3_config)
        res = self.check_SE3_config_reachability(new_SE3_config)
        if not res:
          self._output_debug('TRAPPED : SE(3) config not reachable',
                             bold=False)
          status = TRAPPED
          continue

      # Interpolate a SE3 trajectory for the object
      R_beg = orpy.rotationMatrixFromQuat(q_beg)
      R_end = orpy.rotationMatrixFromQuat(q_end)
      rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                    self._query.interpolation_duration)
      translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, self._query.interpolation_duration)

      # Check translational limit
      # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
      res = utils.check_translation_traj_str_limits(self._query.upper_limits, self._query.lower_limits, translation_traj_str)
      if not res:
        self._output_debug('TRAPPED : SE(3) trajectory exceeds translational limit', bold=False)
        status = TRAPPED
        continue

      translation_traj = TrajectoryFromStr(translation_traj_str)

      # Check collision (object trajectory)
      res = self.is_collision_free_SE3_traj(rot_traj, translation_traj, R_beg)
      if not res:
        self._output_debug('TRAPPED : SE(3) trajectory in collision', 
                           bold=False)
        status = TRAPPED
        continue
      
      # Check reachability (object trajectory)
      res = self.check_SE3_traj_reachability(lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj, v_near.config.q_robots, direction=BW)
      if res[0] is False:
        passed, bimanual_wpts, timestamps = res[1:]
        if not passed:
          self._output_debug('TRAPPED : SE(3) trajectory not reachable', 
                             bold=False)
          status = TRAPPED
          continue

        # Now this trajectory is alright.
        self._output_debug('Successful : new vertex generated', 
                           color='green', bold=False)
        new_q_robots = [wpts[0] for wpts in bimanual_wpts] 
        new_config = CCConfig(new_q_robots, new_SE3_config)
        v_new = CCVertex(new_config)

        if reextend:
          q_robots_pr = np.array(new_q_robots) # post regrasp configs
          q_robots_pr[reextend_info[0]] = reextend_info[1]
          v_new.add_regrasp({0: None, 1: None}, q_robots_pr)
          self._output_info('Adding regrasping-----', 'yellow')

        self._query.tree_end.add_vertex(v_new, v_near.index, rot_traj, translation_traj, bimanual_wpts, timestamps)
        return status,
      else:
        self._output_debug('TRAPPED : SE(3) trajectory need regrasping', 
                           bold=False)
        status = NEEDREGRASP
        return status, res[1:], index
    return status,

  def _connect(self):
    """
    Connect C{tree_start} and C{tree_end} in C{self._query}.

    @rtype:  int
    @return: Result of this connecting attempt. Possible values:
             -  B{TRAPPED}: connection successful
             -  B{REACHED}: connection failed
    """
    if (self._query.iteration_count - 1) % 2 == FW or not self._query.enable_bw:
      # tree_start has just been extended
      return self._connect_fw()
    else:
      # tree_end has just been extended
      return self._connect_bw()


  def _connect_fw(self):
    """
    Connect the newly added vertex in C{tree_start} to other vertices on
    C{tree_end}.

    @rtype:  int
    @return: Result of this connecting attempt. Possible values:
             -  B{TRAPPED}: connection successful
             -  B{REACHED}: connection failed
    """
    v_test = self._query.tree_start.vertices[-1]

    nnindices = self._nearest_neighbor_indices(v_test.config.SE3_config, BW)
    status = TRAPPED
    for index in nnindices:
      v_near = self._query.tree_end[index]

      # quaternion
      q_beg  = v_test.config.SE3_config.q
      qd_beg = v_test.config.SE3_config.qd
      q_end  = v_near.config.SE3_config.q
      qd_end = v_near.config.SE3_config.qd
      
      # translation
      p_beg  = v_test.config.SE3_config.p
      pd_beg = v_test.config.SE3_config.pd
      p_end  = v_near.config.SE3_config.p
      pd_end = v_near.config.SE3_config.pd
      
      # Check distance
      SE3_dist = utils.SE3_distance(v_test.config.SE3_config.T, 
                                    v_near.config.SE3_config.T, 
                                    1.0 / np.pi, 1.0)
      if SE3_dist <= self._query.step_size: # connect directly
        # Interpolate the object trajectory
        R_beg = orpy.rotationMatrixFromQuat(q_beg)
        R_end = orpy.rotationMatrixFromQuat(q_end)
        rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                      self._query.interpolation_duration)
        translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end,
          pd_beg, pd_end, self._query.interpolation_duration)

        # Check translational limit
        # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
        if not utils.check_translation_traj_str_limits(
                self._query.upper_limits, self._query.lower_limits, 
                translation_traj_str):
          self._output_debug('TRAPPED : SE(3) trajectory exceeds '
                             'translational limit', bold=False)
          continue

        translation_traj = TrajectoryFromStr(translation_traj_str)

        # Check collision (object trajectory)
        if not self.is_collision_free_SE3_traj(
                rot_traj, translation_traj, R_beg):
          self._output_debug('TRAPPED : SE(3) trajectory in collision', 
                             bold=False)
          continue
        
        # Check reachability (object trajectory)
        res = self.check_SE3_traj_reachability(lie.LieTraj([R_beg, R_end], 
                [rot_traj]), translation_traj, v_near.config.q_robots, 
                direction=BW)
        if res[0] is False:
          passed, bimanual_wpts, timestamps = res[1:]
          if not passed:
            self._output_debug('TRAPPED : SE(3) trajectory not reachable', 
                               bold=False)
            continue

          # Check similarity of terminal IK solutions
          bimanual_regrasp_traj = {0: None, 1: None}
          eps = 1e-3
          for i in xrange(2):
            if not utils.distance(v_test.config.q_robots_nominal[i], 
                      bimanual_wpts[i][0]) < eps:
              self._output_debug('IK discrepancy (robot {0})'.format(i), 
                                bold=False)
              self.robots[i].SetDOFValues([0], [self.manips[i].GetArmDOF()])
              self.robots[i].SetActiveDOFValues(
                v_test.config.q_robots_nominal[i])
              sleep(0.01)
              self._output_debug('Planning regrasping......')
              # bimanual_regrasp_traj[i] = self.basemanips[i].MoveActiveJoints(goal=bimanual_wpts[i][0], outputtrajobj=True, 
              #    execute=False)
              self.loose_gripper(self._query)

          # Now the connection is successful
          v_test.remove_regrasp() # remove possible existing regrasp action
          v_test.add_regrasp(bimanual_regrasp_traj, 
                             [bimanual_wpts[0][0], bimanual_wpts[1][0]])
          self._output_info('Adding regrasping-----', 'yellow')
          self._query.tree_end.vertices.append(v_near)
          self._query.connecting_rot_traj         = rot_traj
          self._query.connecting_translation_traj = translation_traj
          self._query.connecting_bimanual_wpts    = bimanual_wpts
          self._query.connecting_timestamps       = timestamps
          status = REACHED
          return status,
        else:
          self._output_debug('TRAPPED : SE(3) trajectory need regrasping', 
                             bold=False)
          # continue
          status = NEEDREGRASP
          return status, res[1:], index

      else: # extend towards target by step_size
        if not utils._is_close_axis(q_beg, q_end):
          q_beg = -q_beg
        q_beg = q_end + self._query.step_size * (q_beg - q_end) / SE3_dist
        q_beg /= np.sqrt(np.dot(q_beg, q_beg))
        p_beg = p_end + self._query.step_size * (p_beg - p_end) / SE3_dist
        new_SE3_config = SE3Config(q_beg, p_beg, qd_beg, pd_beg)

        # Check collision (SE3_config)
        res = self.is_collision_free_SE3_config(new_SE3_config)
        if not res:
          self._output_debug('TRAPPED : SE(3) config in collision',
                             bold=False)
          status = TRAPPED
          continue

        # Check reachability (SE3_config)
        res = self.check_SE3_config_reachability(new_SE3_config)
        if not res:
          self._output_debug('TRAPPED : SE(3) config not reachable',
                             bold=False)
          status = TRAPPED
          continue

        # Interpolate a SE3 trajectory for the object
        R_beg = orpy.rotationMatrixFromQuat(q_beg)
        R_end = orpy.rotationMatrixFromQuat(q_end)
        rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                      self._query.interpolation_duration)
        translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, self._query.interpolation_duration)

        # Check translational limit
        # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
        res = utils.check_translation_traj_str_limits(self._query.upper_limits, self._query.lower_limits, translation_traj_str)
        if not res:
          self._output_debug('TRAPPED : SE(3) trajectory exceeds translational limit', bold=False)
          status = TRAPPED
          continue

        translation_traj = TrajectoryFromStr(translation_traj_str)

        # Check collision (object trajectory)
        res = self.is_collision_free_SE3_traj(rot_traj, translation_traj, 
                                              R_beg)
        if not res:
          self._output_debug('TRAPPED : SE(3) trajectory in collision', 
                             bold=False)
          status = TRAPPED
          continue

        # Check reachability (object trajectory)
        res = self.check_SE3_traj_reachability(lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj, v_near.config.q_robots, direction=BW)
        if res[0] is False: # no need regrasp
          passed, bimanual_wpts, timestamps = res[1:]
          if not passed:
            self._output_debug('TRAPPED : SE(3) trajectory not reachable', 
                               bold=False)
            status = TRAPPED
            continue

          # Now this trajectory is alright.
          self._output_debug('Advanced : new vertex generated', 
                             color='green', bold=False)
          new_q_robots = [wpts[0] for wpts in bimanual_wpts] 
          new_config = CCConfig(new_q_robots, new_SE3_config)
          v_new = CCVertex(new_config)
          self._query.tree_end.add_vertex(v_new, v_near.index, rot_traj,
                                          translation_traj, bimanual_wpts,
                                          timestamps)
          status = ADVANCED
          return status,
        else:
          self._output_debug('TRAPPED : SE(3) trajectory need regrasping', 
                             bold=False)
          status = NEEDREGRASP
          return status, res[1:], index
    return status,        


  def _connect_bw(self):
    """
    Connect the newly added vertex in C{tree_end} to other vertices on
    C{tree_start}.

    @rtype:  int
    @return: Result of this connecting attempt. Possible values:
             -  B{TRAPPED}: connection successful
             -  B{REACHED}: connection failed
    """
    v_test = self._query.tree_end.vertices[-1]
    nnindices = self._nearest_neighbor_indices(v_test.config.SE3_config, FW)
    status = TRAPPED
    for index in nnindices:
      v_near = self._query.tree_start[index]
      
      # quaternion
      q_end  = v_test.config.SE3_config.q
      qd_end = v_test.config.SE3_config.qd
      
      q_beg  = v_near.config.SE3_config.q
      qd_beg = v_near.config.SE3_config.qd
      
      # translation
      p_end  = v_test.config.SE3_config.p
      pd_end = v_test.config.SE3_config.pd

      p_beg  = v_near.config.SE3_config.p
      pd_beg = v_near.config.SE3_config.pd
      

      # Check distance
      SE3_dist = utils.SE3_distance(v_test.config.SE3_config.T, 
                                    v_near.config.SE3_config.T, 
                                    1.0 / np.pi, 1.0)
      if SE3_dist <= self._query.step_size: # connect directly
        # Interpolate the object trajectory
        R_beg = orpy.rotationMatrixFromQuat(q_beg)
        R_end = orpy.rotationMatrixFromQuat(q_end)
        rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                      self._query.interpolation_duration)
        translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, self._query.interpolation_duration)  

        # Check translational limit
        # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
        if not utils.check_translation_traj_str_limits(self._query.upper_limits, self._query.lower_limits, translation_traj_str):
          self._output_debug('TRAPPED : SE(3) trajectory exceeds translational limit', bold=False)
          continue

        translation_traj = TrajectoryFromStr(translation_traj_str)

        # Check collision (object trajectory)
        if not self.is_collision_free_SE3_traj(rot_traj, translation_traj, 
                                               R_beg):
          self._output_debug('TRAPPED : SE(3) trajectory in collision', 
                             bold=False)
          continue
        
        # Check reachability (object trajectory)
        res = self.check_SE3_traj_reachability(lie.LieTraj([R_beg, R_end],
                [rot_traj]), translation_traj, v_near.config.q_robots)
        if res[0] is False:
          passed, bimanual_wpts, timestamps = res[1:]
          if not passed:
            self._output_debug('TRAPPED : SE(3) trajectory not reachable', bold=False)
            continue

          # Check similarity of terminal IK solutions
          bimanual_regrasp_traj = {0: None, 1: None}
          eps = 1e-3
          for i in xrange(2):
            if not utils.distance(v_test.config.q_robots_nominal[i], 
                      bimanual_wpts[i][-1]) < eps:
              self._output_debug('IK discrepancy (robot {0})'.format(i), 
                                 bold=False)
              self.robots[i].SetDOFValues([0], [self.manips[i].GetArmDOF()])
              self.robots[i].SetActiveDOFValues(bimanual_wpts[i][-1])
              self._output_debug('Planning regrasping......')
              sleep(0.01)          
              # bimanual_regrasp_traj[i] = self.basemanips[i].MoveActiveJoints(
              #   goal=v_test.config.q_robots_nominal[i], outputtrajobj=True, execute=False)
              self.loose_gripper(self._query)

          # Now the connection is successful
          v_test.remove_regrasp() # remove possible existing regrasp action
          v_test.add_regrasp(bimanual_regrasp_traj,
                             [bimanual_wpts[0][-1], bimanual_wpts[1][-1]])
          self._output_info('Adding regrasping-----', 'yellow')
          self._query.tree_start.vertices.append(v_near)
          self._query.connecting_rot_traj         = rot_traj
          self._query.connecting_translation_traj = translation_traj
          self._query.connecting_bimanual_wpts    = bimanual_wpts
          self._query.connecting_timestamps       = timestamps
          status = REACHED
          return status,
        else: # need regrasp
          self._output_debug('TRAPPED : SE(3) trajectory need regrasping', 
                             bold=False)
          # continue
          status = NEEDREGRASP
          return status, res[1:], index

      else: # extend towards target by step_size
        if not utils._is_close_axis(q_beg, q_end):
          q_end = -q_end
        q_end = q_beg + self._query.step_size * (q_end - q_beg) / SE3_dist
        q_end /= np.sqrt(np.dot(q_end, q_end))
        p_end = p_beg + self._query.step_size * (p_end - p_beg) / SE3_dist
        new_SE3_config = SE3Config(q_end, p_end, qd_end, pd_end)

        # Check collision (SE3_config)
        res = self.is_collision_free_SE3_config(new_SE3_config)
        if not res:
          self._output_debug('TRAPPED : SE(3) config in collision',
                             bold=False)
          status = TRAPPED
          continue

        # Check reachability (SE3_config)
        res = self.check_SE3_config_reachability(new_SE3_config)
        if not res:
          self._output_debug('TRAPPED : SE(3) config not reachable',
                             bold=False)
          status = TRAPPED
          continue
        
        # Interpolate a SE3 trajectory for the object
        R_beg = orpy.rotationMatrixFromQuat(q_beg)
        R_end = orpy.rotationMatrixFromQuat(q_end)
        rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                      self._query.interpolation_duration)
        translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, 
          pd_beg, pd_end, self._query.interpolation_duration)

        # Check translational limit
        # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
        res = utils.check_translation_traj_str_limits(self._query.upper_limits, self._query.lower_limits, translation_traj_str)
        if not res:
          self._output_debug('TRAPPED : SE(3) trajectory exceeds '
                             'translational limit', bold=False)
          status = TRAPPED
          continue

        translation_traj = TrajectoryFromStr(translation_traj_str)

        # Check collision (object trajectory)
        res = self.is_collision_free_SE3_traj(rot_traj, translation_traj, 
                                              R_beg)
        if not res:
          self._output_debug('TRAPPED : SE(3) trajectory in collision', 
                             bold=False)
          status = TRAPPED
          continue
        
        # Check reachability (object trajectory)
        res = self.check_SE3_traj_reachability(lie.LieTraj([R_beg, R_end],
                [rot_traj]), translation_traj, v_near.config.q_robots)
        if res[0] is False: # no need regrasp
          passed, bimanual_wpts, timestamps = res[1:]
          if not passed:
            self._output_debug('TRAPPED : SE(3) trajectory not reachable', 
                               bold=False)
            status = TRAPPED
            continue

          # Now this trajectory is alright.
          self._output_debug('Advanced : new vertex generated', 
                             color='green', bold=False)
          new_q_robots = [wpts[-1] for wpts in bimanual_wpts] 
          new_config = CCConfig(new_q_robots, new_SE3_config)
          v_new = CCVertex(new_config)
          self._query.tree_start.add_vertex(v_new, v_near.index, rot_traj,
                                            translation_traj, bimanual_wpts,
                                            timestamps)
          status = ADVANCED
          return status,
        else:
          self._output_debug('TRAPPED : SE(3) trajectory need regrasping', 
                             bold=False)
          status = NEEDREGRASP
          return status, res[1:], index

    return status,        

  
  def is_collision_free_SE3_config(self, SE3_config):
    """
    Check whether the given C{SE3_config} is collision-free.
    This check ignores the robots, which will be checked later.

    @type  SE3_config: SE3Config
    @param SE3_config: SE3 configuration to be checked.

    @rtype:  bool
    @return: B{True} if the config is collision-free.
    """
    self._enable_robots_collision(False)
    self.obj.SetTransform(SE3_config.T)
    is_free = not self.env.CheckCollision(self.obj)
    self._enable_robots_collision()

    return is_free
  
  def is_collision_free_SE3_traj(self, rot_traj, translation_traj, R_beg):
    """
    Check whether the given SE3 trajectory is collision-free.
    This check ignores the robots, which will be checked later.

    @type          rot_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param         rot_traj: Trajecoty of the manipulated object's rotational 
                             motion in SO(3) space.
    @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Trajecoty of the manipulated object's 
                             translational motion.
    @type             R_beg: numpy.ndarray
    @param            R_beg: Rotation matrix of the manipulated object's 
                             initial pose.

    @rtype:  bool
    @return: B{True} if the trajectory is collision-free.
    """
    T = np.eye(4)
    with self.env:
      self._enable_robots_collision(False)

      for t in np.append(utils.arange(0, translation_traj.duration, 
                         self._query.discr_check_timestep), 
                         translation_traj.duration):
        T[0:3, 0:3] = lie.EvalRotation(R_beg, rot_traj, t)
        T[0:3, 3] = translation_traj.Eval(t)

        self.obj.SetTransform(T)
        in_collision = self.env.CheckCollision(self.obj)
        if in_collision:
          self._enable_robots_collision()
          return False
      
      self._enable_robots_collision()

    return True        
  
  def check_SE3_config_reachability(self, SE3_config):
    """
    Check whether the manipulated object at the given SE3 configuration
    is reachable by both robots.
    @type  SE3_config: SE3Config
    @param SE3_config: SE3 configuration to be checked.

    @rtype:  bool
    @return: B{True} if the IK solutions for both robots exist.
    """
    with self.env:
      self.obj.SetTransform(SE3_config.T)      
      self._enable_robots_collision(False)

      for i in xrange(2):
        T_gripper = np.dot(SE3_config.T, self.bimanual_T_rel[i])
        self.robots[i].Enable(True)
        sol = self.manips[i].FindIKSolution(T_gripper, IK_CHECK_COLLISION)
        if sol is None:
          self._enable_robots_collision()
          return False

    return True

  def check_SE3_traj_reachability(self, lie_traj, translation_traj,
                                  ref_sols, direction=FW):
    """
    Check whether both robots can follow the object moving along the given
    SE3 trajectory. Generate trajectories (including waypoints and timestamps)
    for both robots if they exist.hecked later.

    @type          lie_traj: lie.LieTraj
    @param         rot_traj: Lie trajecoty of the manipulated object.
    @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Trajecoty of the manipulated object's 
                             translational motion.
    @type          ref_sols: list
    @param         ref_sols: list of both robots' initial configuration.
                             This is used as a starting point for object tracking.
    @type         direction: int
    @param        direction: Direction of this tracking. This is to be BW 
                             when this function is called in C{_extend_bw}
                             or C{_connect_bw}.

    @rtype:  bool, list, list
    @return: -  status:        B{True} if the trajectory for both robots exist.
             -  bimanual_wpts: Trajectory of bimanual robots in form
                               of waypoints list.
             -  timestamps:    Timestamps for time parameterization of
                               {bimanual_wpts}.
    """
    return self.bimanual_obj_tracker.plan(lie_traj, translation_traj,
            self.bimanual_T_rel, ref_sols, self._query.discr_timestep,
            self._query.discr_check_timestep, direction)

  def _nearest_neighbor_indices(self, SE3_config, treetype):
    """
    Return indices of C{self.nn} vertices nearest to the given C{SE3_config}
    in the tree specified by C{treetype}.

    @type  SE3_config: SE3Config
    @param SE3_config: SE3 configuration to be used.
    @type    treetype: int
    @param   treetype: Type of the C{CCTree} being examined.

    @rtype:  list
    @return: list of indices of nearest vertices, ordered by
             ascending distance
    """
    if (treetype == FW):
      tree = self._query.tree_start
    else:
      tree = self._query.tree_end
    nv = len(tree)
      
    distance_list = [utils.SE3_distance(SE3_config.T, v.config.SE3_config.T, 
                      1.0 / np.pi, 1.0) for v in tree.vertices]
    distance_heap = heap.Heap(distance_list)
        
    if (self._query.nn == -1):
      # to consider all vertices in the tree as nearest neighbors
      nn = nv
    else:
      nn = min(self._query.nn, nv)
    nnindices = [distance_heap.ExtractMin()[0] for i in range(nn)]
    return nnindices

  @staticmethod
  def visualize_lie_traj(obj, lie_traj, translation_traj, speed=1.0):
    """
    Visualize the given closed-chain trajectory by animating it in 
    openrave viewer.

    @type  cctraj: CCTraj
    @param cctraj: Closed-chain trajectory to be visualized.
    @type   speed: float
    @param  speed: Speed of the visualization.
    """
    sampling_step = 0.01
    refresh_step  = sampling_step / speed

    T_obj = np.eye(4)
    for t in np.append(utils.arange(0, lie_traj.duration, sampling_step), 
                       lie_traj.duration):
      T_obj[0:3, 0:3] = lie_traj.EvalRotation(t)
      T_obj[0:3, 3] = translation_traj.Eval(t)
      obj.SetTransform(T_obj)
      sleep(refresh_step)

  def visualize_regrasp_traj(self, bimanual_traj, speed=1.0):
    sleep(0.5)
    return
    sampling_step = 0.01
    refresh_step  = sampling_step / speed

    for index in bimanual_traj:
      traj = bimanual_traj[index]
      if traj is not None:
        robot = self.robots[index]
        manip = self.manips[index]
        taskmanip = self.taskmanips[index]
        taskmanip.ReleaseFingers()
        robot.WaitForController(0)
        traj_spec = traj.GetConfigurationSpecification()
        traj_duration = traj.GetDuration()
        for t in np.append(utils.arange(0, traj_duration, 
                           sampling_step), traj_duration):
          robot.SetActiveDOFValues(list(traj_spec.ExtractJointValues(
                                   traj.Sample(t), robot, 
                                   manip.GetArmIndices())))
          sleep(refresh_step)
        taskmanip.CloseFingers()
        robot.WaitForController(0)

  def visualize_cctraj(self, cctraj, speed=1.0):
    """
    Visualize the given closed-chain trajectory by animating it in 
    openrave viewer.

    @type  cctraj: CCTraj
    @param cctraj: Closed-chain trajectory to be visualized.
    @type   speed: float
    @param  speed: Speed of the visualization.
    """
    timestamps       = cctraj.timestamps
    lie_traj         = cctraj.lie_traj
    translation_traj = cctraj.translation_traj
    bimanual_trajs   = cctraj.bimanual_trajs

    sampling_step = timestamps[1] - timestamps[0]
    refresh_step  = sampling_step / speed

    timestamp_index = 0
    T_obj = np.eye(4)
    for bimanual_traj in bimanual_trajs:
      if type(bimanual_traj) is not dict:
        for (q_left, q_right) in zip(bimanual_traj[0], bimanual_traj[1]):
          t = timestamps[timestamp_index]
          T_obj[0:3, 0:3] = lie_traj.EvalRotation(t)
          T_obj[0:3, 3] = translation_traj.Eval(t)
          self.obj.SetTransform(T_obj)
          self.robots[0].SetActiveDOFValues(q_left)
          self.robots[1].SetActiveDOFValues(q_right)
          sleep(refresh_step)
          timestamp_index += 1
      else:
        self.visualize_regrasp_traj(bimanual_traj, speed=speed)

  def shortcut(self, query, maxiter=20):
    """
    Shortcut the closed-chain trajectory in the given query (C{query.cctraj}). This
    method replaces the original trajectory with the new one.

    @type    query: CCQuery
    @param   query: The query in which the closed-chain trajectory is to be
                    shortcutted.
    @type  maxiter: int
    @param maxiter: Number of iterations to take.
    """
    # Shortcutting parameters
    min_shortcut_duration = query.interpolation_duration / 2.0
    min_n_timesteps = int(min_shortcut_duration / query.discr_timestep)

    # Statistics
    in_collision_count   = 0
    not_reachable_count  = 0
    not_continuous_count = 0
    not_shorter_count    = 0
    successful_count     = 0

    lie_traj         = query.cctraj.lie_traj
    translation_traj = query.cctraj.translation_traj
    timestamps       = query.cctraj.timestamps[:]
    left_wpts        = query.cctraj.bimanual_wpts[0][:]
    right_wpts       = query.cctraj.bimanual_wpts[1][:]

    self.loose_gripper(query)

    t_begin = time()
    
    for i in xrange(maxiter):  
      self._output_debug('Iteration {0}'.format(i + 1), color='blue', bold=False)

      # Sample two time instants
      timestamps_indices = range(len(timestamps))
      t0_index = _RNG.choice(timestamps_indices[:-min_n_timesteps])
      t1_index = _RNG.choice(timestamps_indices[t0_index + min_n_timesteps:])
      t0 = timestamps[t0_index]
      t1 = timestamps[t1_index]

      # Interpolate a new SE(3) trajectory segment
      new_R0 = lie_traj.EvalRotation(t0)
      new_R1 = lie_traj.EvalRotation(t1)
      new_rot_traj = lie.InterpolateSO3(new_R0, new_R1, 
                                        lie_traj.EvalOmega(t0), 
                                        lie_traj.EvalOmega(t1), 
                                        query.interpolation_duration)
      new_lie_traj = lie.LieTraj([new_R0, new_R1], [new_rot_traj])

      new_translation_traj_str = utils.traj_str_3rd_degree(translation_traj.Eval(t0), translation_traj.Eval(t1), translation_traj.Evald(t0), translation_traj.Evald(t1), query.interpolation_duration)
      new_translation_traj = TrajectoryFromStr(new_translation_traj_str) 
      
      # Check SE(3) trajectory length      
      accumulated_dist     = utils.compute_accumulated_SE3_distance(
                                lie_traj, translation_traj, t0=t0, t1=t1,
                                discr_timestep=query.discr_check_timestep)
      new_accumulated_dist = utils.compute_accumulated_SE3_distance(
                                new_lie_traj, new_translation_traj,
                                discr_timestep=query.discr_check_timestep)

      if new_accumulated_dist >= accumulated_dist:
        not_shorter_count += 1
        self._output_debug('Not shorter', color='yellow', bold=False)
        continue

      # Check collision (object trajectory)
      if not self.is_collision_free_SE3_traj(new_rot_traj, 
                    new_translation_traj, new_R0):          
        in_collision_count += 1
        self._output_debug('In collision', color='yellow', bold=False)
        continue

      # Check reachability (object trajectory)
      passed, bimanual_wpts, new_timestamps = self.check_SE3_traj_reachability(
        lie.LieTraj([new_R0, new_R1], [new_rot_traj]), new_translation_traj,
        [left_wpts[t0_index], right_wpts[t0_index]])

      if not passed:
        not_reachable_count += 1
        self._output_debug('Not reachable', color='yellow', bold=False)
        continue

      # Check continuity between newly generated bimanual_wpts and original one
      eps = 1e-3
      if not (utils.distance(bimanual_wpts[0][0], left_wpts[t0_index]) < eps and utils.distance(bimanual_wpts[1][0], right_wpts[t0_index]) < eps):
        not_continuous_count += 1
        self._output_debug('Not continuous', color='yellow', bold=False)
        continue

      # Now the new trajectory passes all tests
      # Replace all the old trajectory segments with the new ones
      lie_traj = utils.replace_lie_traj_segment(lie_traj, new_lie_traj.trajlist[0],
                                                t0, t1)            
      translation_traj = utils.replace_traj_segment(translation_traj,
                                                    new_translation_traj, t0, t1)
      
      first_timestamp_chunk       = timestamps[:t0_index + 1]
      last_timestamp_chunk_offset = timestamps[t1_index]
      last_timestamp_chunk        = [t - last_timestamp_chunk_offset for t in timestamps[t1_index:]]

      timestamps = utils.merge_timestamps_list([first_timestamp_chunk, new_timestamps, last_timestamp_chunk])
      left_wpts  = utils.merge_wpts_list([left_wpts[:t0_index + 1], bimanual_wpts[0], left_wpts[t1_index:]])            
      right_wpts = utils.merge_wpts_list([right_wpts[:t0_index + 1], bimanual_wpts[1], right_wpts[t1_index:]])
      
      self._output_debug('Shortcutting successful.', color='green', bold=False)
      successful_count += 1

    t_end = time()
    self.reset_config(query)
    self._output_info('Shortcutting done. Total running time : {0} s.'. format(t_end - t_begin), 'green')
    self._output_debug('Successful: {0} times. In collision: {1} times. Not shorter: {2} times. Not reachable: {3} times. Not continuous: {4} times.'.format(successful_count, in_collision_count, not_shorter_count, not_reachable_count, not_continuous_count), 'yellow')

    query.cctraj = CCTrajectory(lie_traj, translation_traj, [left_wpts, right_wpts], timestamps)
    
  def _enable_robots_collision(self, enable=True):
    """
    Enable or disable collision checking for the robots.

    @type  enable: boll
    @param query: B{True} to enable collision checking for the robots.
                    B{False} to disable.
    """
    for robot in self.robots:
      robot.Enable(enable)

  def _output_debug(self, msg, color=None, bold=True):
    """
    Output the given message with specified color and boldness if {self._debug}
    is True, formatted with leading info of the calling function.

    @type    msg: str
    @param   msg: message to be displayed.
    @type  color: str
    @param color: Color of the displayed message.
    @type   bold: bool
    @param  bold: B{True} if the message is to be displayed in bold.
    """
    if self._debug:
      if color is None:
        formatted_msg = msg
      else:
        formatted_msg = colorize(msg, color, bold)
      func_name = traceback.extract_stack(None, 2)[0][2]
      print '[CCPlanner::' + func_name + '] ' + formatted_msg

  def _output_info(self, msg, color=None, bold=True):
    """
    Output the given message with specified color, formatted with
    leading info of the calling function.

    @type    msg: str
    @param   msg: message to be displayed.
    @type  color: str
    @param color: Color of the displayed message.
    @type   bold: bool
    @param  bold: B{True} if the message is to be displayed in bold.
    """
    if color is None:
      formatted_msg = msg
    else:
      formatted_msg = colorize(msg, color, bold)
    func_name = traceback.extract_stack(None, 2)[0][2]
    print '[CCPlanner::' + func_name + '] ' + formatted_msg

class BimanualObjectTracker(object):
  """
  Class containing method for tracking an object with two robots
  in a closed-chain motion.

  Requirements:
  - two identical robots

  TODO: Make it general for mutiple robots of different types.
  """
  
  def __init__(self, robots, obj, debug=False):
    """
    BimanualObjectTracker constructor. It requires infomation of the robots
    and the object being tracked.

    @type  robots: list of openravepy.Robot
    @param robots: List of robots for the closed-chain motion.
    @type     obj: openravepy.KinBody
    @param    obj: Object to be manipulated in the closed-chain motion. It
                   connects the end-effectors of the two robots.
    @type   debug: bool
    @param  debug: B{True} if debug info is to be displayed.
    """
    self.robots = robots
    self.manips = [robot.GetActiveManipulator() for robot in robots]
    self.obj    = obj
    self.env    = obj.GetEnv()

    self._ndof   = robots[0].GetActiveDOF()
    self._debug   = debug
    self._nrobots = len(robots)
    self._vmax    = robots[0].GetDOFVelocityLimits()[0:self._ndof]
    self._jmin    = robots[0].GetDOFLimits()[0][0:self._ndof]
    self._jmax    = robots[0].GetDOFLimits()[1][0:self._ndof]
    self._maxiter = 8
    self._weight  = 10.
    self._tol     = 1e-6

  def update_vmax(self):
    """
    Update attribute C{_vmax} in case velocity limit of the robot changed.
    """
    self._vmax = self.robots[0].GetDOFVelocityLimits()[0:self._ndof]

  def plan(self, lie_traj, translation_traj, bimanual_T_rel, q_robots_init, discr_timestep, discr_check_timestep, direction=FW):
    """
    Plan trajectories for both robots to track the manipulated object.

    @type              lie_traj: lie.LieTraj
    @param             lie_traj: Lie trajectory of the manipulated object.
    @type      translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param     translation_traj: Trajecoty of the manipulated object's 
                                 translational motion.
    @type        bimanual_T_rel: list
    @param       bimanual_T_rel: Relative transformations of the robots'
                                 end-effectors w.r.t. to the object.
    @type         q_robots_init: list
    @param        q_robots_init: Initial configurations of both robots.
    @type        discr_timestep: list
    @param       discr_timestep: Timestep between adjacent waypoints to be
                                 generated.
    @type  discr_check_timestep: list
    @param discr_check_timestep: Timestep for 
                                 -  taking samples in feasibility checking
                                 -  generating waypoints by solving IK. 
                                 (other waypoints are interpolated)
                                 This needs to be multiple of C{discr_timestep} for uniformity in 
                                 trajectory generated.
    @type             direction: int
    @param            direction: Direction of this planning. This is to be BW 
                                 when this function is called in 
                                 C{_extend_bw} or C{_connect_bw}.

    @rtype:  bool, list, list
    @return: -  result:        B{True} if the trajectory for both robots 
                               exist.
             -  bimanual_wpts: Trajectory of bimanual robots in form
                               of waypoints list.
             -  timestamps:    Timestamps for time parameterization of
                               {bimanual_wpts}.
    """
    if direction == FW:
      duration = lie_traj.duration

      cycle_length = int(discr_check_timestep / discr_timestep)

      # Trajectory tracking loop
      bimanual_wpts = [[], []] 
      timestamps = []    

      for i in xrange(self._nrobots):
        bimanual_wpts[i].append(q_robots_init[i])
      timestamps.append(0.)

      # Check feasibility and compute IK only once per cycle to ensure speed
      # Other IK solutions are generated by interpolation
      self._jd_max = self._vmax * discr_check_timestep

      t_prev = 0
      q_robots_prev = q_robots_init
      T_obj_prev = np.eye(4)
      T_obj_prev[0:3, 0:3] = lie_traj.EvalRotation(t_prev)
      T_obj_prev[0:3, 3] = translation_traj.Eval(t_prev)

      T_obj = np.eye(4)
      for t in np.append(utils.arange(discr_check_timestep, duration, 
                         discr_check_timestep), duration):
        T_obj[0:3, 0:3] = lie_traj.EvalRotation(t)
        T_obj[0:3, 3] = translation_traj.Eval(t)

        bimanual_T_gripper = []
        for i in xrange(self._nrobots):
          bimanual_T_gripper.append(np.dot(T_obj, bimanual_T_rel[i]))
        
        q_robots_new = []
        for i in xrange(self._nrobots):
          need_regrasp, q_sol = self._compute_IK(i, bimanual_T_gripper[i],
                                                 q_robots_prev[i])
          if q_sol is None:
            return False, False, [], []

          if need_regrasp:
            return True, i, q_sol, q_robots_prev, T_obj_prev 

          else:
            q_robots_new.append(q_sol)

        if not self._is_feasible_bimanual_config(q_robots_new, q_robots_prev, T_obj):
          return False, False, [], []

        # New bimanual config now passed all checks
        # Interpolate waypoints in-between
        for i in xrange(self._nrobots):
          bimanual_wpts[i] += utils.discretize_wpts(q_robots_prev[i], q_robots_new[i], cycle_length)
        timestamps += list(np.linspace(t_prev+discr_timestep, 
                                       t, cycle_length))
        t_prev = t
        q_robots_prev = np.array(q_robots_new)
        T_obj_prev = np.array(T_obj)
      
      return False, True, bimanual_wpts, timestamps


    else:
      duration = lie_traj.duration

      cycle_length = int(discr_check_timestep / discr_timestep)

      # Trajectory tracking loop
      bimanual_wpts = [[], []] 
      timestamps = []    

      for i in xrange(self._nrobots):
        bimanual_wpts[i].append(q_robots_init[i])
      timestamps.append(duration)

      # Check feasibility and compute IK only once per cycle to ensure speed
      # Other IK solutions are generated by interpolation
      self._jd_max = self._vmax * discr_check_timestep
      
      q_robots_prev = q_robots_init
      t_prev = duration
      T_obj_prev = np.eye(4)
      T_obj_prev[0:3, 0:3] = lie_traj.EvalRotation(t_prev)
      T_obj_prev[0:3, 3] = translation_traj.Eval(t_prev)

      T_obj = np.eye(4)
      for t in np.append(utils.arange(duration-discr_check_timestep, 0, 
                                      -discr_check_timestep), 0):
        T_obj[0:3, 0:3] = lie_traj.EvalRotation(t)
        T_obj[0:3, 3] = translation_traj.Eval(t)

        bimanual_T_gripper = []
        for i in xrange(self._nrobots):
          bimanual_T_gripper.append(np.dot(T_obj, bimanual_T_rel[i]))
        
        q_robots_new = []
        for i in xrange(self._nrobots):
          need_regrasp, q_sol = self._compute_IK(i, bimanual_T_gripper[i],
                                                 q_robots_prev[i])
          if q_sol is None:
            return False, False, [], []

          if need_regrasp:
            return True, i, q_sol, q_robots_prev, T_obj_prev 
          else:
            q_robots_new.append(q_sol)

        if not self._is_feasible_bimanual_config(q_robots_new, q_robots_prev, T_obj):
          return False, False, [], []

        # New bimanual config now passed all checks
        # Interpolate waypoints in-between
        for i in xrange(self._nrobots):
          bimanual_wpts[i] += utils.discretize_wpts(q_robots_prev[i], q_robots_new[i], cycle_length)
        timestamps += list(np.linspace(t_prev-discr_timestep, t, cycle_length))
        t_prev = t
        q_robots_prev = np.array(q_robots_new)
        T_obj_prev = np.array(T_obj)

      # Reverse waypoints and timestamps
      bimanual_wpts[0].reverse()
      bimanual_wpts[1].reverse()
      timestamps.reverse()
      return False, True, bimanual_wpts, timestamps


  def _is_feasible_bimanual_config(self, q_robots, q_robots_prev, T_obj):
    """
    Check whether the bimanual configuration is feasible.

    @type       q_robots: list
    @param      q_robots: Configurations of both robots
    @type  q_robots_prev: list
    @param q_robots_prev: Configurations of both robots at previous timestamp.
                          This is used for velocity checking.
    @type          T_obj: numpy.ndarray
    @param         T_obj: Transformation matrix of the object.

    @rtype:  bool
    @return: B{True} if the configuration is feasible.
    """

    # Check robot DOF velocity limits (position limits already 
    # checked in _compute_IK)
    for i in xrange(self._nrobots):
      for j in xrange(self._ndof):
        if abs(q_robots[i][j] - q_robots_prev[i][j]) > self._jd_max[j]:
          self._output_debug('Move too FFFAASSSTTTT', 'red')
          return False

    # Update environment for collision checking
    self.obj.SetTransform(T_obj)
    for i in xrange(self._nrobots):
      self.robots[i].SetActiveDOFValues(q_robots[i])

    # Check collision
    for robot in self.robots:
      if self.env.CheckCollision(robot) or robot.CheckSelfCollision():
        return False

    return True

  def _compute_IK(self, robot_index, T, q):
    """    
    Return an IK solution for a robot reaching an end-effector transformation
    using differential IK.

    @type  robot_index: int
    @param robot_index: Index specifying which robot in C{self.robots} to use 
    @type            T: numpy.ndarray
    @param           T: Goal transformation matrix of the robot's end-effector.
    @type            q: list
    @param           q: Initial configuration of the robot.

    @rtype:  numpy.ndarray
    @return: IK solution computed. B{None} if no solution exist.
    """
    q_orig = np.array(q)
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    target_pose = np.hstack([orpy.quatFromRotationMatrix(R), p])
    if target_pose[0] < 0:
      target_pose[0:4] *= -1.

    reached = False
    for i in xrange(self._maxiter):
      q_delta = self._compute_q_delta(robot_index, target_pose, q)
      q = q + q_delta

      # Ensure IK solution returned is within joint position limit
      q = np.maximum(np.minimum(q, self._jmax), self._jmin)

      cur_objective = self._compute_objective(robot_index, target_pose, q)
      if cur_objective < self._tol:
        reached = True
        break
    if not reached:
      self._output_debug('Max iteration ({0}) exceeded.'.format(self._maxiter), 'red')
      if not self._reach_joint_limit(q): # reach manifold boundry
        self._output_debug('stop NOT by joint limit', 'red')
        return False, None

      self._output_debug('stop IS by joint limit', 'green')
      self.obj.Enable(False) # TODO: this is not accurate
      manip = self.manips[robot_index]
      if manip.FindIKSolution(T, IK_CHECK_COLLISION) is None:
        self.obj.Enable(True) # TODO: this is not accurate
        self._output_debug('No other IK solutions', 'yellow')
        return False, None

      T_cur = utils.compute_endeffector_transform(manip, q)
      sorted_sols = self._sort_IK(manip.FindIKSolutions(T_cur, 
                                  IK_CHECK_COLLISION))
      self.obj.Enable(True) # TODO: this is not accurate

      # try to go back with new IK class
      T_orig = utils.compute_endeffector_transform(manip, q_orig)
      orig_pose = np.hstack([orpy.quatFromRotationMatrix(T_orig[0:3, 0:3]),
                             T_orig[0:3, 3]])
      for sol in sorted_sols:
        q = np.array(sol)
        for i in xrange(self._maxiter):
          q_delta = self._compute_q_delta(robot_index, orig_pose, q)
          q = q + q_delta
          # Ensure IK solution returned is within joint position limit
          q = np.maximum(np.minimum(q, self._jmax), self._jmin)

          cur_objective = self._compute_objective(robot_index, orig_pose, q)
          if cur_objective < self._tol:
            return True, q # back to original transformation with a new q
      self._output_debug('No other free IK solutions', 'yellow')
      return False, None

    return False, q

  def _sort_IK(self, sols):
    """
    Return a sorted list of IK solutions according to their scores.
    """
    feasible_IKs = []
    scores = []
    for sol in sols:
      if not self._reach_joint_limit(sol): # remove IK at joint limit
        feasible_IKs.append(sol)
        scores.append(np.dot(self._jmax - sol, sol - self._jmin))
    sorted_IKs = np.array(feasible_IKs)[np.array(scores).argsort()[::-1]]
    return sorted_IKs

  def _reach_joint_limit(self, q):
    return np.isclose(self._jmax, q).any() or np.isclose(self._jmin, q).any()

  def _compute_objective(self, robot_index, target_pose, q):
    """    
    Return difference between the robot's target pose and current pose.

    @type  robot_index: int
    @param robot_index: Index specifying which robot in C{self.robots} to use 
    @type  target_pose: numpy.ndarray
    @param target_pose: Target pose.
    @type            q: list
    @param           q: Current configuration of the robot.

    @rtype:  numpy.ndarray
    @return: Difference between the robot's target pose and current pose.
    """
    with self.robots[robot_index]:
      self.robots[robot_index].SetActiveDOFValues(q)
      cur_pose = self.manips[robot_index].GetTransformPose()

    if not utils._is_close_axis(cur_pose[1:4], target_pose[1:4]):
      cur_pose[0:4] *= -1.

    error = target_pose - cur_pose

    return self._weight * np.dot(error, error)
    
  def _compute_q_delta(self, robot_index, target_pose, q):
    """    
    Return delta q the robot needs to move to reach C{target_pose} using
    Jacobian matrix.

    @type  robot_index: int
    @param robot_index: Index specifying which robot in C{self.robots} to use 
    @type  target_pose: numpy.ndarray
    @param target_pose: Target pose.
    @type            q: list
    @param           q: Current configuration of the robot.

    @rtype:  numpy.ndarray
    @return: Dealta_q the robot needs to move to reach the target pose.
    """
    with self.robots[robot_index]:
      self.robots[robot_index].SetActiveDOFValues(q)
      # Jacobian
      J_trans = self.manips[robot_index].CalculateJacobian()
      J_quat = self.manips[robot_index].CalculateRotationJacobian()
      
      cur_pose = self.manips[robot_index].GetTransformPose()

    if not utils._is_close_axis(cur_pose[1:4], target_pose[1:4]):
      cur_pose[0:4] *= -1.
      J_quat *= -1.
    
    # Full Jacobian
    J = np.vstack([J_quat, J_trans])

    # J is a [7x6] matrix, need to use pinv()
    q_delta = np.dot(np.linalg.pinv(J), (target_pose - cur_pose))
    return q_delta

  def _output_debug(self, msg, color=None, bold=True):
    """
    Output the given message with specified color and boldness if {self._debug}
    is True, formatted with leading info of the calling function.

    @type    msg: str
    @param   msg: message to be displayed.
    @type  color: str
    @param color: Color of the displayed message.
    @type   bold: bool
    @param  bold: B{True} if the message is to be displayed in bold.
    """
    if self._debug:
      if color is None:
        formatted_msg = msg
      else:
        formatted_msg = colorize(msg, color, bold)
      func_name = traceback.extract_stack(None, 2)[0][2]
      print '[BimanualObjectTracker::' + func_name + '] ' + formatted_msg

  def _output_info(self, msg, color=None, bold=True):
    """
    Output the given message with specified color, formatted with
    leading info of the calling function.

    @type    msg: str
    @param   msg: message to be displayed.
    @type  color: str
    @param color: Color of the displayed message.
    @type   bold: bool
    @param  bold: B{True} if the message is to be displayed in bold.
    """
    if color is None:
      formatted_msg = msg
    else:
      formatted_msg = colorize(msg, color, bold)
    func_name = traceback.extract_stack(None, 2)[0][2]
    print '[BimanualObjectTracker::' + func_name + '] ' + formatted_msg
      
class CCPlannerException(Exception):
  """
  Base class for exceptions for cc planners
  """
  pass