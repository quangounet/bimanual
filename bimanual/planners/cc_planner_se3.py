"""Closed-chain motion planner.

This is an RRT-based planner adapted for planner transfer motions for bimanual
manipulation. In each RRT extension step, a trajectory in SE(3) for the movable object is
planned first. Then robot configuration waypoints are computed accordingly to 'track' the
object movement.

The object reference point should be the COM instead of the origin of the local frame.

"""

import openravepy as orpy
import numpy as np
import random
from time import time, sleep
from copy import deepcopy
import TOPP
import cPickle as pickle
from ..utils.loggers import TextColors
from ..utils import utils, heap, lie
from ..utils.trajectory import CCTrajectory

# Global parameters
FW       = 0
BW       = 1
REACHED  = 0
ADVANCED = 1
TRAPPED  = 2

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

    @type  q: numpy.ndarray
    @param q: Quaternion, (s, vx, vy, vz).
    @type  p: numpy.ndarray
    @param p: Translation 3-vector, (x, y, z).
    @type  qd: numop.ndarray
    @param qd: Time derivative of q.
    @type  pd: numpy.ndarray
    @param pd: Time derivative of p (translational velocity).
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
    @rtype: SE3Config
    @return: An SE3Config object initialized using T.
    """    
    if T is None:
      return None

    quat = orpy.quatFromRotationMatrix(T[0:3, 0:3])
    p = T[0:3, 3]
    return SE3Config(quat, p)        


class CCConfig(object):
  """
  Configuration Class to contain configuration information of
  both robots and the object in closed-chain motion.
  """

  def __init__(self, q_robots, SE3_config):
    """
    CCConfig constructor.

    @type  q_robots: list
    @param q_robots: List of onfigurations of both robots in the bimanual set-up.
    @type  SE3_config: SE3Config
    @param SE3_config: Configuration of the manipulated object.
    """
    self.q_robots   = np.array(q_robots)
    self.SE3_config = SE3_config


class CCVertex(object):  
  """
  Vertex of closed-chain motion. It stores all information required
  to generated a C{CCTree} in this planner.
  """

  def __init__(self, config):
    """
    CCVertex constructor.

    @type  config: CCConfig
    @param config: Configuration of the bimanual set-up in this vertex.
    """
    self.config = config

    # These parameters are to be assigned when the vertex is added to the tree
    self.index            = 0
    self.parent_index     = None
    self.rot_traj         = None # TOPP trajectory
    self.translation_traj = None # TOPP trajectory
    self.bimanual_wpts    = []
    self.timestamps       = []
    self.level            = 0
    self.invisible        = False

class CCTree(object):  
  """
  An RRT tree class for planning closed-chain motion.
  """

  def __init__(self, v_root, treetype=FW):
    """
    CCTree constructor.

    @type  v_root: CCVertex
    @param v_root: Root vertex to grow the tree from.
    @type  treetype: int
    @param treetype: The direction of the tree in the closed-chain motion.
                     It can be either forward(C{FW}) or backward(C{BW}).
    """
    self.vertices = []
    self.vertices.append(v_root)
    self.length = 1
    self.treetype = treetype

  def __len__(self):
    return len(self.vertices)

  def __getitem__(self, index):
    return self.vertices[index]        
  
  def add_vertex(self, v_new, parent_index, rot_traj, translation_traj, bimanual_wpts, timestamps, invisible=False):
    """
    Add a C{CCVertex} to the tree.

    @type  v_new: CCVertex
    @param v_new: New vertex to be added.
    @type  parent_index: int
    @param parent_index: Index of C{v_new}'s parent vertex in the tree's C{vertices} list.
    @type  rot_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param rot_traj: Trajecoty of the manipulated object's rotational motion in SO(3) space.
    @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Trajecoty of the manipulated object's translational motion.
    @type  bimanual_wpts: list
    @param bimanual_wpts: Trajectory of bimanual robots in form of waypoints list.
    @type  timestamps: list
    @param timestamps: Timestamps for time parameterization of C{bimanual_wpts}.
    """
    v_new.parent_index     = parent_index
    v_new.rot_traj         = rot_traj
    v_new.translation_traj = translation_traj
    v_new.bimanual_wpts    = bimanual_wpts
    v_new.timestamps       = timestamps
    v_new.index            = self.length
    v_new.level            = self.vertices[parent_index].level + 1
    v_new.invisible        = invisible
    
    self.vertices.append(v_new)
    self.length += 1
    

  def generate_rot_traj_list(self, end_index=-1):
    """
    Return all C{rot_traj} of vertices 
    connecting the specified vertex and C{v_root}.

    @type  end_index: int
    @param end_index: Index of the vertex used as one end of the connection. The other end is C{v_root}.

    @rtype: list
    @return: A list containing all C{rot_traj} of all vertices, starting from vertex with a earlier C{timestamp}.
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
    @param end_index: Index of the vertex used as one end of the connection. The other end is C{v_root}.

    @rtype: list
    @return: A list containing rotation matrices of all vertices, starting from vertex with a earlier C{timestamp}.
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
    @param end_index: Index of the vertex used as one end of the connection. The other end is C{v_root}.

    @rtype: list
    @return: A list containing all C{translation_traj} of all vertices, starting from vertex with a earlier C{timestamp}.
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


  def generate_bimanual_wpts_list(self, end_index=-1):
    """
    Return all C{bimanual_wpts} of vertices 
    connecting the specified vertex and C{v_root}.

    @type  end_index: int
    @param end_index: Index of the vertex used as one end of the connection. The other end is C{v_root}.

    @rtype: list
    @return: A list containing all C{bimanual_wpts} of all vertices, starting from vertex with a earlier C{timestamp}.
    """
    bimanual_wpts_list = [[], []]
      
    vertex = self.vertices[end_index]
    while (vertex.parent_index is not None):
      parent = self.vertices[vertex.parent_index]
      for i in xrange(2):
        bimanual_wpts_list[i].append(vertex.bimanual_wpts[i])
      vertex = parent

    if (self.treetype == FW):
      for i in xrange(2):
        bimanual_wpts_list[i].reverse()
        
    return bimanual_wpts_list


  def generate_timestamps_list(self, end_index=-1):
    """
    Return all C{timestamps} of vertices 
    connecting the specified vertex and C{v_root}.

    @type  end_index: int
    @param end_index: Index of the vertex used as one end of the connection. The other end is C{v_root}.

    @rtype: list
    @return: A list containing all C{timestamps} of all vertices, starting from vertex with a earlier C{timestamp}.
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

  def __init__(self, my_object, obj_translation_limits, q_robots_start, q_robots_goal, 
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
      - interpolated trajectory would result in joint velocity within limit (specifically for denso robot)
      - generated trajectory is smooth
    When user specifies different velocity_scale or step_size, these value 
    are scaled accordingly to satisfy the abovementioned criteria.

    @type  obj_translation_limits: list
    @param obj_translation_limits: Cartesian workspace limits of the object
    @type  q_robots_start: list
    @param q_robots_start: Start configurations of the two robots.
    @type  q_robots_goal: list
    @param q_robots_goal: Goal configurations of the two robots.
    @type  q_robots_grasp: list
    @param q_robots_grasp: Configurations of the two robots' grippers when grasping the object.
    @type  T_obj_start: numpy.ndarray
    @param T_obj_start: Start transformation matrix of the object.
    @type  T_obj_goal: numpy.ndarray
    @param T_obj_goal: Goal transformation matrix of the object. This is optional, since goal transformation can be computed based on robots' goal configurations.
    @type  nn: int
    @param nn: Number of nearest vertices to consider for connection with the new one in extension and connection.
    @type  step_size: float
    @param step_size: Size of each step for tree extension, >=0.1
    @type  velocity_scale: float
    @param velocity_scale: Ratio of the robots' velocity limit to their full velocity.
    @type  interpolation_duration: float
    @param interpolation_duration: Length of time used for interpolating trajectory connecting vertices. This needs to be multiple of C{discr_check_timestep}
    @type  discr_timestep: float
    @param discr_timestep: Timestep between adjacent waypoints.
    @type  discr_check_timestep: float
    @param discr_check_timestep: Timestep for 
      -  taking samples in feasibility checking
      -  generating waypoints by solving IK. (other waypoints are interpolated)
    This needs to be multiple of C{discr_timestep} for uniformity in trajectory generated.
    @type  enable_bw: bool
    @param enable_bw: B{True} to enable extension of C{tree_end}.
    """
    self.my_object = my_object
    
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
    self.bimanual_wpts               = None

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
    Generate final translational trajectory of this query (if solved) and 
    store it in {self.translation_traj}. This trajectory is used for the 
    manipulated object's translational motion.
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

  def generate_final_bimanual_wpts(self):
    """
    Generate final waypoints for both robots in the bimanual set-up of this 
    query (if solved) and store it in {self.bimanual_wpts}. 
    """
    if not self.solved:
      raise CCPlannerException('Query not solved.')
      return

    bimanual_wpts_list = self.tree_start.generate_bimanual_wpts_list()  
    if self.connecting_bimanual_wpts is not None:
      for i in xrange(2):
        bimanual_wpts_list[i].append(self.connecting_bimanual_wpts[i])        
    bimanual_wpts_list_bw = self.tree_end.generate_bimanual_wpts_list()
    for i in xrange(2):
      bimanual_wpts_list[i] += bimanual_wpts_list_bw[i]

    left_wpts = utils.merge_wpts_list(bimanual_wpts_list[0])
    right_wpts = utils.merge_wpts_list(bimanual_wpts_list[1])
    self.bimanual_wpts = [left_wpts, right_wpts]

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
    self.generate_final_bimanual_wpts()
    
    self.cctraj = CCTrajectory.init_with_lie_trans_trajs(self.lie_traj, self.translation_traj, self.bimanual_wpts, self.timestamps, self.discr_timestep)


class CCPlanner(object):
  """
  Closed-chain motion planner for bimanual set-up.

  Requirements:
  - two identical robots
  """
  
  def __init__(self, manip_obj, robots, planner_type='RRTConnect', 
               logger=None):
    """
    CCPlanner constructor. It requires infomation of the robots and the object
    being manipulated.

    @type  manip_obj: pymanip's ORKinBodyWrapper
    @param manip_obj: Object to be manipulated in the closed-chain motion. It connects the end-effectors of the two robots.
    @type  robots: list of openravepy.Robot
    @param robots: List of robots for the closed-chain motion.
    @type  planner_type: str
    @param planner_type: Available planner type: 'BiRRT' and 'RRTConnect'.
    @type  logger: logger
    @param logger: Logger for CCPlanner.
    """
    self.obj = manip_obj
    self.robots = robots
    self.manips = []
    if planner_type not in ('RRTConnect', 'BiRRT'):
      raise CCPlannerException('planner_type has to be either RRTConnect '
                               'or BiRRT.')
    self.planner_type = planner_type
    if logger is None:
      self.logger = TextColors()
    else:
      self.logger = logger
    for (i, robot) in enumerate(self.robots):
      self.manips.append(robot.GetActiveManipulator())
      robot.SetActiveDOFs(self.manips[i].GetArmIndices())

    self.bimanual_obj_tracker = BimanualObjectTracker(self.robots, manip_obj,
                                                      logger=self.logger)

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

    @rtype: SE3Config
    @return: A random SE3 Configuration.
    """ 
    q_rand = lie.RandomQuat()
    p_rand = np.asarray([_RNG.uniform(self._query.lower_limits[i], 
                                      self._query.upper_limits[i]) 
                        for i in xrange(3)])
    # Now perform a quick check and fix in case the object is colliding
    # with the supporting surface
    pose = np.hstack([q_rand, p_rand])
    Tobj = orpy.matrixFromPose(pose)
    # Create a list of vertices of the convex hull of the object. All the
    # vertices are described in the world's frame.
    vertices = [np.dot(Tobj, np.append(v, 1))[0:3]
                for v in self._query.my_object.vertices]
    min_z = min([v[2] for v in vertices])
    diff_z = self._query.my_object.Trest[2, 3] - min_z
    if diff_z > 0:
      # The object is colliding with the supporting surface. Translate the
      # object up along the vertical axis
      p_rand += self._query.my_object.Trest[0:3, 2]*(diff_z + 1e-6)
      Tobj = orpy.matrixFromPose(pose)
    return SE3Config.from_matrix(Tobj)

    return SE3Config(q_rand, p_rand, qd_rand, pd_rand)
  
  def _check_grasping_pose(self):
    """
    Check if the start and goal grasping pose matches; if the check is passed,
    complete tree_end in C{self._query}. Meanwhile, initialize attribute
    C{self.bimanual_T_rel} to store relative pose of the robots' end-effectors
    w.r.t. to the object.
    """
    query = self._query
    # Compute relative transformation from end-effectors to object
    self.bimanual_T_rel = []
    for i in xrange(2):
      self.bimanual_T_rel.append(np.dot(np.linalg.inv(query.v_start.config.SE3_config.T), utils.compute_endeffector_transform(self.manips[i], query.v_start.config.q_robots[i])))

    # Compute object SE3_config at goal if not specified
    if query.v_goal.config.SE3_config is None:
      T_left_robot_goal = utils.compute_endeffector_transform(self.manips[0], query.v_goal.config.q_robots[0])
      T_obj_goal = np.dot(T_left_robot_goal, np.linalg.inv(self.bimanual_T_rel[0]))
      query.v_goal.config.SE3_config = SE3Config.from_matrix(T_obj_goal)

    # Check start and goal grasping pose
    bimanual_goal_rel_T = []
    for i in xrange(2):
      bimanual_goal_rel_T.append(np.dot(np.linalg.inv(query.v_goal.config.SE3_config.T), utils.compute_endeffector_transform(self.manips[i], query.v_goal.config.q_robots[i])))

    if not np.isclose(self.bimanual_T_rel, bimanual_goal_rel_T, atol=1e-3).all():
      raise CCPlannerException('Start and goal grasping pose not matching.')

    # Complete tree_end in the query 
    query.tree_end = CCTree(query.v_goal, BW)


  def loose_gripper(self, query):
    """
    Open grippers of C{self.robots} by a small amount from C{q_robots_grasp}
    stored in C{query}. This is necessary to avoid collision between object
    and gripper in planning.

    @type  query: CCQuery
    @param query: Query used to extract the grippers' grasping configuration.
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

    # query validity check
    # translational limit
    T_obj_start = query.v_start.config.SE3_config.T
    T_obj_goal = query.v_goal.config.SE3_config.T
    if ((T_obj_start[:3,3] < query.lower_limits).any() or 
        (T_obj_start[:3,3] > query.upper_limits).any() or 
        (T_obj_goal[:3,3] < query.lower_limits).any() or 
        (T_obj_goal[:3,3] > query.upper_limits).any()):
      self.logger.logerr('Start or goal object translation exceeds given limits.')
      return False
    # collision
    self.loose_gripper(query)
    self.obj.SetTransform(T_obj_start)
    self.robots[0].SetActiveDOFValues(query.v_start.config.q_robots[0])
    self.robots[1].SetActiveDOFValues(query.v_start.config.q_robots[1])
    in_collision = False
    in_collision |= self.env.CheckCollision(self.obj.mobj)
    for robot in self.robots:
      in_collision |= self.env.CheckCollision(robot)
      in_collision |= robot.CheckSelfCollision()
    self.obj.SetTransform(T_obj_goal)
    self.robots[0].SetActiveDOFValues(query.v_goal.config.q_robots[0])
    self.robots[1].SetActiveDOFValues(query.v_goal.config.q_robots[1])
    in_collision |= self.env.CheckCollision(self.obj.mobj)
    for robot in self.robots:
      in_collision |= self.env.CheckCollision(robot)
      in_collision |= robot.CheckSelfCollision()
    self.reset_config(query)
    if in_collision:
      self.logger.logerr('Start or goal configuration in collision.')
      return False

    self.bimanual_obj_tracker.update_vmax()
    self.logger.loginfo('Query set successfully.')
    return True

  def solve(self, timeout=20):
    """
    Solve the query stored in the planner. 

    @type  timeout: float
    @param timeout: Time limit for solving the query.

    @rtype: int
    @return: Whether the query is solved within given time limit.
    """
    query = self._query
    if query.solved:
      self.logger.loginfo('This query has already been solved.')
      return True

    self.loose_gripper(query)

    t = 0.0
    prev_iter = query.iteration_count
    
    t_begin = time()
    if (self._connect() == REACHED):
      t_end = time()
      query.running_time += (t_end - t_begin)
      
      self.logger.loginfo('Path found. Iterations: {0}. Running time: {1}s.'
                          .format(query.iteration_count, query.running_time))
      query.solved = True
      query.generate_final_cctraj()
      self.reset_config(query)
      return True

    elasped_time = time() - t_begin
    t += elasped_time
    query.running_time += elasped_time

    while (t < timeout):
      query.iteration_count += 1
      self.logger.logdebug('Iteration no. {0}'.format(query.iteration_count))
      t_begin = time()

      SE3_config = self.sample_SE3_config()
      if (self._extend(SE3_config) != TRAPPED):
        self.logger.logdebug('Tree start : {0}; Tree end : {1}'.format(len(query.tree_start.vertices), len(query.tree_end.vertices)))

        if (self._connect() == REACHED):
          t_end = time()
          query.running_time += (t_end - t_begin)
          self.logger.loginfo('Path found. Iterations: {0}. Running time: {1}s.'.format(query.iteration_count, query.running_time))
          query.solved = True
          query.generate_final_cctraj()
          self.reset_config(query)
          return True
        
      elasped_time = time() - t_begin
      t += elasped_time
      query.running_time += elasped_time

    self.logger.loginfo('Timeout {0}s reached after {1} iterations'.format(
                        timeout, query.iteration_count - prev_iter))
    self.reset_config(query)
    return False

  def reset_config(self, query):
    """
    Reset everything to their starting configuration according to the query,
    including re-closing the grippers which were probably opened for planning.
    This is used after planning is done since in planning process robots and 
    object will be moved for collision checking.

    @type  query: CCQuery
    @param query: Query to be used to extract starting configuration.
    """
    for i in xrange(len(self.robots)):
      self.robots[i].SetActiveDOFValues(query.v_start.config.q_robots[i])
      self.robots[i].SetDOFValues([query.q_robots_grasp[i]],
                                  [self.manips[i].GetArmDOF()])
    self.obj.SetTransform(query.v_start.config.SE3_config.T)

  def _extend(self, SE3_config):
    """
    Extend the tree(s) in C{self._query} towards the given SE3 config.

    @type  SE3_config: SE3Config
    @param SE3_config: Configuration towards which the tree will be extended.

    @rtype: int
    @return: Result of this extension attempt. Possible values:
      - B{TRAPPED}:  when the extension fails
      - B{REACHED}:  when the extension reaches the given config
      - B{ADVANCED}: when the tree is extended towards the given config
    """
    if (self._query.iteration_count - 1) % 2 == FW or not self._query.enable_bw:
      return self._extend_fw(SE3_config)
    else:
      return self._extend_bw(SE3_config)

  def _extend_fw(self, SE3_config):
    """
    Extend C{tree_start} (rooted at v_start) towards the given SE3 config.

    @type  SE3_config: SE3Config
    @param SE3_config: Configuration towards which the tree will be extended.

    @rtype: int
    @return: Result of this extension attempt. Possible values:
      - B{TRAPPED}:  when the extension fails
      - B{REACHED}:  when the extension reaches the given config
      - B{ADVANCED}: when the tree is extended towards the given config
    """
    query = self._query
    status = TRAPPED
    nnindices = self._nearest_neighbor_indices(SE3_config, FW, self.planner_type)
    for index in nnindices:
      v_near = query.tree_start[index]
      
      q_beg  = v_near.config.SE3_config.q
      p_beg  = v_near.config.SE3_config.p
      qd_beg = v_near.config.SE3_config.qd
      pd_beg = v_near.config.SE3_config.pd

      q_end = SE3_config.q
      p_end = SE3_config.p
      qd_end = SE3_config.qd
      pd_end = SE3_config.pd

      # Check if SE3_config is too far from v_near.SE3_config
      SE3_dist = utils.SE3_distance(SE3_config.T, v_near.config.SE3_config.T, 1.0 / np.pi, 1.0)
      if SE3_dist <= query.step_size:
        status = REACHED
        new_SE3_config = SE3_config
      else:
        if not utils._is_close_axis(q_beg, q_end):
          q_end = -q_end
        q_end = q_beg + query.step_size * (q_end - q_beg) / SE3_dist
        q_end /= np.sqrt(np.dot(q_end, q_end))

        p_end = p_beg + query.step_size * (p_end - p_beg) / SE3_dist

        new_SE3_config = SE3Config(q_end, p_end, qd_end, pd_end)
        status = ADVANCED
      # Check collision (SE3_config)
      res = self.is_collision_free_SE3_config(new_SE3_config)
      if not res:
        self.logger.logdebug('TRAPPED : SE(3) config in collision')
        status = TRAPPED
        continue

      # Check reachability (SE3_config)
      res = self.check_SE3_config_reachability(new_SE3_config)
      if not res:
        self.logger.logdebug('TRAPPED : SE(3) config not reachable')
        status = TRAPPED
        continue
      
      # Interpolate a SE3 trajectory for the object
      R_beg = orpy.rotationMatrixFromQuat(q_beg)
      R_end = orpy.rotationMatrixFromQuat(q_end)
      rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                    query.interpolation_duration)
      translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

      # Check translational limit
      # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
      # res = utils.check_translation_traj_str_limits(query.upper_limits, query.lower_limits, translation_traj_str)
      # if not res:
      #   self.logger.logdebug('TRAPPED : SE(3) trajectory exceeds translational limit')
      #   status = TRAPPED
      #   continue

      translation_traj = TrajectoryFromStr(translation_traj_str)

      # Check collision (object trajectory)
      res = self.is_collision_free_SE3_traj(rot_traj, translation_traj, R_beg)
      if not res:
        self.logger.logdebug('TRAPPED : SE(3) trajectory in collision')
        status = TRAPPED
        continue
      
      # Check reachability (object trajectory)
      passed, bimanual_wpts, timestamps = self.check_SE3_traj_reachability(
        lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj,
        v_near.config.q_robots)
      if not passed:
        self.logger.logdebug('TRAPPED : SE(3) trajectory not reachable')
        status = TRAPPED
        continue

      # Now this trajectory is alright.
      self.logger.logdebug('Successful : new vertex generated')
      new_q_robots = [wpts[-1] for wpts in bimanual_wpts] 
      new_config = CCConfig(new_q_robots, new_SE3_config)
      v_new = CCVertex(new_config)
      query.tree_start.add_vertex(v_new, v_near.index, rot_traj, translation_traj, bimanual_wpts, timestamps)
      return status
    return status

  def _extend_bw(self, SE3_config):
    """
    Extend C{tree_end} (rooted at v_goal) towards the given SE3 config.

    @type  SE3_config: SE3Config
    @param SE3_config: Configuration towards which the tree will be extended.

    @rtype: int
    @return: Result of this extension attempt. Possible values:
      - B{TRAPPED}:  when the extension fails
      - B{REACHED}:  when the extension reaches the given config
      - B{ADVANCED}: when the tree is extended towards the given config
    """
    query = self._query
    status = TRAPPED
    nnindices = self._nearest_neighbor_indices(SE3_config, BW, self.planner_type)
    for index in nnindices:
      v_near = query.tree_end[index]
      
      q_end  = v_near.config.SE3_config.q
      p_end  = v_near.config.SE3_config.p
      qd_end = v_near.config.SE3_config.qd
      pd_end = v_near.config.SE3_config.pd

      q_beg = SE3_config.q
      p_beg = SE3_config.p
      qd_beg = SE3_config.qd
      pd_beg = SE3_config.pd

      # Check if SE3_config is too far from v_near.SE3_config
      SE3_dist = utils.SE3_distance(SE3_config.T, v_near.config.SE3_config.T, 1.0 / np.pi, 1.0)
      if SE3_dist <= query.step_size:
        status = REACHED
        new_SE3_config = SE3_config
      else:
        if not utils._is_close_axis(q_beg, q_end):
          q_beg = -q_beg
        q_beg = q_end + query.step_size * (q_beg - q_end) / SE3_dist
        q_beg /= np.sqrt(np.dot(q_beg, q_beg))

        p_beg = p_end + query.step_size * (p_beg - p_end) / SE3_dist

        new_SE3_config = SE3Config(q_beg, p_beg, qd_beg, pd_beg)
        status = ADVANCED

      # Check collision (SE3_config)
      res = self.is_collision_free_SE3_config(new_SE3_config)
      if not res:
        self.logger.logdebug('TRAPPED : SE(3) config in collision')
        status = TRAPPED
        continue

      # Check reachability (SE3_config)
      res = self.check_SE3_config_reachability(new_SE3_config)
      if not res:
        self.logger.logdebug('TRAPPED : SE(3) config not reachable')
        status = TRAPPED
        continue

      # Interpolate a SE3 trajectory for the object
      R_beg = orpy.rotationMatrixFromQuat(q_beg)
      R_end = orpy.rotationMatrixFromQuat(q_end)
      rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                    query.interpolation_duration)
      translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

      # Check translational limit
      # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
      # res = utils.check_translation_traj_str_limits(query.upper_limits, query.lower_limits, translation_traj_str)
      # if not res:
      #   self.logger.logdebug('TRAPPED : SE(3) trajectory exceeds translational limit')
      #   status = TRAPPED
      #   continue

      translation_traj = TrajectoryFromStr(translation_traj_str)

      # Check collision (object trajectory)
      res = self.is_collision_free_SE3_traj(rot_traj, translation_traj, R_beg)
      if not res:
        self.logger.logdebug('TRAPPED : SE(3) trajectory in collision')
        status = TRAPPED
        continue
      
      # Check reachability (object trajectory)
      passed, bimanual_wpts, timestamps = self.check_SE3_traj_reachability(
        lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj,
        v_near.config.q_robots, direction=BW)
      if not passed:
        self.logger.logdebug('TRAPPED : SE(3) trajectory not reachable')
        status = TRAPPED
        continue

      # Now this trajectory is alright.
      self.logger.logdebug('Successful : new vertex generated')
      new_q_robots = [wpts[0] for wpts in bimanual_wpts] 
      new_config = CCConfig(new_q_robots, new_SE3_config)
      v_new = CCVertex(new_config)
      query.tree_end.add_vertex(v_new, v_near.index, rot_traj, translation_traj, bimanual_wpts, timestamps)
      return status
    return status

  def _connect(self):
    """
    Connect C{tree_start} and C{tree_end} in C{self._query}.

    @rtype: int
    @return: Result of this connecting attempt. Possible values:
             - B{TRAPPED}: connect failed
             - B{REACHED}: connect succeeded
    """
    if self.planner_type == 'RRTConnect':
      if (self._query.iteration_count - 1) % 2 == FW or not self._query.enable_bw:
        # tree_start has just been extended
        return self._connect_fw_RRTC()
      else:
        # tree_end has just been extended
        return self._connect_bw_RRTC()

    elif self.planner_type == 'BiRRT':
      if (self._query.iteration_count - 1) % 2 == FW or not self._query.enable_bw:
        # tree_start has just been extended
        return self._connect_fw_BRRT()
      else:
        # tree_end has just been extended
        return self._connect_bw_BRRT()
  
  def _connect_fw_BRRT(self):
    """
    Connect the newly added vertex in C{tree_start} to other vertices on
    C{tree_end}.

    @rtype: int
    @return: Result of this connecting attempt. Possible values:
             - B{TRAPPED}: connect failed
             - B{REACHED}: connect succeeded
    """
    query = self._query

    v_test = query.tree_start.vertices[-1]
    nnindices = self._nearest_neighbor_indices(v_test.config.SE3_config, BW, self.planner_type)
    status = TRAPPED
    for index in nnindices:
      v_near = query.tree_end[index]

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
      
      # Interpolate the object trajectory
      R_beg = orpy.rotationMatrixFromQuat(q_beg)
      R_end = orpy.rotationMatrixFromQuat(q_end)
      rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                    query.interpolation_duration)
      translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

      # Check translational limit
      # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
      # if not utils.check_translation_traj_str_limits(query.upper_limits, query.lower_limits, translation_traj_str):
      #   self.logger.logdebug('TRAPPED : SE(3) trajectory exceeds translational limit')
      #   continue

      translation_traj = TrajectoryFromStr(translation_traj_str)

      # Check collision (object trajectory)
      if not self.is_collision_free_SE3_traj(rot_traj, translation_traj, R_beg):
        self.logger.logdebug('TRAPPED : SE(3) trajectory in collision')
        continue
      
      # Check reachability (object trajectory)
      passed, bimanual_wpts, timestamps = \
        self.check_SE3_traj_reachability(
        lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj,
        v_near.config.q_robots, direction=BW)
      if not passed:
        self.logger.logdebug('TRAPPED : SE(3) trajectory not reachable')
        continue

      # Check similarity of terminal IK solutions
      eps = 1e-3
      for i in xrange(2):
        if not utils.distance(v_test.config.q_robots[i], 
                  bimanual_wpts[i][0]) < eps:
          passed = False
          break
      if not passed:
        self.logger.logdebug('TRAPPED : IK solution discrepancy (robot {0})'.format(i))
        continue

      # Now the connection is successful
      query.tree_end.vertices.append(v_near)
      query.connecting_rot_traj         = rot_traj
      query.connecting_translation_traj = translation_traj
      query.connecting_bimanual_wpts    = bimanual_wpts
      query.connecting_timestamps       = timestamps
      status = REACHED
      return status
    return status        


  def _connect_bw_BRRT(self):
    """
    Connect the newly added vertex in C{tree_end} to other vertices on
    C{tree_start}.

    @rtype: int
    @return: Result of this connecting attempt. Possible values:
             - B{TRAPPED}: connect failed
             - B{REACHED}: connect succeeded
    """
    query = self._query

    v_test = query.tree_end.vertices[-1]
    nnindices = self._nearest_neighbor_indices(v_test.config.SE3_config, FW, self.planner_type)
    status = TRAPPED
    for index in nnindices:
      v_near = query.tree_start[index]

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
      
      # Interpolate the object trajectory
      R_beg = orpy.rotationMatrixFromQuat(q_beg)
      R_end = orpy.rotationMatrixFromQuat(q_end)
      rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                    query.interpolation_duration)
      translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

      # Check translational limit
      # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
      # if not utils.check_translation_traj_str_limits(query.upper_limits, query.lower_limits, translation_traj_str):
      #   self.logger.logdebug('TRAPPED : SE(3) trajectory exceeds translational limit')
      #   continue

      translation_traj = TrajectoryFromStr(translation_traj_str)

      # Check collision (object trajectory)
      if not self.is_collision_free_SE3_traj(rot_traj, translation_traj, R_beg):
        self.logger.logdebug('TRAPPED : SE(3) trajectory in collision')
        continue
      
      # Check reachability (object trajectory)
      passed, bimanual_wpts, timestamps = self.check_SE3_traj_reachability(
        lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj,
        v_near.config.q_robots)

      if not passed:
        self.logger.logdebug('TRAPPED : SE(3) trajectory not reachable')
        continue

      # Check similarity of terminal IK solutions
      eps = 1e-3
      for i in xrange(2):
        if not utils.distance(v_test.config.q_robots[i], 
                  bimanual_wpts[i][-1]) < eps:
          passed = False
          break
      if not passed:
        self.logger.logdebug('TRAPPED : IK solution discrepancy (robot {0})'.format(i))
        continue

      # Now the connection is successful
      query.tree_start.vertices.append(v_near)
      query.connecting_rot_traj         = rot_traj
      query.connecting_translation_traj = translation_traj
      query.connecting_bimanual_wpts    = bimanual_wpts
      query.connecting_timestamps       = timestamps
      status = REACHED
      return status
    return status        
     


  def _connect_fw_RRTC(self):
    """
    Connect the newly added vertex in C{tree_start} to other vertices on
    C{tree_end}.

    @rtype: int
    @return: Result of this connecting attempt. Possible values:
             - B{TRAPPED}: connect failed
             - B{REACHED}: connect succeeded
    """
    query = self._query

    v_test = query.tree_start.vertices[-1]
    nnindices = self._nearest_neighbor_indices(v_test.config.SE3_config, BW, self.planner_type)
    for index in nnindices:
      v_near = query.tree_end[index]
      while True:
        q_end  = v_near.config.SE3_config.q
        p_end  = v_near.config.SE3_config.p
        qd_end = v_near.config.SE3_config.qd
        pd_end = v_near.config.SE3_config.pd
        
        q_beg  = v_test.config.SE3_config.q
        p_beg  = v_test.config.SE3_config.p
        qd_beg = v_test.config.SE3_config.qd
        pd_beg = v_test.config.SE3_config.pd

        # Check distance
        SE3_dist = utils.SE3_distance(v_test.config.SE3_config.T, 
                                      v_near.config.SE3_config.T, 
                                      1.0 / np.pi, 1.0)
        if SE3_dist <= query.step_size: # connect directly
          status = REACHED

          # Interpolate the object trajectory
          R_beg = orpy.rotationMatrixFromQuat(q_beg)
          R_end = orpy.rotationMatrixFromQuat(q_end)
          rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                        query.interpolation_duration)
          translation_traj_str = utils.traj_str_3rd_degree(
            p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

          # Check translational limit
          # NB: Can skip this step, since it's not likely the traj will 
          # exceed the limits given that p_beg and p_end are within limits
          # if not utils.check_translation_traj_str_limits(
          #   query.upper_limits, query.lower_limits, translation_traj_str):
          #   self.logger.logdebug('TRAPPED : SE(3) trajectory exceeds '
          #                        'translational limit')
          #   status = TRAPPED
          #   break

          translation_traj = TrajectoryFromStr(translation_traj_str)

          # Check collision (object trajectory)
          if not self.is_collision_free_SE3_traj(
            rot_traj, translation_traj, R_beg):
            self.logger.logdebug('TRAPPED : SE(3) trajectory in collision')
            status = TRAPPED
            break
          
          # Check reachability (object trajectory)
          passed, bimanual_wpts, timestamps = \
            self.check_SE3_traj_reachability(
              lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj,
              v_near.config.q_robots, direction=BW)
          if not passed:
            self.logger.logdebug('TRAPPED : SE(3) trajectory not reachable')
            status = TRAPPED
            break

          # Check similarity of terminal IK solutions
          eps = 1e-3
          for i in xrange(2):
            if not utils.distance(v_test.config.q_robots[i], 
                                  bimanual_wpts[i][0]) < eps:
              passed = False
              break
          if not passed:
            self.logger.logdebug('TRAPPED : IK solution discrepancy '
                                 '(robot {0})'.format(i))
            status = TRAPPED
            break

          # Now the connection is successful
          query.tree_end.vertices.append(v_near)
          query.connecting_rot_traj         = rot_traj
          query.connecting_translation_traj = translation_traj
          query.connecting_bimanual_wpts    = bimanual_wpts
          query.connecting_timestamps       = timestamps

          return status

        else: # extend towards target
          status = ADVANCED

          if not utils._is_close_axis(q_beg, q_end):
            q_beg = -q_beg
          q_beg = q_end + query.step_size * (q_beg - q_end) / SE3_dist
          q_beg /= np.sqrt(np.dot(q_beg, q_beg))
          p_beg = p_end + query.step_size * (p_beg - p_end) / SE3_dist
          new_SE3_config = SE3Config(q_beg, p_beg, qd_beg, pd_beg)

          # Check new config collision
          if not self.is_collision_free_SE3_config(new_SE3_config):
            self.logger.logdebug('TRAPPED : SE(3) config in collision')
            status = TRAPPED
            break

          # Check reachability
          if not self.check_SE3_config_reachability(new_SE3_config):
            self.logger.logdebug('TRAPPED : SE(3) config not reachable')
            status = TRAPPED
            break

          # Interpolate the object trajectory
          R_beg = orpy.rotationMatrixFromQuat(q_beg)
          R_end = orpy.rotationMatrixFromQuat(q_end)
          rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                        query.interpolation_duration)
          translation_traj_str = utils.traj_str_3rd_degree(
            p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

          # Check translational limit
          # NB: Can skip this step, since it's not likely the traj will 
          # exceed the limits given that p_beg and p_end are within limits
          # if not utils.check_translation_traj_str_limits(
          #   query.upper_limits, query.lower_limits, translation_traj_str):
          #   self.logger.logdebug('TRAPPED : SE(3) trajectory exceeds '
          #                        'translational limit')
          #   status = TRAPPED
          #   break

          translation_traj = TrajectoryFromStr(translation_traj_str)

          # Check collision (object trajectory)
          if not self.is_collision_free_SE3_traj(
            rot_traj, translation_traj, R_beg):
            self.logger.logdebug('TRAPPED : SE(3) trajectory in collision')
            status = TRAPPED
            break

          # Check reachability (object trajectory)
          passed, bimanual_wpts, timestamps = \
            self.check_SE3_traj_reachability(
              lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj,
              v_near.config.q_robots, direction=BW)
          if not passed:
            self.logger.logdebug('TRAPPED : SE(3) trajectory not reachable')
            status = TRAPPED
            break

          # Now this trajectory is alright.
          self.logger.logdebug('ADVANCED : new vertex generated')
          new_q_robots = [wpts[0] for wpts in bimanual_wpts] 
          new_config = CCConfig(new_q_robots, new_SE3_config)
          v_new = CCVertex(new_config)
          query.tree_end.add_vertex(v_new, v_near.index, rot_traj, 
                                    translation_traj, bimanual_wpts, 
                                    timestamps, invisible=True)
          v_near = v_new

      if status == TRAPPED:
        v_near.invisible = False # validate last vertex of this attempt

    return status

  def _connect_bw_RRTC(self):
    """
    Connect the newly added vertex in C{tree_end} to other vertices on
    C{tree_start}.

    @rtype: int
    @return: Result of this connecting attempt. Possible values:
             - B{TRAPPED}: connect failed
             - B{REACHED}: connect succeeded
    """
    query = self._query

    v_test = query.tree_end.vertices[-1]
    nnindices = self._nearest_neighbor_indices(v_test.config.SE3_config, FW, self.planner_type)
    for index in nnindices:
      v_near = query.tree_start[index]
      while True:
        q_end  = v_test.config.SE3_config.q
        p_end  = v_test.config.SE3_config.p
        qd_end = v_test.config.SE3_config.qd
        pd_end = v_test.config.SE3_config.pd
        
        q_beg  = v_near.config.SE3_config.q
        p_beg  = v_near.config.SE3_config.p
        qd_beg = v_near.config.SE3_config.qd
        pd_beg = v_near.config.SE3_config.pd

        # Check distance
        SE3_dist = utils.SE3_distance(v_near.config.SE3_config.T, 
                                      v_test.config.SE3_config.T, 
                                      1.0 / np.pi, 1.0)
        if SE3_dist <= query.step_size: # connect directly
          status = REACHED

          # Interpolate the object trajectory
          R_beg = orpy.rotationMatrixFromQuat(q_beg)
          R_end = orpy.rotationMatrixFromQuat(q_end)
          rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                        query.interpolation_duration)
          translation_traj_str = utils.traj_str_3rd_degree(
            p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

          # Check translational limit
          # NB: Can skip this step, since it's not likely the traj will 
          # exceed the limits given that p_beg and p_end are within limits
          if not utils.check_translation_traj_str_limits(
            query.upper_limits, query.lower_limits, translation_traj_str):
            self.logger.logdebug('TRAPPED : SE(3) trajectory exceeds '
                                 'translational limit')
            status = TRAPPED
            break

          translation_traj = TrajectoryFromStr(translation_traj_str)

          # Check collision (object trajectory)
          if not self.is_collision_free_SE3_traj(
            rot_traj, translation_traj, R_beg):
            self.logger.logdebug('TRAPPED : SE(3) trajectory in collision')
            status = TRAPPED
            break
          
          # Check reachability (object trajectory)
          passed, bimanual_wpts, timestamps = \
            self.check_SE3_traj_reachability(
              lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj,
              v_near.config.q_robots)
          if not passed:
            self.logger.logdebug('TRAPPED : SE(3) trajectory not reachable')
            status = TRAPPED
            break

          # Check similarity of terminal IK solutions
          eps = 1e-3
          for i in xrange(2):
            if not utils.distance(v_test.config.q_robots[i], 
                      bimanual_wpts[i][-1]) < eps:
              passed = False
              break
          if not passed:
            self.logger.logdebug('TRAPPED : IK solution discrepancy '
                                 '(robot {0})'.format(i))
            status = TRAPPED
            break

          # Now the connection is successful
          query.tree_start.vertices.append(v_near)
          query.connecting_rot_traj         = rot_traj
          query.connecting_translation_traj = translation_traj
          query.connecting_bimanual_wpts    = bimanual_wpts
          query.connecting_timestamps       = timestamps

          return status

        else: # extend towards target
          status = ADVANCED

          if not utils._is_close_axis(q_beg, q_end):
            q_end = -q_end
          q_end = q_beg + query.step_size * (q_end - q_beg) / SE3_dist
          q_end /= np.sqrt(np.dot(q_end, q_end))
          p_end = p_beg + query.step_size * (p_end - p_beg) / SE3_dist
          new_SE3_config = SE3Config(q_end, p_end, qd_end, pd_end)

          # Check new config collision
          if not self.is_collision_free_SE3_config(new_SE3_config):
            self.logger.logdebug('TRAPPED : SE(3) config in collision')
            status = TRAPPED
            break

          # Check reachability
          if not self.check_SE3_config_reachability(new_SE3_config):
            self.logger.logdebug('TRAPPED : SE(3) config not reachable')
            status = TRAPPED
            break

          # Interpolate the object trajectory
          R_beg = orpy.rotationMatrixFromQuat(q_beg)
          R_end = orpy.rotationMatrixFromQuat(q_end)
          rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                        query.interpolation_duration)
          translation_traj_str = utils.traj_str_3rd_degree(
            p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

          # Check translational limit
          # NB: Can skip this step, since it's not likely the traj will 
          # exceed the limits given that p_beg and p_end are within limits
          if not utils.check_translation_traj_str_limits(
            query.upper_limits, query.lower_limits, translation_traj_str):
            self.logger.logdebug('TRAPPED : SE(3) trajectory exceeds '
                                 'translational limit')
            status = TRAPPED
            break

          translation_traj = TrajectoryFromStr(translation_traj_str)

          # Check collision (object trajectory)
          if not self.is_collision_free_SE3_traj(
            rot_traj, translation_traj, R_beg):
            self.logger.logdebug('TRAPPED : SE(3) trajectory in collision')
            status = TRAPPED
            break
          
          # Check reachability (object trajectory)
          passed, bimanual_wpts, timestamps = \
            self.check_SE3_traj_reachability(
              lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj,
              v_near.config.q_robots)
          if not passed:
            self.logger.logdebug('TRAPPED : SE(3) trajectory not reachable')
            status = TRAPPED
            break

          # Now this trajectory is alright.
          self.logger.logdebug('ADVANCED : new vertex generated')
          new_q_robots = [wpts[-1] for wpts in bimanual_wpts] 
          new_config = CCConfig(new_q_robots, new_SE3_config)
          v_new = CCVertex(new_config)
          query.tree_start.add_vertex(v_new, v_near.index, rot_traj,
                                      translation_traj, bimanual_wpts,
                                      timestamps, invisible=True)
          v_near = v_new

      if status == TRAPPED:
        v_near.invisible = False # validate last vertex of this attempt

    return status

  def is_collision_free_SE3_config(self, SE3_config):
    """
    Check whether the given C{SE3_config} is collision-free.
    This check ignores the robots, which will be checked later.

    @type  SE3_config: SE3Config
    @param SE3_config: SE3 configuration to be checked.

    @rtype: bool
    @return: B{True} if the config is collision-free.
    """
    self._enable_robots_collision(False)
    self.obj.SetTransform(SE3_config.T)
    is_free = not self.env.CheckCollision(self.obj.mobj)
    self._enable_robots_collision()
    return is_free
  
  def is_collision_free_SE3_traj(self, rot_traj, translation_traj, R_beg):
    """
    Check whether the given SE3 trajectory is collision-free.
    This check ignores the robots, which will be checked later.

    @type  rot_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param rot_traj: Trajecoty of the manipulated object's rotational motion in SO(3) space.
    @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Trajecoty of the manipulated object's translational motion.
    @type  R_beg: numpy.ndarray
    @param R_beg: Rotation matrix of the manipulated object's initial pose.

    @rtype: bool
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
        in_collision = self.env.CheckCollision(self.obj.mobj)
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

    @rtype: bool
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

    @type  lie_traj: lie.LieTraj
    @param rot_traj: Lie trajecoty of the manipulated object.
    @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Trajecoty of the manipulated object's translational motion.
    @type  ref_sols: list
    @param ref_sols: list of both robots' initial configuration. This is used as a starting point for object tracking.
    @type  direction: int
    @param direction: Direction of this tracking. This is to be BW when this function is called in C{_extend_bw} or C{_connect_bw}.

    @rtype: bool, list, list
    @return: - status: B{True} if the trajectory for both robots exist.
             - bimanual_wpts: Trajectory of bimanual robots in form of waypoints list.
             - timestamps: Timestamps for time parameterization of C{bimanual_wpts}.
    """
    return self.bimanual_obj_tracker.plan(lie_traj, translation_traj,
            self.bimanual_T_rel, ref_sols, self._query.discr_timestep,
            self._query.discr_check_timestep, direction)

  def _nearest_neighbor_indices(self, SE3_config, treetype, planner_type):
    """
    Return indices of C{self.nn} vertices nearest to the given C{SE3_config}
    in the tree specified by C{treetype}.

    @type  SE3_config: SE3Config
    @param SE3_config: SE3 configuration to be used.
    @type  treetype: int
    @param treetype: Type of the C{CCTree} being examined.
    @type  planner_type: str
    @param planner_type: Data structure used for planning.

    @rtype: list
    @return: list of indices of nearest vertices, ordered by ascending distance
    """
    if (treetype == FW):
      tree = self._query.tree_start
    else:
      tree = self._query.tree_end
    nv = 0
    if planner_type == 'RRTConnect':
      for v in tree.vertices:
        if not v.invisible:
          nv += 1
    elif planner_type == 'BiRRT':
      nv = len(tree)

    distance_list = [utils.SE3_distance(SE3_config.T, v.config.SE3_config.T, 
                      1.0 / np.pi, 1.0) for v in tree.vertices]
    distance_heap = heap.Heap(distance_list)
        
    if (self._query.nn == -1):
      # to consider all vertices in the tree as nearest neighbors
      nn = nv
    else:
      nn = min(self._query.nn, nv)

    if planner_type == 'RRTConnect':
      i = 0
      nnindices = []
      while i < nn:
        vertex_index = distance_heap.ExtractMin()[0]
        if not tree[vertex_index].invisible:
          i += 1
          nnindices.append(vertex_index)
    elif planner_type == 'BiRRT':
      nnindices = [distance_heap.ExtractMin()[0] for i in range(nn)]

    return nnindices

  def visualize_cctraj(self, cctraj, speed=1.0):
    """
    Visualize the given closed-chain trajectory by animating it in 
    openrave viewer.

    @type  cctraj: CCTraj
    @param cctraj: Closed-chain trajectory to be visualized.
    @type  speed: float
    @param speed: Speed of the visualization.
    """
    timestamps = cctraj.timestamps
    se3_traj   = cctraj.se3_traj
    left_wpts  = cctraj.bimanual_wpts[0]
    right_wpts = cctraj.bimanual_wpts[1]

    sampling_step = cctraj.timestep
    refresh_step  = sampling_step / speed

    for (q_left, q_right, t) in zip(left_wpts, right_wpts, timestamps):
      self.obj.SetTransform(se3_traj.Eval(t))
      self.robots[0].SetActiveDOFValues(q_left)
      self.robots[1].SetActiveDOFValues(q_right)
      sleep(refresh_step)

  def shortcut(self, query, maxiter=20):
    """
    Shortcut the closed-chain trajectory in the given query 
    (C{query.cctraj}). This method replaces the original trajectory with the 
    new one.

    @type  query: CCQuery
    @param query: The query in which the closed-chain trajectory is to be shortcutted.
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

    lie_traj         = query.cctraj.se3_traj.lie_traj
    translation_traj = query.cctraj.se3_traj.translation_traj
    timestamps       = query.cctraj.timestamps[:]
    left_wpts        = query.cctraj.bimanual_wpts[0][:]
    right_wpts       = query.cctraj.bimanual_wpts[1][:]

    self.loose_gripper(query)

    t_begin = time()
    
    for i in xrange(maxiter):  
      self.logger.logdebug('Iteration {0}'.format(i + 1))

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
        self.logger.logdebug('Not shorter')
        continue

      # Check collision (object trajectory)
      if not self.is_collision_free_SE3_traj(new_rot_traj, 
                    new_translation_traj, new_R0):          
        in_collision_count += 1
        self.logger.logdebug('In collision')
        continue

      # Check reachability (object trajectory)
      passed, bimanual_wpts, new_timestamps = self.check_SE3_traj_reachability(
        lie.LieTraj([new_R0, new_R1], [new_rot_traj]), new_translation_traj,
        [left_wpts[t0_index], right_wpts[t0_index]])

      if not passed:
        not_reachable_count += 1
        self.logger.logdebug('Not reachable')
        continue

      # Check continuity between newly generated bimanual_wpts and original one
      eps = 5e-2 # Might be too big!!!
      if not (utils.distance(bimanual_wpts[0][-1], left_wpts[t1_index]) < eps and utils.distance(bimanual_wpts[1][-1], right_wpts[t1_index]) < eps):
        not_continuous_count += 1
        self.logger.logdebug('Not continuous')
        continue

      # Now the new trajectory passes all tests
      # Replace all the old trajectory segments with the new ones
      lie_traj = utils.replace_lie_traj_segment(lie_traj, 
                                                new_lie_traj.trajlist[0],
                                                t0, t1)            
      translation_traj = utils.replace_traj_segment(translation_traj,
                                                    new_translation_traj, 
                                                    t0, t1)
      
      first_timestamp_chunk       = timestamps[:t0_index + 1]
      last_timestamp_chunk_offset = timestamps[t1_index]
      last_timestamp_chunk        = [t - last_timestamp_chunk_offset for t in timestamps[t1_index:]]

      timestamps = utils.merge_timestamps_list([first_timestamp_chunk, new_timestamps, last_timestamp_chunk])
      left_wpts  = utils.merge_wpts_list([left_wpts[:t0_index + 1], bimanual_wpts[0], left_wpts[t1_index:]], eps=eps)            
      right_wpts = utils.merge_wpts_list([right_wpts[:t0_index + 1], bimanual_wpts[1], right_wpts[t1_index:]], eps=eps)
      
      self.logger.logdebug('Shortcutting successful.')
      successful_count += 1

    t_end = time()
    self.reset_config(query)
    self.logger.loginfo('Shortcutting done. Total running time: {0}s.'. format(t_end - t_begin))
    self.logger.logdebug('Successful: {0} times. In collision: {1} times. Not shorter: {2} times. Not reachable: {3} times. Not continuous: {4} times.'.format(successful_count, in_collision_count, not_shorter_count, not_reachable_count, not_continuous_count))

    query.cctraj = CCTrajectory.init_with_lie_trans_trajs(lie_traj, translation_traj, [left_wpts, right_wpts], timestamps, query.discr_timestep)
    
  def _enable_robots_collision(self, enable=True):
    """
    Enable or disable collision checking for the robots.

    @type  enable: boll
    @param enable: B{True} to enable collision checking for the robots. B{False} to disable.
    """
    for robot in self.robots:
      robot.Enable(enable)

class BimanualObjectTracker(object):
  """
  Class containing method for tracking an object with two robots
  in a closed-chain motion.

  Requirements:
  - two identical robots

  TODO: Make it general for mutiple robots of different types.
  """
  
  def __init__(self, robots, obj, logger=None):
    """
    BimanualObjectTracker constructor. It requires infomation of the robots
    and the object being tracked.

    @type  robots: list of openravepy.Robot
    @param robots: List of robots for the closed-chain motion.
    @type  obj: openravepy.KinBody
    @param obj: Object to be manipulated in the closed-chain motion. It connects the end-effectors of the two robots.
    @type  logger: logger type
    @param logger: Desired logger for logging.
    """
    self.robots = robots
    self.manips = [robot.GetActiveManipulator() for robot in robots]
    self.obj    = obj
    self.env    = obj.GetEnv()

    if logger is None:
      self.logger = TextColors()
    else:
      self.logger = logger

    self._ndof   = robots[0].GetActiveDOF()
    self._nrobots = len(robots)
    self._vmax    = robots[0].GetDOFVelocityLimits()[0:self._ndof]
    self._jmin    = robots[0].GetDOFLimits()[0][0:self._ndof]
    self._jmax    = robots[0].GetDOFLimits()[1][0:self._ndof]
    self._maxiter = 8
    self._tol     = 1e-7

  def update_vmax(self):
    """
    Update attribute C{_vmax} in case velocity limit of the robot changed.
    """
    self._vmax = self.robots[0].GetDOFVelocityLimits()[0:self._ndof]

  def plan(self, lie_traj, translation_traj, bimanual_T_rel, q_robots_init, discr_timestep, discr_check_timestep, direction=FW):
    """
    Plan trajectories for both robots to track the manipulated object.

    @type  lie_traj: lie.LieTraj
    @param lie_traj: Lie trajectory of the manipulated object.
    @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Trajecoty of the manipulated object's translational motion.
    @type  bimanual_T_rel: list
    @param bimanual_T_rel: Relative transformations of the robots' end-effectors w.r.t. to the object.
    @type  q_robots_init: list
    @param q_robots_init: Initial configurations of both robots.
    @type  discr_timestep: list
    @param discr_timestep: Timestep between adjacent waypoints to be generated.
    @type  discr_check_timestep: list
    @param discr_check_timestep: Timestep for 
      - taking samples in feasibility checking
      - generating waypoints by solving IK. (other waypoints are interpolated)
    This needs to be multiple of C{discr_timestep} for uniformity in trajectory generated.
    @type  direction: int
    @param direction: Direction of this planning. This is to be BW when this function is called in C{_extend_bw} or C{_connect_bw}.

    @rtype: bool, list, list
    @return: - result: B{True} if the trajectory for both robots exist.
             - bimanual_wpts: Trajectory of bimanual robots in form of waypoints list.
             - timestamps: Timestamps for time parameterization of C{bimanual_wpts}.
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
      q_robots_prev = q_robots_init
      t_prev = 0
      self._jd_max = self._vmax * discr_check_timestep

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
          q_sol = self._compute_IK(i, bimanual_T_gripper[i], q_robots_prev[i])
          if q_sol is None:
            return False, [], []
          q_robots_new.append(q_sol)

        if not self._is_feasible_bimanual_config(q_robots_new, q_robots_prev, T_obj):
          return False, [], []

        # New bimanual config now passed all checks
        # Interpolate waypoints in-between
        for i in xrange(self._nrobots):
          bimanual_wpts[i] += utils.discretize_wpts(q_robots_prev[i], q_robots_new[i], cycle_length)
        timestamps += list(np.linspace(t_prev+discr_timestep, t, cycle_length))
        t_prev = t
        q_robots_prev = q_robots_new
      
      return True, bimanual_wpts, timestamps

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
      q_robots_prev = q_robots_init
      t_prev = duration
      self._jd_max = self._vmax * discr_check_timestep
      
      T_obj = np.eye(4)
      for t in np.append(utils.arange(duration-discr_check_timestep, 
                         0, -discr_check_timestep), 0):
        T_obj[0:3, 0:3] = lie_traj.EvalRotation(t)
        T_obj[0:3, 3] = translation_traj.Eval(t)

        bimanual_T_gripper = []
        for i in xrange(self._nrobots):
          bimanual_T_gripper.append(np.dot(T_obj, bimanual_T_rel[i]))
        
        q_robots_new = []
        for i in xrange(self._nrobots):
          q_sol = self._compute_IK(i, bimanual_T_gripper[i], q_robots_prev[i])
          if q_sol is None:
            return False, [], []
          q_robots_new.append(q_sol)

        if not self._is_feasible_bimanual_config(q_robots_new, q_robots_prev, T_obj):
          return False, [], []

        # New bimanual config now passed all checks
        # Interpolate waypoints in-between
        for i in xrange(self._nrobots):
          bimanual_wpts[i] += utils.discretize_wpts(q_robots_prev[i], q_robots_new[i], cycle_length)
        timestamps += list(np.linspace(t_prev-discr_timestep, t, cycle_length))
        t_prev = t
        q_robots_prev = q_robots_new

      # Reverse waypoints and timestamps
      bimanual_wpts[0].reverse()
      bimanual_wpts[1].reverse()
      timestamps.reverse()
      return True, bimanual_wpts, timestamps


  def _is_feasible_bimanual_config(self, q_robots, q_robots_prev, T_obj):
    """
    Check whether the bimanual configuration is feasible.

    @type  q_robots: list
    @param q_robots: Configurations of both robots
    @type  q_robots_prev: list
    @param q_robots_prev: Configurations of both robots at previous timestamp.
                          This is used for velocity checking.
    @type  T_obj: numpy.ndarray
    @param T_obj: Transformation matrix of the object.

    @rtype: bool
    @return: B{True} if the configuration is feasible.
    """

    # Check robot DOF velocity limits (position limits already 
    # checked in _compute_IK)
    for i in xrange(self._nrobots):
      for j in xrange(self._ndof):
        if abs(q_robots[i][j] - q_robots_prev[i][j]) > self._jd_max[j]:
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
    @type  T: numpy.ndarray
    @param T: Goal transformation matrix of the robot's end-effector.
    @type  q: list
    @param q: Initial configuration of the robot.

    @rtype: numpy.ndarray
    @return: IK solution computed. B{None} if no solution exist.
    """
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
      self.logger.logdebug('Max iteration ({0}) exceeded.'.format(self._maxiter))
      return None

    return q

  def _compute_objective(self, robot_index, target_pose, q):
    """    
    Return difference between the robot's target pose and current pose.

    @type  robot_index: int
    @param robot_index: Index specifying which robot in C{self.robots} to use 
    @type  target_pose: numpy.ndarray
    @param target_pose: Target pose.
    @type  q: list
    @param q: Current configuration of the robot.

    @rtype: numpy.ndarray
    @return: Difference between the robot's target pose and current pose.
    """
    with self.robots[robot_index]:
      self.robots[robot_index].SetActiveDOFValues(q)
      cur_pose = self.manips[robot_index].GetTransformPose()

    if not utils._is_close_axis(cur_pose[1:4], target_pose[1:4]):
      cur_pose[0:4] *= -1.

    error = target_pose - cur_pose
    return np.dot(error, error)
    
  def _compute_q_delta(self, robot_index, target_pose, q):
    """    
    Return delta q the robot needs to move to reach C{target_pose} using
    Jacobian matrix.

    @type  robot_index: int
    @param robot_index: Index specifying which robot in C{self.robots} to use 
    @type  target_pose: numpy.ndarray
    @param target_pose: Target pose.
    @type  q: list
    @param q: Current configuration of the robot.

    @rtype: numpy.ndarray
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

class CCPlannerException(Exception):
  """
  Base class for exceptions for cc planners
  """
  pass
