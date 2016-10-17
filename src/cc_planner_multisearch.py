"""
Closed-chain motion planner for bimual setup.
This one plans one robot first, but sampling tree extension target in 
multi spaces.
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
FW       = 0
BW       = 1
REACHED  = 0
ADVANCED = 1
TRAPPED  = 2
ROBOT    = 0
OBJECT   = 1

IK_CHECK_COLLISION = orpy.IkFilterOptions.CheckEnvCollisions
IK_IGNORE_COLLISION = orpy.IkFilterOptions.IgnoreSelfCollisions
TrajectoryFromStr = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString
_RNG = random.SystemRandom()

class CCTrajectory(object):
  """
  Class of closed-chain trajectory, storing all information needed for 
  a trajectory in closed-chain motions.
  """

  def __init__(self, master_traj, slave_wpts, timestamps, obj_T_rel):
    """
    CCTrajectory constructor.

    @type     bimanual_wpts: list
    @param    bimanual_wpts: Trajectory of bimanual robots in form 
                             of waypoints list.
    @type        timestamps: list
    @param       timestamps: Timestamps for time parameterization 
                             of C{bimanual_wpts}.
    """
    self.master_traj = master_traj
    self.slave_wpts  = slave_wpts
    self.timestamps  = timestamps
    self.obj_T_rel   = obj_T_rel

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
    self.q = q / quat_length
    if qd is None:
      self.qd = np.zeros(3)
    else:
      self.qd = qd

    self.p = p
    if pd is None:
      self.pd = np.zeros(3)
    else:
      self.pd = pd

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

class MasterConfig(object):
  def __init__(self, q, qd=None, qdd=None):
    """
    MasterConfig constructor.

    @type    q_robots: list
    @param   q_robots: List of onfigurations of both robots in the bimanual set-up.
    """
    self.q = q
    if qd is None:
      self.qd = np.zeros(len(self.q))
    else:
      self.qd = qd
    if qdd is None:
      self.qdd = np.zeros(len(self.q))
    else:
      self.qdd = qdd

class CCConfig(object):
  """
  Configuration Class to contain configuration information of
  both robots and the object in closed-chain motion.
  """

  def __init__(self, master_config, q_slave, T_obj=None):
    """
    CCConfig constructor.

    @type    q_robots: list
    @param   q_robots: List of onfigurations of both robots in the bimanual set-up.
    """
    self.master_config = master_config
    self.q_slave = q_slave
    self.T_obj = T_obj

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
    self.index           = 0
    self.parent_index    = None
    self.master_traj_str = ''
    self.slave_wpts      = []
    self.timestamps      = []
    self.level           = 0


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
      self.length += 1

    self.treetype = treetype

  def __len__(self):
    return len(self.vertices)

  def __getitem__(self, index):
    return self.vertices[index]        
  
  def add_vertex(self, v_new, parent_index, master_traj_str,
                 slave_wpts, timestamps):
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
    v_new.parent_index    = parent_index
    v_new.master_traj_str = master_traj_str
    v_new.slave_wpts      = slave_wpts
    v_new.timestamps      = timestamps
    v_new.index           = self.length
    v_new.level           = self.vertices[parent_index].level + 1
    
    self.vertices.append(v_new)
    self.length += 1
    
  def generate_slave_wpts_list(self, end_index=-1):
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
    slave_wpts_list = []
      
    vertex = self.vertices[end_index]
    while vertex.parent_index is not None:
      parent = self.vertices[vertex.parent_index]
      slave_wpts_list.append(vertex.slave_wpts)
      vertex = parent

    if self.treetype == FW:
        slave_wpts_list.reverse()
        
    return slave_wpts_list


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

  def generate_master_traj_str(self, end_index=-1):
    trajs = []

    vertex = self.vertices[end_index]
    while vertex.parent_index is not None:
      parent = self.vertices[vertex.parent_index]
      trajs.append(vertex.master_traj_str)
      vertex = parent

    if (self.treetype == FW):
      trajs.reverse()
    
    res_traj = '' # resulting trajectory
    separator = ''
    for i in range(len(trajs)):
      res_traj += separator
      res_traj += trajs[i]
      separator = '\n'
    return res_traj

class CCQuery(object):
  """
  Class to store all information needed in a closed-chain query.
  """

  def __init__(self, q_master_start, q_slave_start, 
               q_master_goal, q_slave_goal, q_master_grasp, q_slave_grasp,
               T_obj_start,  obj_translation_limits, nn=-1, 
               step_size=0.7, velocity_scale=1, interpolation_duration=None, 
               discr_timestep=5e-3, discr_check_timestep=None, 
               enable_bw=False):
    """
    CCQuery constructor. It is independent of robots to be planned since robot
    info will be stored in planner itself.

    Default step_size (when robot has full velocity, i.e. velocity_scale = 1) 
    for each trajectory interpolation is 0.7, with interpolation_duration = 
    1.25s and discr_check_timestep = 0.025s. Defualt discr_timestep is 0.005s.
    These values are determined by experiments to make sure 
      - planning is not too slow
      - interpolated trajectory would result in joint velocity within limit 
        (specifically for denso robot)
      - generated trajectory is smooth
    When user specifies different velocity_scale or step_size, these value 
    are scaled accordingly to satisfy the abovementioned criteria.

    @type          q_robots_start: list
    @param         q_robots_start: Start configurations of the two robots.
    @type           q_robots_goal: list
    @param          q_robots_goal: Goal configurations of the two robots.
    @type          q_robots_grasp: list
    @param         q_robots_grasp: Configurations of the two robots' grippers
                                   when grasping the object.
    @type             T_obj_start: numpy.ndarray
    @param            T_obj_start: Start transformation matrix of the object.
    @type                      nn: int
    @param                     nn: Number of nearest vertices to consider for
                                   connection with the new one in extension
                                   and connection.
    @type               step_size: float
    @param              step_size: Size of each step for tree extension
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
    self.v_start = CCVertex(CCConfig(MasterConfig(q_master_start), 
                                     q_slave_start, T_obj_start))
    self.v_goal  = CCVertex(CCConfig(MasterConfig(q_master_goal), 
                                     q_slave_goal))

    self.q_master_grasp = q_master_grasp
    self.q_slave_grasp  = q_slave_grasp

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
    self.connecting_master_traj_str = ''
    self.connecting_timestamps      = None
    self.connecting_slave_wpts      = None
    self.master_traj_str            = ''
    self.master_traj                = None
    self.timestamps                 = None
    self.slave_wpts                 = None
    self.obj_T_rel                  = None
    self.slave_T_rel                = None

    # Statistics
    self.running_time    = 0.0
    self.iteration_count = 0
    self.solved          = False

    # Parameters
    self.obj_upper_limits = obj_translation_limits[0]
    self.obj_lower_limits = obj_translation_limits[1]

  def generate_final_master_traj(self): 
    if not self.solved:
      raise CCPlannerException('Query not solved.')
      return

    master_traj_str = ''
    master_traj_str += self.tree_start.generate_master_traj_str()
    if not self.connecting_master_traj_str == '':
      if not master_traj_str == '':
        master_traj_str += '\n'
      master_traj_str += self.connecting_master_traj_str
    master_traj_str_bw = self.tree_end.generate_master_traj_str()
    if not master_traj_str_bw == '':
      master_traj_str += '\n'
      master_traj_str += master_traj_str_bw
    
    self.master_traj = TrajectoryFromStr(master_traj_str)

  def generate_final_slave_wpts(self):
    """
    Generate final waypoints for both robots in the bimanual set-up of this 
    query (if solved) and store it in {self.bimanual_wpts}. 
    """
    if not self.solved:
      raise CCPlannerException('Query not solved.')
      return

    slave_wpts_list = self.tree_start.generate_slave_wpts_list()  
    if self.connecting_slave_wpts is not None:
      slave_wpts_list.append(self.connecting_slave_wpts)        
    slave_wpts_list_bw = self.tree_end.generate_slave_wpts_list()
    slave_wpts_list += slave_wpts_list_bw

    self.slave_wpts = utils.merge_wpts_list(slave_wpts_list)

  def generate_final_timestamps(self):
    """
    Generate final timestamps of this query (if solved) and store it in
    {self.timestamps}. It is used as time discretization together with
    C{self.bimanual_wpts} for both robots' motion.
    """
    if not self.solved:
      raise CCPlannerException('Query not solved.')
      return

    timestamps_list = self.tree_start.generate_timestamps_list()
    if self.connecting_timestamps is not None:
      timestamps_list.append(self.connecting_timestamps)
    timestamps_list += self.tree_end.generate_timestamps_list()

    self.timestamps = utils.merge_timestamps_list(timestamps_list)

  def generate_final_cctraj(self):
    """
    Generate final closed-chain trajectory (C{CCTrajectory}) of this query 
    (if solved) and store it in {self.cctraj}. It combines all the components required for a closed-chain motion.
    """
    if not self.solved:
      raise CCPlannerException('Query not solved.')
      return

    # Generate CCTrajectory components
    self.generate_final_master_traj()
    self.generate_final_slave_wpts()
    self.generate_final_timestamps()
    
    self.cctraj = CCTrajectory(self.master_traj, self.slave_wpts, 
                               self.timestamps, self.obj_T_rel)


class CCPlanner(object):
  """
  A closed-chain motion planner for bimanual set-up.

  Requirements:
  - two identical robots
  """
  
  def __init__(self, manip_obj, master, slave, debug=False):
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
    self.obj          = manip_obj
    self._debug       = debug
    self.master       = master
    self.master_manip = master.GetActiveManipulator()
    self.slave        = slave
    self.slave_manip  = slave.GetActiveManipulator()
    self._ndof        = self.master.GetActiveDOF()
    self.master.SetActiveDOFs(self.master_manip.GetArmIndices())
    self.slave.SetActiveDOFs(self.slave_manip.GetArmIndices())

    self.ms_tracker = MSTracker(master, slave, manip_obj, debug=self._debug)
    self.env = self.obj.GetEnv()

    self._active_dofs = self.master_manip.GetArmIndices()

  def sample_master_config(self):
    [lowerlimits, upperlimits] = self.master.GetDOFLimits()
    vmax = self.master.GetDOFVelocityLimits()[self._active_dofs]
    
    q_rand = np.zeros(self._ndof)
    qd_rand = np.zeros(self._ndof)

    for i in xrange(self._ndof):
      q_rand[i] = _RNG.uniform(lowerlimits[i], upperlimits[i])
      # qd_rand[i] = _RNG.uniform(-vmax[i], vmax[i])

    return MasterConfig(q_rand, qd_rand)


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
    p_rand = np.asarray([_RNG.uniform(self._query.obj_lower_limits[i], 
                        self._query.obj_upper_limits[i]) for i in xrange(3)])

    return (q_rand, p_rand)
  
  def _check_grasping_pose(self):
    """
    Check if the start and goal grasping pose matches; if the check is passed,
    complete tree_end in C{self._query}. Meanwhile, initialize attribute
    C{self.bimanual_T_rel} to store relative pose of the robots' end-effectors
    w.r.t. to the object.
    """
    # Compute relative transformation from end-effectors to object
    self._query.obj_T_rel = np.dot(
      np.linalg.inv(utils.compute_endeffector_transform(
      self.master_manip, self._query.v_start.config.master_config.q)),
      self._query.v_start.config.T_obj)
    self._query.slave_T_rel = np.dot(
      np.linalg.inv(utils.compute_endeffector_transform(
      self.master_manip, self._query.v_start.config.master_config.q)), 
      utils.compute_endeffector_transform(self.slave_manip, 
      self._query.v_start.config.q_slave))

    # Compute T_obj at goal
    T_master_goal = utils.compute_endeffector_transform(
      self.master_manip, self._query.v_goal.config.master_config.q)
    self._query.v_goal.config.T_obj = np.dot(T_master_goal, self._query.obj_T_rel)

    # Check slave start and goal pose
    T_slave_goal_given = utils.compute_endeffector_transform(
      self.slave_manip, self._query.v_goal.config.q_slave)
    T_slave_goal_computed = np.dot(T_master_goal, self._query.slave_T_rel)

    if not np.isclose(T_slave_goal_given, T_slave_goal_computed, 
                      atol=1e-3).all():
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
    self.master.SetDOFValues([query.q_master_grasp*0.7], [self.master_manip.GetArmDOF()])
    self.slave.SetDOFValues([query.q_slave_grasp*0.7], [self.slave_manip.GetArmDOF()])

  def set_query(self, query):
    """
    Set a C{CCQuery} object to the planner for planning. Then checks whether 
    the start and goal grasping pose matches.

    @type  query: CCQuery
    @param query: Query to be used for planning.
    """
    self._query = query
    self._check_grasping_pose()
    self.ms_tracker.update_vmax()

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
    
    t_begin = time()
    if (self._connect() == REACHED):
      self._query.iteration_count += 1
      t_end = time()
      self._query.running_time += (t_end - t_begin)
      
      self._output_info('Path found. Iterations: {0}. Running time: {1}s.'
                        .format(self._query.iteration_count, 
                        self._query.running_time), 'green')
      self._query.solved = True
      self._query.generate_final_cctraj()
      self.reset_config(self._query)
      return True

    elasped_time = time() - t_begin
    t += elasped_time
    self._query.running_time += elasped_time

    while (t < timeout):
      self._query.iteration_count += 1
      self._output_debug('Iteration no. {0}'.format(self._query.iteration_count), 'blue')
      t_begin = time()

      if (self._extend() != TRAPPED):
        self._output_debug('Tree start : {0}; Tree end : {1}'.format(len(self._query.tree_start.vertices), len(self._query.tree_end.vertices)), 'green')

        if (self._connect() == REACHED):
          t_end = time()
          self._query.running_time += (t_end - t_begin)
          self._output_info('Path found. Iterations: {0}. Running time: {1}s.'.format(self._query.iteration_count, self._query.running_time), 'green')
          self._query.solved = True
          self._query.generate_final_cctraj()
          self.reset_config(self._query)
          return True
        
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
    self.master.SetActiveDOFValues(query.v_start.config.master_config.q)
    self.master.SetDOFValues([query.q_master_grasp],
                             [self.master_manip.GetArmDOF()])
    self.slave.SetActiveDOFValues(query.v_start.config.q_slave)
    self.slave.SetDOFValues([query.q_slave_grasp],
                            [self.slave_manip.GetArmDOF()])
    self.obj.SetTransform(query.v_start.config.T_obj)

  def _extend(self):
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
    space_type = random.choice([OBJECT, ROBOT])
    if ((self._query.iteration_count - 1) % 2 == FW 
        or not self._query.enable_bw):
      return self._extend_fw(space_type)
    else:
      return self._extend_bw(space_type)

  def _extend_fw(self, space_type):
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
    if space_type == ROBOT:
      master_config_rand = self.sample_master_config()
      nnindices = self._nearest_neighbor_indices(master_config_rand, FW)
    elif space_type == OBJECT:
      (q_obj_rand, p_obj_rand) = self.sample_SE3_config()
      T_obj_rand = orpy.matrixFromPose(np.hstack([q_obj_rand, p_obj_rand]))
      nnindices = self._nearest_neighbor_indices(T_obj_rand, FW, 
                                                 config_type=OBJECT)

    for index in nnindices:
      v_near = self._query.tree_start[index]
      
      q_master_beg   = v_near.config.master_config.q
      qd_master_beg  = v_near.config.master_config.qd
      qdd_master_beg = v_near.config.master_config.qdd

      if space_type == ROBOT:
        q_master_end   = master_config_rand.q
        qd_master_end  = master_config_rand.qd
        qdd_master_end = master_config_rand.qdd

        # Check if master_config_rand is too far from v_near
        delta = utils.distance(q_master_beg, q_master_end)
        if np.sqrt(delta) <= self._query.step_size:
          status = REACHED
        else:
          q_master_end = (q_master_beg + self._query.step_size 
                          * (q_master_end - q_master_beg) / np.sqrt(delta))
          status = ADVANCED

        T_master_end = utils.compute_endeffector_transform(
                         self.master_manip, q_master_end)
        T_obj_end = np.dot(T_master_end, self._query.obj_T_rel)

      elif space_type == OBJECT:
        T_obj_beg = v_near.config.T_obj

        # Check if T_obj_rand is too far from v_near
        delta = utils.SE3_distance(T_obj_rand, T_obj_beg, 1.0 / np.pi, 1.0)
        if delta <= self._query.step_size:
          self.master.SetActiveDOFValues(q_master_beg)
          T_obj_end    = T_obj_rand
          T_master_end = np.dot(T_obj_end, 
                                np.linalg.inv(self._query.obj_T_rel))
          q_master_end   = self.master_manip.FindIKSolution(
                            T_master_end, IK_CHECK_COLLISION)
          qd_master_end  = np.zeros(self._ndof)
          qdd_master_end = np.zeros(self._ndof)
          status = REACHED
        else:
          q_beg = orpy.quatFromRotationMatrix(T_obj_beg[0:3, 0:3])
          p_beg = T_obj_beg[0:3, 3]
          q_end = q_obj_rand
          p_end = p_obj_rand

          if not utils._is_close_axis(q_beg, q_end):
            q_end = -q_end
          q_end = q_beg + self._query.step_size * (q_end - q_beg) / delta
          q_end /= np.sqrt(np.dot(q_end, q_end))
          p_end = p_beg + self._query.step_size * (p_end - p_beg) / delta
          T_obj_end = orpy.matrixFromPose(np.hstack([q_end, p_end]))
          T_master_end = np.dot(T_obj_end, 
                                np.linalg.inv(self._query.obj_T_rel))
          q_master_end = self.master_manip.FindIKSolution(
                            T_master_end, IK_CHECK_COLLISION)
          qd_master_end  = np.zeros(self._ndof)
          qdd_master_end = np.zeros(self._ndof)
          status = ADVANCED

      if q_master_end is None:
        self._output_debug('TRAPPED : no master IK solution', bold=False)
        status = TRAPPED
        continue

      # Check collision
      res = self.is_collision_free_master_obj_config(q_master_end, T_obj_end)
      if not res:
        self._output_debug('TRAPPED : master/obj in collision', bold=False)
        status = TRAPPED
        continue

      # Check reachability
      res = self.check_master_config_reachability(q_master_end, T_master_end,
                                                  T_obj_end)
      if not res:
        self._output_debug('TRAPPED : config not reachable', bold=False)
        status = TRAPPED
        continue

      # Interpolate trajectory
      master_traj_str = utils.traj_str_5th_degree(
                          q_master_beg, q_master_end,
                          qd_master_beg, qd_master_end,
                          qdd_master_beg, qdd_master_end,
                          self._query.interpolation_duration)
      # Check master DOF limit
      if not utils.check_traj_str_DOF_limits(self.master, master_traj_str):
        self._output_debug('TRAPPED : interpolated trajectory '
                           'exceeds DOF limits', bold=False)
        status = TRAPPED
        continue

      # Check collision
      master_traj = TrajectoryFromStr(master_traj_str)
      res = self.is_collision_free_master_obj_traj(master_traj)
      if not res:
        self._output_debug('TRAPPED : master/obj trajectory in collision',
                           bold=False)
        status = TRAPPED
        continue
      
      # Check reachability
      passed, slave_wpts, timestamps = self.check_master_traj_reachability(
                                         master_traj, v_near.config.q_slave)
      if not passed:
        self._output_debug('TRAPPED : master trajectory not reachable', 
                           bold=False)
        status = TRAPPED
        continue

      # Now this trajectory is alright.
      self._output_debug('Successful : new vertex generated', 
                         color='green', bold=False)
      new_q_slave       = slave_wpts[-1]
      new_master_config = MasterConfig(q_master_end, qd_master_end, 
                                       qdd_master_end)
      new_cc_config     = CCConfig(new_master_config, new_q_slave, T_obj_end)
      v_new             = CCVertex(new_cc_config)
      self._query.tree_start.add_vertex(v_new, v_near.index, 
                                        master_traj_str, slave_wpts, 
                                        timestamps)
      return status
    return status
      

  def _extend_bw(self, space_type):
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
    if space_type == ROBOT:
      master_config_rand = self.sample_master_config()
      nnindices = self._nearest_neighbor_indices(master_config_rand, BW)
    elif space_type == OBJECT:
      (q_obj_rand, p_obj_rand) = self.sample_SE3_config()
      T_obj_rand = orpy.matrixFromPose(np.hstack([q_obj_rand, p_obj_rand]))
      nnindices = self._nearest_neighbor_indices(T_obj_rand, BW, 
                                                 config_type=OBJECT)

    for index in nnindices:
      v_near = self._query.tree_end[index]
      
      q_master_end   = v_near.config.master_config.q
      qd_master_end  = v_near.config.master_config.qd
      qdd_master_end = v_near.config.master_config.qdd
      
      if space_type == ROBOT:
        q_master_beg   = master_config_rand.q
        qd_master_beg  = master_config_rand.qd
        qdd_master_beg = master_config_rand.qdd

        # Check if master_config_rand is too far from v_near
        delta = utils.distance(q_master_beg, q_master_end)
        if np.sqrt(delta) <= self._query.step_size:
          status = REACHED
        else:        
          q_master_beg = (q_master_end + self._query.step_size 
                          * (q_master_beg - q_master_end) / np.sqrt(delta))
          status = ADVANCED

        T_master_beg = utils.compute_endeffector_transform(
                         self.master_manip, q_master_beg)
        T_obj_beg = np.dot(T_master_beg, self._query.obj_T_rel)

      elif space_type == OBJECT:
        T_obj_end = v_near.config.T_obj

        # Check if T_obj_rand is too far from v_near
        delta = utils.SE3_distance(T_obj_rand, T_obj_end, 1.0 / np.pi, 1.0)
        if delta <= self._query.step_size:
          self.master.SetActiveDOFValues(q_master_end)
          T_obj_beg    = T_obj_rand
          T_master_beg = np.dot(T_obj_beg, 
                                np.linalg.inv(self._query.obj_T_rel))
          q_master_beg   = self.master_manip.FindIKSolution(
                            T_master_beg, IK_CHECK_COLLISION)
          qd_master_beg  = np.zeros(self._ndof)
          qdd_master_beg = np.zeros(self._ndof)
          status = REACHED
        else:
          q_beg = q_obj_rand
          p_beg = p_obj_rand
          q_end = orpy.quatFromRotationMatrix(T_obj_end[0:3, 0:3])
          p_end = T_obj_end[0:3, 3]

          if not utils._is_close_axis(q_beg, q_end):
            q_beg = -q_beg
          q_beg = q_end + self._query.step_size * (q_beg - q_end) / delta
          q_beg /= np.sqrt(np.dot(q_beg, q_beg))
          p_beg = p_end + self._query.step_size * (p_beg - p_end) / delta
          T_obj_beg = orpy.matrixFromPose(np.hstack([q_beg, p_beg]))
          T_master_beg = np.dot(T_obj_beg, 
                                np.linalg.inv(self._query.obj_T_rel))
          q_master_beg = self.master_manip.FindIKSolution(
                            T_master_beg, IK_CHECK_COLLISION)
          qd_master_beg  = np.zeros(self._ndof)
          qdd_master_beg = np.zeros(self._ndof)
          status = ADVANCED

      if q_master_beg is None:
        self._output_debug('TRAPPED : no master IK solution', bold=False)
        status = TRAPPED
        continue

      # Check collision
      res = self.is_collision_free_master_obj_config(q_master_beg, T_obj_beg)
      if not res:
        self._output_debug('TRAPPED : master/obj in collision', bold=False)
        status = TRAPPED
        continue

      # Check reachability
      res = self.check_master_config_reachability(q_master_beg, T_master_beg,
                                                  T_obj_beg)
      if not res:
        self._output_debug('TRAPPED : config not reachable', bold=False)
        status = TRAPPED
        continue

      # Interpolate trajectory
      master_traj_str = utils.traj_str_5th_degree(
                          q_master_beg, q_master_end,
                          qd_master_beg, qd_master_end,
                          qdd_master_beg, qdd_master_end,
                          self._query.interpolation_duration)
      # Check master DOF limit
      if not utils.check_traj_str_DOF_limits(self.master, master_traj_str):
        self._output_debug('TRAPPED : interpolated trajectory '
                           'exceeds DOF limits', bold=False)
        status = TRAPPED
        continue

      # Check collision
      master_traj = TrajectoryFromStr(master_traj_str)
      res = self.is_collision_free_master_obj_traj(master_traj)
      if not res:
        self._output_debug('TRAPPED : master/obj trajectory in collision',
                           bold=False)
        status = TRAPPED
        continue
      
      # Check reachability
      passed, slave_wpts, timestamps = self.check_master_traj_reachability(
        master_traj, v_near.config.q_slave, direction=BW)
      if not passed:
        self._output_debug('TRAPPED : master trajectory not reachable', 
                           bold=False)
        status = TRAPPED
        continue

      # Now this trajectory is alright.
      self._output_debug('Successful : new vertex generated', 
                          color='green', bold=False)
      new_q_slave       = slave_wpts[0]
      new_master_config = MasterConfig(q_master_beg, qd_master_beg, 
                                       qdd_master_beg)
      new_cc_config     = CCConfig(new_master_config, new_q_slave, T_obj_beg)
      v_new             = CCVertex(new_cc_config)
      self._query.tree_end.add_vertex(v_new, v_near.index, 
                                      master_traj_str, slave_wpts, 
                                      timestamps)
      return status
    return status

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
    nnindices = self._nearest_neighbor_indices(
                  v_test.config.master_config, BW)
    status = TRAPPED
    for index in nnindices:
      v_near = self._query.tree_end[index]

      q_master_beg   = v_test.config.master_config.q
      qd_master_beg  = v_test.config.master_config.qd
      qdd_master_beg = v_test.config.master_config.qdd
      
      q_master_end   = v_near.config.master_config.q
      qd_master_end  = v_near.config.master_config.qd
      qdd_master_end = v_near.config.master_config.qdd
      
      # Interpolate trajectory
      master_traj_str = utils.traj_str_5th_degree(
                          q_master_beg, q_master_end,
                          qd_master_beg, qd_master_end,
                          qdd_master_beg, qdd_master_end,
                          self._query.interpolation_duration)
      # Check master DOF limit
      if not utils.check_traj_str_DOF_limits(self.master, master_traj_str):
        self._output_debug('TRAPPED : interpolated trajectory '
                           'exceeds DOF limits', bold=False)
        continue

      # Check collision
      master_traj = TrajectoryFromStr(master_traj_str)
      if not self.is_collision_free_master_obj_traj(master_traj):
        self._output_debug('TRAPPED : master/obj trajectory in collision',
                           bold=False)
        continue
      
      # Check reachability
      passed, slave_wpts, timestamps = self.check_master_traj_reachability(
        master_traj, v_near.config.q_slave, direction=BW)
      if not passed:
        self._output_debug('TRAPPED : master trajectory not reachable', 
                           bold=False)
        continue

      # Check similarity of terminal IK solutions
      eps = 1e-3
      if not utils.distance(v_test.config.q_slave, slave_wpts[0]) < eps:
        self._output_debug('TRAPPED : slave IK solution discrepancy',
                           bold=False)
        continue

      # Now the connection is successful
      self._query.tree_end.vertices.append(v_near)
      self._query.connecting_slave_wpts      = slave_wpts
      self._query.connecting_timestamps      = timestamps
      self._query.connecting_master_traj_str = master_traj_str
      status = REACHED
      return status
    return status        


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
    nnindices = self._nearest_neighbor_indices(
                  v_test.config.master_config, FW)
    status = TRAPPED
    for index in nnindices:
      v_near = self._query.tree_start[index]
      
      q_master_beg   = v_near.config.master_config.q
      qd_master_beg  = v_near.config.master_config.qd
      qdd_master_beg = v_near.config.master_config.qdd
      
      q_master_end   = v_test.config.master_config.q
      qd_master_end  = v_test.config.master_config.qd
      qdd_master_end = v_test.config.master_config.qdd
      
      
      # Interpolate trajectory
      master_traj_str = utils.traj_str_5th_degree(
                          q_master_beg, q_master_end,
                          qd_master_beg, qd_master_end,
                          qdd_master_beg, qdd_master_end,
                          self._query.interpolation_duration)
      # Check master DOF limit
      if not utils.check_traj_str_DOF_limits(self.master, master_traj_str):
        self._output_debug('TRAPPED : interpolated trajectory '
                           'exceeds DOF limits', bold=False)
        continue

      # Check collision
      master_traj = TrajectoryFromStr(master_traj_str)
      if not self.is_collision_free_master_obj_traj(master_traj):
        self._output_debug('TRAPPED : master/obj trajectory in collision',
                           bold=False)
        continue
      
      # Check reachability
      passed, slave_wpts, timestamps = self.check_master_traj_reachability(
        master_traj, v_near.config.q_slave)
      if not passed:
        self._output_debug('TRAPPED : master trajectory not reachable', 
                           bold=False)
        continue

      # Check similarity of terminal IK solutions
      eps = 1e-3
      if not utils.distance(v_test.config.q_slave, slave_wpts[-1]) < eps:
        self._output_debug('TRAPPED : slave IK solution discrepancy',
                           bold=False)
        continue

      # Now the connection is successful
      self._query.tree_start.vertices.append(v_near)
      self._query.connecting_slave_wpts      = slave_wpts
      self._query.connecting_timestamps      = timestamps
      self._query.connecting_master_traj_str = master_traj_str
      status = REACHED
      return status
    return status        

  
  def is_collision_free_master_obj_config(self, q_master, T_obj):
    """
    Check whether the given C{master_config} is collision-free.
    This check ignores the robots, which will be checked later.

    @type  master_config: MasterConfig
    @param master_config: Master robot configuration to be checked.

    @rtype:  bool
    @return: B{True} if the config is collision-free.
    """
    with self.env:
      self.slave.Enable(False)
      self.master.SetActiveDOFValues(q_master)
      self.obj.SetTransform(T_obj)
      is_free = not ((T_obj[:3,3] < self._query.obj_lower_limits).any()
                     or (T_obj[:3,3] > self._query.obj_upper_limits).any()
                     or self.env.CheckCollision(self.obj)
                     or self.env.CheckCollision(self.master)
                     or self.master.CheckSelfCollision())
      self.slave.Enable(True)

    return is_free
  
  def check_master_config_reachability(self, q_master, T_master, T_obj):
    """
    Check whether the manipulated object at the given SE3 configuration
    is reachable by both robots.
    @type  SE3_config: SE3Config
    @param SE3_config: SE3 configuration to be checked.

    @rtype:  bool
    @return: B{True} if the IK solutions for both robots exist.
    """
    with self.env:
      self.master.SetActiveDOFValues(q_master)
      self.obj.SetTransform(T_obj)
      T_slave = np.dot(T_master, self._query.slave_T_rel)
      sol = self.slave_manip.FindIKSolution(T_slave, IK_CHECK_COLLISION)
    if sol is None:
      return False
    return True

  def is_collision_free_master_obj_traj(self, traj):
    """
    This checks object's collision as well
    """
    with self.env:
      self.slave.Enable(False)
      for t in np.append(np.arange(0, traj.duration, 
                         self._query.discr_check_timestep), traj.duration):
        self.master.SetActiveDOFValues(traj.Eval(t))
        T_master = self.master_manip.GetEndEffectorTransform()
        self.obj.SetTransform(np.dot(T_master, self._query.obj_T_rel))
        in_collision = (self.env.CheckCollision(self.master)
                        or self.env.CheckCollision(self.obj)
                        or self.master.CheckSelfCollision())
        if in_collision:
          self.slave.Enable(True)
          return False

    self.slave.Enable(True)
    return True

  def check_master_traj_reachability(self, master_traj,
                                     ref_sol, direction=FW):
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
    passed, slave_wpts, timestamps = self.ms_tracker.plan(
      master_traj, self._query.slave_T_rel, self._query.obj_T_rel, ref_sol, 
      self._query.discr_timestep, self._query.discr_check_timestep, direction)
    if not passed:
      return False, [], []

    return True, slave_wpts, timestamps

  def _nearest_neighbor_indices(self, config_rand, treetype, 
                                config_type=ROBOT):
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
      
    if config_type == ROBOT:
      distance_list = [utils.distance(config_rand.q,
                       v.config.master_config.q) for v in tree.vertices]
    elif config_type == OBJECT:
      distance_list = [utils.SE3_distance(config_rand, v.config.T_obj, 
                       1.0 / np.pi, 1.0) for v in tree.vertices]

    distance_heap = heap.Heap(distance_list)
        
    if (self._query.nn == -1):
      # to consider all vertices in the tree as nearest neighbors
      nn = nv
    else:
      nn = min(self._query.nn, nv)
    nnindices = [distance_heap.ExtractMin()[0] for i in range(nn)]
    return nnindices

  def visualize_cctraj(self, cctraj, speed=1.0):
    """
    Visualize the given closed-chain trajectory by animating it in openrave viewer.

    @type  cctraj: CCTraj
    @param cctraj: Closed-chain trajectory to be visualized.
    @type   speed: float
    @param  speed: Speed of the visualization.
    """
    master_traj = cctraj.master_traj
    slave_wpts  = cctraj.slave_wpts
    timestamps  = cctraj.timestamps
    obj_T_rel   = cctraj.obj_T_rel

    sampling_step = timestamps[1] - timestamps[0]
    refresh_step  = sampling_step / speed

    for (q_slave, t) in zip(slave_wpts, timestamps):
      self.slave.SetActiveDOFValues(q_slave)
      self.master.SetActiveDOFValues(master_traj.Eval(t))
      T_obj = np.dot(self.master_manip.GetEndEffectorTransform(), obj_T_rel)
      self.obj.SetTransform(T_obj)
      sleep(refresh_step)

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
    raise NotImplementedError('shorcut method not implemented yet')
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

class MSTracker(object):
  """
  Class containing method for slave robot(s) to track one master robot
  in a closed-chain motion.

  Requirements:
  - two identical robots

  TODO: Make it general for mutiple robots of different types.
  """
  
  def __init__(self, master, slave, obj, debug=False):
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
    self.master       = master
    self.slave        = slave
    self.master_manip = master.GetActiveManipulator()
    self.slave_manip  = slave.GetActiveManipulator()
    self.obj          = obj
    self.env          = obj.GetEnv()

    self._ndof    = slave.GetActiveDOF()
    self._debug   = debug
    self._vmax    = slave.GetDOFVelocityLimits()[0:self._ndof]
    self._jmax    = slave.GetDOFLimits()[1][0:self._ndof]
    self._maxiter = 8
    self._weight  = 10.
    self._tol     = 1e-6

  def update_vmax(self):
    """
    Update attribute C{_vmax} in case velocity limit of the robot changed.
    """
    self._vmax = self.slave.GetDOFVelocityLimits()[0:self._ndof]

  def plan(self, master_traj, slave_T_rel, obj_T_rel, q_slave_init, 
           discr_timestep, discr_check_timestep, direction=FW):
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
    @type           q_threshold: float
    @param          q_threshold: Distance threshold for predicting whether 
                                 the new end-effector transformation is 
                                 trackable. This value is set to 8x 
                                 query.step_size, determined from experiments.
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
      duration = master_traj.duration
      cycle_length = int(discr_check_timestep / discr_timestep)

      slave_wpts = [q_slave_init] 
      timestamps = [0.0]    

      # Check feasibility and compute IK only once per cycle to ensure speed
      # Other IK solutions are generated by interpolation
      q_slave_prev = q_slave_init
      t_prev = 0.0
      self._jd_max = self._vmax * discr_check_timestep

      for t in np.append(np.arange(discr_check_timestep, duration, discr_check_timestep), duration):
        self.master.SetActiveDOFValues(master_traj.Eval(t))
        T_master = self.master_manip.GetEndEffectorTransform()
        T_slave = np.dot(T_master, slave_T_rel)

        q_slave = self._compute_IK(T_slave, q_slave_prev)
        if q_slave is None:
          return False, [], []

        T_obj = np.dot(T_master, obj_T_rel)
        self.obj.SetTransform(T_obj)
        # Both master and object are in position now
        if not self._is_feasible_slave_config(q_slave, q_slave_prev):
          return False, [], []

        # New bimanual config now passed all checks
        # Interpolate waypoints in-between
        slave_wpts += utils.discretize_wpts(q_slave_prev, q_slave, 
                                            cycle_length)
        timestamps += list(np.linspace(t_prev + discr_timestep, 
                                       t, cycle_length))
        t_prev = t
        q_slave_prev = q_slave
      
      return True, slave_wpts, timestamps

    else:
      duration = master_traj.duration

      cycle_length = int(discr_check_timestep / discr_timestep)

      slave_wpts = [q_slave_init] 
      timestamps = [duration]    

      # Check feasibility and compute IK only once per cycle to ensure speed
      # Other IK solutions are generated by interpolation
      q_slave_prev = q_slave_init
      t_prev = duration
      self._jd_max = self._vmax * discr_check_timestep

      for t in np.append(np.arange(duration-discr_check_timestep, 
                         discr_check_timestep, -discr_check_timestep), 0):
        self.master.SetActiveDOFValues(master_traj.Eval(t))
        T_master = self.master_manip.GetEndEffectorTransform()
        T_slave = np.dot(T_master, slave_T_rel)

        q_slave = self._compute_IK(T_slave, q_slave_prev)
        if q_slave is None:
          return False, [], []

        T_obj = np.dot(T_master, obj_T_rel)
        self.obj.SetTransform(T_obj)
        # Both master and object are in position now
        if not self._is_feasible_slave_config(q_slave, q_slave_prev):
          return False, [], []

        # New bimanual config now passed all checks
        # Interpolate waypoints in-between
        slave_wpts += utils.discretize_wpts(q_slave_prev, q_slave, 
                                            cycle_length)
        timestamps += list(np.linspace(t_prev - discr_timestep, 
                                       t, cycle_length))
        t_prev = t
        q_slave_prev = q_slave

      # Reverse waypoints and timestamps
      slave_wpts.reverse()
      timestamps.reverse()
      
      return True, slave_wpts, timestamps

  def _is_feasible_slave_config(self, q_slave, q_slave_prev):
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

    # Check slave DOF velocity limits (position limits already 
    # checked in _compute_IK)
    for i in xrange(self._ndof):
      if abs(q_slave[i] - q_slave_prev[i]) > self._jd_max[i]:
        return False

    # Update environment for collision checking
    with self.env:
      self.slave.SetActiveDOFValues(q_slave)
      if self.env.CheckCollision(self.slave) or \
         self.slave.CheckSelfCollision():
        return False

    return True

  def _compute_IK(self, T, q):
    """    
    Return an IK solution for a robot reaching an end-effector transformation
    using differential IK.

    @type            T: numpy.ndarray
    @param           T: Goal transformation matrix of the robot's end-effector.
    @type            q: list
    @param           q: Initial configuration of the robot.

    @rtype:  numpy.ndarray
    @return: IK solution computed. B{None} if no solution exist.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    target_pose = np.hstack([orpy.quatFromRotationMatrix(R), p])
    if target_pose[0] < 0:
      target_pose[0:4] *= -1.

    reached = False
    for i in xrange(self._maxiter):
      q_delta = self._compute_q_delta(target_pose, q)
      q = q + q_delta

      # Ensure IK solution returned is within joint position limit
      q = np.maximum(np.minimum(q, self._jmax), -self._jmax)

      cur_objective = self._compute_objective(target_pose, q)
      if cur_objective < self._tol:
        reached = True
        break
    if not reached:
      self._output_debug('Max iteration ({0}) exceeded.'.format(
                            self._maxiter), 'red')
      return None

    return q


  def _compute_objective(self, target_pose, q):
    """    
    Return difference between the robot's target pose and current pose.

    @type  target_pose: numpy.ndarray
    @param target_pose: Target pose.
    @type            q: list
    @param           q: Current configuration of the robot.

    @rtype:  numpy.ndarray
    @return: Difference between the robot's target pose and current pose.
    """
    with self.slave:
      self.slave.SetActiveDOFValues(q)
      cur_pose = self.slave_manip.GetTransformPose()

    if not utils._is_close_axis(cur_pose[1:4], target_pose[1:4]):
      cur_pose[0:4] *= -1.

    error = target_pose - cur_pose

    return self._weight * np.dot(error, error)
    
  def _compute_q_delta(self, target_pose, q):
    """    
    Return delta q the robot needs to move to reach C{target_pose} using
    Jacobian matrix.

    @type  target_pose: numpy.ndarray
    @param target_pose: Target pose.
    @type            q: list
    @param           q: Current configuration of the robot.

    @rtype:  numpy.ndarray
    @return: Dealta_q the robot needs to move to reach the target pose.
    """
    with self.slave:
      self.slave.SetActiveDOFValues(q)
      # Jacobian
      J_trans = self.slave_manip.CalculateJacobian()
      J_quat = self.slave_manip.CalculateRotationJacobian()
      cur_pose = self.slave_manip.GetTransformPose()

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
      print '[MSTracker::' + func_name + '] ' + formatted_msg

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
    print '[MSTracker::' + func_name + '] ' + formatted_msg

class CCPlannerException(Exception):
  """
  Base class for exceptions for cc planners
  """
  pass