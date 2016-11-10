"""
Closed-chain motion planner with multiple regrasping for bimaual setup.
This version is based on a transferable RRT-connect structure.
NB: For regrasp planner, a CCQuery object can only be sent to it for planning 
once. If the planning does not yield a successful outcome, a new CCQuery is
required for planning again.
"""

import openravepy as orpy
import numpy as np
import random
from time import time, sleep
import traceback
import TOPP
from utils.utils import colorize, reverse_traj
from utils import utils, heap, lie
from IPython import embed

# Global parameters
FW = 0
BW = 1

LR = 0
RL = 1

REACHED     = 0
ADVANCED    = 1
TRAPPED     = 2
NEEDREGRASP = 3

FROMEXTEND  = 0
FROMCONNECT = 1

NOREGRASP    = 0
STARTREGRASP = 1
ENDREGRASP   = 2

IK_CHECK_COLLISION  = orpy.IkFilterOptions.CheckEnvCollisions
IK_IGNORE_COLLISION = orpy.IkFilterOptions.IgnoreSelfCollisions
HAS_SOLUTION        = orpy.PlannerStatus.HasSolution
TrajectoryFromStr   = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString
_RNG                = random.SystemRandom()

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

class BimanualRegraspTrajectory(object):
  def __init__(self, trajs={0: None, 1: None}, order=LR):
    self.trajs = dict(trajs)
    self.order = order

  def reverse(self):
    new_bimanual_regrasp_traj = BimanualRegraspTrajectory(order=1-self.order)
    for key in self.trajs.keys():
      if self.trajs[key] is not None:
        new_bimanual_regrasp_traj.trajs[key] = orpy.planningutils.ReverseTrajectory(self.trajs[key])
    return new_bimanual_regrasp_traj

class CCVertex(object):  
  """
  Vertex of closed-chain motion. It stores all information required
  to generated a RRT tree (C{CCTree}) in this planner.
  """

  def __init__(self, q_robots_start=None, q_robots_end=None,
               q_robots_inter=None, SE3_config_start=None, 
               SE3_config_end=None):
    """
    CCVertex constructor.

    """
    self.q_robots_start   = q_robots_start
    self.q_robots_inter   = q_robots_inter
    self.q_robots_end     = q_robots_end
    self.SE3_config_start = SE3_config_start
    self.SE3_config_end   = SE3_config_end

    # These parameters are to be assigned when the vertex is added to the tree
    self.regrasp_count         = 0
    self.index                 = None
    self.parent_index          = None
    self.child_indices         = []
    self.rot_traj              = None # TOPP trajectory
    self.translation_traj      = None # TOPP trajectory
    self.bimanual_wpts         = []
    self.timestamps            = []
    self.level                 = 0
    self.contain_regrasp       = NOREGRASP
    self.bimanual_regrasp_traj = None
    self.filled                = False
    self.type                  = None

  def _add_regrasp_action(self):
    """
    Add regrasp action to a vertex.
    If the vertex contains regrasp already, replace it.
    """
    if self.contain_regrasp == NOREGRASP:
      self.contain_regrasp = ENDREGRASP
      self.regrasp_count += 1

  def _fill_regrasp_traj(self, bimanual_regrasp_traj):
    """
    Add regrasp traj to a vertex containing regrasp action.
    """
    self.bimanual_regrasp_traj = bimanual_regrasp_traj
    self.filled = True

class CCVertexDatabase(object):
  def __init__(self):
    self.vertices = []

  def __len__(self):
    return len(self.vertices)

  def __getitem__(self, index):
    return self.vertices[index]  

  def append(self, vertex, vertex_type):
    vertex.type = vertex_type
    vertex.index = len(self.vertices)
    self.vertices.append(vertex)  

  def _update_subtree_stats(self, index):
    v_parent = self.vertices[index]
    for child_index in v_parent.child_indices:
      v_child = self.vertices[child_index]
      v_child.type = v_parent.type
      v_child.level = v_parent.level + 1
      if v_child.contain_regrasp != NOREGRASP:
        v_child.regrasp_count = v_parent.regrasp_count + 1
      else:
        v_child.regrasp_count = v_parent.regrasp_count
      self._update_subtree_stats(child_index)

  @staticmethod
  def set_relation(v_child, v_parent):
    v_child.parent_index = v_parent.index
    if v_parent.index in v_child.child_indices:
      v_child.child_indices.remove(v_parent.index)
    if v_child.index not in v_parent.child_indices:
      v_parent.child_indices.append(v_child.index)

  @staticmethod
  def remove_relation(v_child, v_parent):
    if v_child.parent_index == v_parent.index:
      v_child.parent_index = None
    if v_child.index in v_parent.child_indices:
      v_parent.child_indices.remove(v_child.index)

  def output(self):
    for v in self.vertices:
      print '{0}.{1}'.format(v.index,v.regrasp_count),'\t', v.parent_index, '\t', v.child_indices,'\t',['NO', 'START','END'][v.contain_regrasp], '\t', ['FW','BW'][v.type]


class CCTree(object):  
  """
  An RRT tree class for planning closed-chain motion.
  """

  def __init__(self, database, v_root, treetype=FW):
    """
    CCTree constructor.

    @type    v_root: CCVertex
    @param   v_root: Root vertex to grow the tree from.
    @type  treetype: int
    @param treetype: The direction of the tree in the closed-chain motion.
                     It can be either forward(C{FW}) or backward(C{BW}).
    """
    self.treetype = treetype
    self.database = database
    self.database.append(v_root, self.treetype)
    self.end_index = v_root.index

  def __len__(self):
    return len([1 for v in self.database.vertices if v.type == self.treetype])
  
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
    v_new.level            = self.database[parent_index].level + 1

    if v_new.contain_regrasp != NOREGRASP:
      v_new.regrasp_count = self.database[parent_index].regrasp_count + 1
    else:
      v_new.regrasp_count = self.database[parent_index].regrasp_count

    self.database.append(v_new, self.treetype)
    self.database[parent_index].child_indices.append(v_new.index)
    self.end_index = v_new.index

  def generate_rot_traj_list(self):
    """
    Return all C{rot_traj} of vertices 
    connecting the specified vertex and C{v_root}.

    @rtype:  list
    @return: A list containing all C{rot_traj} of all vertices, 
             starting from vertex with a earlier C{timestamp}.
    """
    rot_traj_list = []

    vertex = self.database[self.end_index]
    if self.treetype == FW:
      while vertex.parent_index is not None:
        rot_traj_list.append(vertex.rot_traj)
        vertex = self.database[vertex.parent_index]
      rot_traj_list.reverse()
    else:
      while vertex.parent_index is not None:
        R_beg  = vertex.SE3_config_end.T[0:3, 0:3]
        R_end  = vertex.SE3_config_start.T[0:3, 0:3]
        qd_beg = vertex.SE3_config_end.qd * -1.0
        qd_end = vertex.SE3_config_start.qd * -1.0

        directed_rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                               vertex.rot_traj.duration)
        rot_traj_list.append(directed_rot_traj)
        vertex = self.database[vertex.parent_index]

    return rot_traj_list

  def generate_rot_mat_list(self):
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

    vertex = self.database[self.end_index]
    while vertex.parent_index is not None:
      rot_mat_list.append(vertex.SE3_config_end.T[0:3, 0:3])
      vertex = self.database[vertex.parent_index]
    rot_mat_list.append(vertex.SE3_config_end.T[0:3, 0:3])

    if self.treetype == FW:
      rot_mat_list.reverse()

    return rot_mat_list

  def generate_translation_traj_list(self):
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
      
    vertex = self.database[self.end_index]
    if self.treetype == FW:
      while vertex.parent_index is not None:
        translation_traj_list.append(vertex.translation_traj)
        vertex = self.database[vertex.parent_index]
      translation_traj_list.reverse()
    else:
      while vertex.parent_index is not None:
        directed_translation_traj = reverse_traj(vertex.translation_traj)
        translation_traj_list.append(directed_translation_traj)
        vertex = self.database[vertex.parent_index]

    return translation_traj_list

  def generate_bimanual_trajs(self):
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
      
    vertex = self.database[self.end_index]
    if (self.treetype == FW):
      while (vertex.parent_index is not None):
        if vertex.contain_regrasp == ENDREGRASP:
          bimanual_trajs.append(vertex.bimanual_regrasp_traj)
        bimanual_trajs.append(vertex.bimanual_wpts)
        if vertex.contain_regrasp == STARTREGRASP:
          bimanual_trajs.append(vertex.bimanual_regrasp_traj)
        vertex = self.database[vertex.parent_index]
      bimanual_trajs.reverse()
    else:
      while (vertex.parent_index is not None):
        if vertex.contain_regrasp == ENDREGRASP:
          bimanual_regrasp_traj = vertex.bimanual_regrasp_traj.reverse()
          bimanual_trajs.append(bimanual_regrasp_traj)
        bimanual_trajs.append([vertex.bimanual_wpts[0][::-1],
                               vertex.bimanual_wpts[1][::-1]])
        if vertex.contain_regrasp == STARTREGRASP:
          bimanual_regrasp_traj = vertex.bimanual_regrasp_traj.reverse()
          bimanual_trajs.append(bimanual_regrasp_traj)

        vertex = self.database[vertex.parent_index]

    return bimanual_trajs

  def generate_timestamps_list(self):
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
    
    vertex = self.database[self.end_index]
    while vertex.parent_index is not None:
      timestamps_list.append(vertex.timestamps)
      vertex = self.database[vertex.parent_index]

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
               regrasp_limit=1):
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
    """
    # Initialize v_start and v_goal
    SE3_config_start    = SE3Config.from_matrix(T_obj_start)
    SE3_config_goal     = SE3Config.from_matrix(T_obj_goal)
    self.v_start        = CCVertex(q_robots_end=q_robots_start, 
                                   SE3_config_end=SE3_config_start)
    self.v_goal         = CCVertex(q_robots_end=q_robots_goal, 
                                   SE3_config_end=SE3_config_goal)
    self.q_robots_grasp = q_robots_grasp

    # Initialize RRTs
    self.database               = CCVertexDatabase()
    self.tree_start             = CCTree(self.database, self.v_start, FW)
    self.tree_end               = None # to be initialized when being passed 
                                       # to a planner (after grasping pose 
                                       # check is passed)
    self.nn                     = nn
    self.step_size              = step_size
    self.interpolation_duration = interpolation_duration
    self.discr_timestep         = discr_timestep
    self.discr_check_timestep   = discr_check_timestep
    self.regrasp_limit          = regrasp_limit

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
    self.connecting_dir                   = None
    self.connecting_rot_traj              = None
    self.connecting_translation_traj      = None
    self.connecting_bimanual_wpts         = None
    self.connecting_timestamps            = None
    self.connecting_contain_endregrasp    = False
    self.connecting_q_robots_inter        = None
    self.connecting_bimanual_regrasp_traj = None

    self.rot_traj_list         = None
    self.rot_mat_list          = None
    self.lie_traj              = None
    self.translation_traj_list = None
    self.translation_traj      = None
    self.timestamps            = None
    self.bimanual_trajs        = None

    # Statistics
    self.running_time          = 0.0
    self.regrasp_planning_time = 0.0
    self.iteration_count       = 0
    self.solved                = False
    self.regrasp_T_blacklist   = [[],[]]
    
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
    if self.connecting_dir == BW:
      self.rot_traj_list.append(self.connecting_rot_traj)
    else:
      duration = self.connecting_rot_traj.duration
      R_beg  = self.database[self.tree_start.end_index].SE3_config_end.T[0:3, 0:3]
      R_end  = self.database[self.tree_end.end_index].SE3_config_end.T[0:3, 0:3]
      qd_beg = self.database[self.tree_start.end_index].SE3_config_end.qd * -1.0
      qd_end = self.database[self.tree_end.end_index].SE3_config_end.qd * -1.0
      connecting_rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                               duration)
      self.rot_traj_list.append(connecting_rot_traj)
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
    if self.connecting_dir == BW:
      self.translation_traj_list.append(self.connecting_translation_traj)
    else:
      connecting_translation_traj = reverse_traj(
        self.connecting_translation_traj)
      self.translation_traj_list.append(connecting_translation_traj)
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
    if self.connecting_dir == BW:
      bimanual_trajs.append(self.connecting_bimanual_wpts)
      if self.connecting_contain_endregrasp:
        bimanual_trajs.append(self.connecting_bimanual_regrasp_traj)
    else:
      if self.connecting_contain_endregrasp:
        bimanual_trajs.append(self.connecting_bimanual_regrasp_traj.reverse())
      bimanual_trajs.append([self.connecting_bimanual_wpts[0][::-1],
                             self.connecting_bimanual_wpts[1][::-1]])
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
  
  def __init__(self, manip_obj, robots, plan_regrasp=True, debug=False):
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
    self.obj        = manip_obj
    self.env        = self.obj.GetEnv()
    self.robots     = robots
    self.manips     = []
    self.basemanips = []
    self.taskmanips = []
    for (i, robot) in enumerate(self.robots):
      self.manips.append(robot.GetActiveManipulator())
      self.basemanips.append(orpy.interfaces.BaseManipulation(robot))
      self.taskmanips.append(orpy.interfaces.TaskManipulation(robot))
      robot.SetActiveDOFs(self.manips[i].GetArmIndices())

    self._nrobots = len(self.robots)
    self._ndof = self.manips[0].GetArmDOF()
    self._vmax = self.robots[0].GetDOFVelocityLimits()[:self._ndof]
    self._amax = self.robots[0].GetDOFAccelerationLimits()[:self._ndof]

    self._debug        = debug
    self._plan_regrasp = plan_regrasp
    
    self.bimanual_obj_tracker = BimanualObjectTracker(self.robots, manip_obj, debug=self._debug)
    self.rave_planner = orpy.RaveCreatePlanner(self.env, 'birrt')
    self.rave_smoother = orpy.RaveCreatePlanner(self.env, 'parabolicsmoother')

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
    query = self._query
    # Compute relative transformation from end-effectors to object
    self.bimanual_T_rel = []
    for i in xrange(2):
      self.bimanual_T_rel.append(
        np.dot(np.linalg.inv(query.v_start.SE3_config_end.T),
        utils.compute_endeffector_transform(self.manips[i], 
        query.v_start.q_robots_end[i])))

    # Compute object SE3_config at goal if not specified
    if query.v_goal.SE3_config_end is None:
      T_left_robot_goal = utils.compute_endeffector_transform(self.manips[0],
                            query.v_goal.q_robots_end[0])
      T_obj_goal = np.dot(T_left_robot_goal, 
                          np.linalg.inv(self.bimanual_T_rel[0]))
      query.v_goal.SE3_config_end = SE3Config.from_matrix(T_obj_goal)

    # Check start and goal grasping pose
    bimanual_goal_rel_T = []
    for i in xrange(2):
      bimanual_goal_rel_T.append(
        np.dot(np.linalg.inv(query.v_goal.SE3_config_end.T), 
        utils.compute_endeffector_transform(self.manips[i], 
        query.v_goal.q_robots_end[i])))

    if not np.isclose(self.bimanual_T_rel, 
                      bimanual_goal_rel_T, atol=1e-3).all():
      raise CCPlannerException('Start and goal grasping pose not matching.')

    # Complete tree_end in the query 
    query.tree_end = CCTree(query.database, query.v_goal, BW)

  def loose_gripper(self, query, robot_indices=None):
    """
    Open grippers of C{self.robots} by a small amount from C{q_robots_grasp}
    stored in C{query}. This is necessary to avoid collision between object
    and gripper in planning.

    @type  query: CCQuery
    @param query: Query used to extract the grippers' configuration when 
                  grasping the object, which is taken as a reference for 
                  loosing the gripper.
    """
    if robot_indices is None:
      robot_indices = xrange(self._nrobots)
    for i in robot_indices:
      self.robots[i].SetDOFValues([query.q_robots_grasp[i]*0.7],
                                  [self._ndof])

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
    query = self._query
    if query.solved:
      self._output_info('This query has already been solved.', 'green')
      return True

    t_begin = time()

    self.loose_gripper(query)
    reextend = False
    reextend_type = None
    reextend_info = None
    
    res = self._connect()
    if len(res) == 1:
      status = res[0]
      if status == REACHED:
        if self._plan_regrasp_trajs():
          query.running_time = time() - t_begin
          self._finish()
          return True
      reextend = False
    else:
      robot_index, q_new, q_robots_orig, T_obj_orig = res[1]
      nearest_index = res[2]
      reextend = True
      reextend_type = FROMCONNECT
      reextend_info = (robot_index, q_new, q_robots_orig, 
                       T_obj_orig, nearest_index)

    query.running_time = time() - t_begin

    while (query.running_time < timeout):
      if reextend:
        SE3_config = SE3Config.from_matrix(reextend_info[3])
        if reextend_type == FROMCONNECT:
          query.iteration_count += 1
          self._output_debug('Iteration no. {0} (try extension from '
                             'interuptted connect)'.format(
                              query.iteration_count), 'blue')
        elif reextend_type == FROMEXTEND:
          self._output_debug('Iteration no. {0} (retry extension from '
                             'interuptted extend)'.format(
                              query.iteration_count), 'blue')
      else:
        query.iteration_count += 1
        self._output_debug('Iteration no. {0} (start new extension)'.format(
                            query.iteration_count), 'blue')
        SE3_config = self.sample_SE3_config()

      res = self._extend(SE3_config, reextend=reextend, 
                         reextend_info=reextend_info)
      if len(res) == 1:
        status = res[0]
        if status != TRAPPED:
          self._output_debug('Tree start : {0}; Tree end : {1}'.format(
                             len(query.tree_start), len(query.tree_end)),
                             'green')

          res = self._connect()
          if len(res) == 1:
            status = res[0]
            if status == REACHED:
              if self._plan_regrasp_trajs():
                query.running_time = time() - t_begin
                self._finish()
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
        
      query.running_time = time() - t_begin

    query.running_time = time() - t_begin
    self._finish(success=False, timeout=timeout)
    return False

  def _finish(self, success=True, timeout=None):
    """
    Display planning result and wrap up.
    """
    query = self._query

    if success:
      global_time = query.running_time - query.regrasp_planning_time
      self._output_info('Path found after {0} iterations. Total running time'
                        ': [{1:.2f}]s.'.format(query.iteration_count, 
                        query.running_time), 'green')
      self._output_info('Global planning time: [{0:.2f}]s. Regrasp planning '
                        'time: [{1:.2f}]s.'.format(global_time, 
                        query.regrasp_planning_time), 'green')
      query.solved = True
      query.generate_final_cctraj()
    else:
      self._output_info('Timeout {0}s reached after {1} iterations.'.format(
                        timeout, query.iteration_count), 'red')

    self.reset_config(query)

  def _plan_regrasp_trajs(self):
    self._output_info('Global path found.', 'yellow')
    t_begin = time()
    query = self._query
    regrasp_count = 0

    for tree in (query.tree_start, query.tree_end):
      tree_type = tree.treetype
      vertex = query.database[tree.end_index]
      spine_indices = []
      while (vertex.parent_index) is not None:
        spine_indices.append(vertex.index)
        vertex = query.database[vertex.parent_index]
      spine_indices = spine_indices[::-1]
      for index_i, index in enumerate(spine_indices):
        vertex = query.database[index]
        if vertex.contain_regrasp != NOREGRASP:
          regrasp_count += 1
          if vertex.filled:
            self._output_info('Planning regrasp no.[{0}] for [{1}] tree' 
                              ' skipped'.format(regrasp_count, 
                              ['FW', 'BW'][tree_type]), 'cyan')
            continue
          self._output_info('Planning regrasp no.[{0}] for [{1}] tree...'.format(regrasp_count, ['FW', 'BW'][tree_type]), 'yellow')

          bimanual_regrasp_traj = BimanualRegraspTrajectory()
          if vertex.contain_regrasp == ENDREGRASP:
            # position everything correctly 
            self.obj.SetTransform(vertex.SE3_config_end.T)
            for robot, q_robot_inter in zip(self.robots,
                                            vertex.q_robots_inter):
              robot.SetActiveDOFValues(q_robot_inter)
            # start planning
            for i in xrange(self._nrobots):
              q_robot_inter = vertex.q_robots_inter[i]
              q_robot_end   = vertex.q_robots_end[i]
              if not np.isclose(q_robot_inter, q_robot_end, rtol=1e-3).all():
                robot = self.robots[i]
                robot.SetDOFValues([0], [self._ndof])
                if self._plan_regrasp:
                  params = orpy.Planner.PlannerParameters()
                  params.SetRobotActiveJoints(robot)
                  params.SetGoalConfig(q_robot_end)
                  params.SetExtraParameters(
                    """
                    <_nmaxiterations>300</_nmaxiterations>
                    <_postprocessing></_postprocessing>
                  """)
                  self.rave_planner.InitPlan(robot, params)
                  traj = orpy.RaveCreateTrajectory(self.env, '')

                  if self.rave_planner.PlanPath(traj) == HAS_SOLUTION:
                    # print colorize('{0}'.format(self._is_bad_regrasp_T([i], vertex.SE3_config_end.T)),'green')
                    bimanual_regrasp_traj.trajs[i] = traj
                    robot.SetActiveDOFValues(q_robot_end)
                    self.loose_gripper(query, [i])
                  else:
                    self._output_info('Planning failed, reforming trees...', 
                                      'red')
                    # print colorize('{0}'.format(self._is_bad_regrasp_T([i], vertex.SE3_config_end.T)),'red')
                    query.regrasp_T_blacklist[i].append(vertex.SE3_config_end.T)
                    self._output_info('index: {0}'.format(index), 'red')
                    self.loose_gripper(query)
                    self._reform_trees(tree_type, spine_indices[index_i:])
                    return False
          elif vertex.contain_regrasp == STARTREGRASP:
            # position everything correctly 
            self.obj.SetTransform(vertex.SE3_config_start.T)
            for robot, q_robot_start in zip(self.robots,
                                            vertex.q_robots_start):
              robot.SetActiveDOFValues(q_robot_start)
            # start planning
            for i in xrange(self._nrobots):
              q_robot_start = vertex.q_robots_start[i]
              q_robot_inter = vertex.q_robots_inter[i]
              if not np.isclose(q_robot_start,q_robot_inter,rtol=1e-3).all():
                robot = self.robots[i]
                robot.SetDOFValues([0], [self._ndof])
                if self._plan_regrasp:
                  params = orpy.Planner.PlannerParameters()
                  params.SetRobotActiveJoints(robot)
                  params.SetGoalConfig(q_robot_inter)
                  params.SetExtraParameters(
                    """
                    <_nmaxiterations>300</_nmaxiterations>
                    <_postprocessing></_postprocessing>
                  """)
                  self.rave_planner.InitPlan(robot, params)
                  traj = orpy.RaveCreateTrajectory(self.env, '')
                  if self.rave_planner.PlanPath(traj) == HAS_SOLUTION:
                    # print colorize('{0}'.format(self._is_bad_regrasp_T([i], vertex.SE3_config_start.T)),'green')
                    bimanual_regrasp_traj.trajs[i] = traj
                    robot.SetActiveDOFValues(q_robot_inter)
                    self.loose_gripper(query, [i])
                  else:
                    self._output_info('Planning failed, reforming trees...', 
                                      'red')
                    # print colorize('{0}'.format(self._is_bad_regrasp_T([i], vertex.SE3_config_start.T)),'red')
                    query.regrasp_T_blacklist[i].append(vertex.SE3_config_start.T)
                    self._output_info('index: {0}'.format(index), 'red')
                    self.loose_gripper(query)
                    self._reform_trees(tree_type, spine_indices[index_i:])
                    return False
                    
          vertex._fill_regrasp_traj(bimanual_regrasp_traj)

    # connecting link
    if query.connecting_contain_endregrasp:
      regrasp_count += 1
      self._output_info('Planning regrasp no.[{0}] for [connecting]'\
                        .format(regrasp_count), 'yellow')
      bimanual_regrasp_traj = BimanualRegraspTrajectory()
      if query.connecting_dir == FW:
        v_goal = query.database[query.tree_start.end_index]
      else: # query.connecting_dir == BW
        v_goal = query.database[query.tree_end.end_index]
      # position everything correctly 
      self.obj.SetTransform(v_goal.SE3_config_end.T)
      for robot, q_robot_inter in zip(self.robots, 
                                      query.connecting_q_robots_inter):
        robot.SetActiveDOFValues(q_robot_inter)
      # start planning
      for i in xrange(self._nrobots):
        q_robot_inter = query.connecting_q_robots_inter[i]
        q_robot_end = v_goal.q_robots_end[i]
        if not np.isclose(q_robot_inter, q_robot_end, rtol=1e-3).all():
          robot = self.robots[i]
          robot.SetDOFValues([0], [self._ndof])
          if self._plan_regrasp:
            params = orpy.Planner.PlannerParameters()
            params.SetRobotActiveJoints(robot)
            params.SetGoalConfig(q_robot_end)
            params.SetExtraParameters(
              """
              <_nmaxiterations>300</_nmaxiterations>
              <_postprocessing></_postprocessing>
            """)
            self.rave_planner.InitPlan(robot, params)
            traj = orpy.RaveCreateTrajectory(self.env, '')
            if self.rave_planner.PlanPath(traj) == HAS_SOLUTION:
              bimanual_regrasp_traj.trajs[i] = traj
              robot.SetActiveDOFValues(q_robot_end)
              self.loose_gripper(query, [i])
            else:
              self._output_info('Planning failed', 'red')
              self.loose_gripper(query)
              return False

      query.connecting_bimanual_regrasp_traj = bimanual_regrasp_traj

    query.regrasp_planning_time += time() - t_begin
    return True

  def _reform_trees(self, bad_tree_type, bad_indices):
    query = self._query
    # convert bad vertices and transfer to good tree
    v0 = query.database[bad_indices[0]]
    query.database.remove_relation(v_child=v0,
                                   v_parent=query.database[v0.parent_index])
    for i, index in enumerate(bad_indices[:-1]):
      v = query.database[index]
      v_new_parent = query.database[bad_indices[i+1]]
      query.database.set_relation(v_child=v, v_parent=v_new_parent)
      v.q_robots_start = v_new_parent.q_robots_end
      v.q_robots_end = v_new_parent.q_robots_start
      if v_new_parent.contain_regrasp == STARTREGRASP:
        v.contain_regrasp = ENDREGRASP
        v.q_robots_inter = v_new_parent.q_robots_inter
      elif v_new_parent.contain_regrasp == ENDREGRASP:
        v.contain_regrasp = STARTREGRASP
        v.q_robots_inter = v_new_parent.q_robots_inter
      else:
        v.contain_regrasp = NOREGRASP
        v.q_robots_inter = None
      v.SE3_config_start = v_new_parent.SE3_config_end
      v.SE3_config_start.qd *= -1.0
      v.SE3_config_start.pd *= -1.0
      v.SE3_config_end = v_new_parent.SE3_config_start
      v.SE3_config_end.qd *= -1.0
      v.SE3_config_end.pd *= -1.0
      v.rot_traj = lie.InterpolateSO3(v.SE3_config_start.T[0:3, 0:3], 
                                      v.SE3_config_end.T[0:3, 0:3], 
                                      v.SE3_config_start.qd, 
                                      v.SE3_config_end.qd,
                                      v_new_parent.rot_traj.duration)
      v.translation_traj = reverse_traj(v_new_parent.translation_traj)
      v.bimanual_wpts = [v_new_parent.bimanual_wpts[0][::-1],
                         v_new_parent.bimanual_wpts[1][::-1]]
      v.timestamps = v_new_parent.timestamps

    if bad_tree_type == FW:
      good_tree = query.tree_end
    else: # bad_tree_type == BW
      good_tree = query.tree_start

    # convert last bad vertex and transfer to good tree
    v = query.database[bad_indices[-1]]
    v_new_parent = query.database[good_tree.end_index]
    query.database.set_relation(v_child=v, v_parent=v_new_parent)
    v.q_robots_start = v_new_parent.q_robots_end
    # v.q_robots_end = v.q_robots_end # redundant
    if query.connecting_contain_endregrasp:
      v.q_robots_inter = query.connecting_q_robots_inter
      if query.connecting_dir == bad_tree_type:
        v.contain_regrasp = ENDREGRASP
      else: # query.connecting_dir != bad_tree_type
        v.contain_regrasp = STARTREGRASP
    else:
      v.contain_regrasp = NOREGRASP
      v.q_robots_inter = None
    v.SE3_config_start = v_new_parent.SE3_config_end
    v.SE3_config_start.qd *= -1.0
    v.SE3_config_start.pd *= -1.0
    # v.SE3_config_end = v.SE3_config_end # redundant
    v.SE3_config_end.qd *= -1.0
    v.SE3_config_end.pd *= -1.0
    if query.connecting_dir == bad_tree_type:
      v.rot_traj = query.connecting_rot_traj
      v.translation_traj = query.connecting_translation_traj
      v.bimanual_wpts = query.connecting_bimanual_wpts
    else: # query.connecting_dir != bad_tree_type
      v.rot_traj = lie.InterpolateSO3(v.SE3_config_start.T[0:3, 0:3], 
                                      v.SE3_config_end.T[0:3, 0:3], 
                                      v.SE3_config_start.qd, 
                                      v.SE3_config_end.qd,
                                      query.connecting_rot_traj.duration)
      v.translation_traj = reverse_traj(query.connecting_translation_traj)
      v.bimanual_wpts = [query.connecting_bimanual_wpts[0][::-1],
                         query.connecting_bimanual_wpts[1][::-1]]
    v.timestamps = query.connecting_timestamps

    query.database._update_subtree_stats(good_tree.end_index)

  def reset_config(self, query):
    """
    Reset everything to their starting configuration according to the query,
    including re-closing the grippers which were probably opened for planning.
    This is used after planning is done since in planning process robots and object
    will be moved for collision checking.

    @type  query: CCQuery
    @param query: Query to be used to extract starting configuration.
    """
    for i in xrange(self._nrobots):
      self.robots[i].SetActiveDOFValues(query.v_start.q_robots_end[i])
      self.robots[i].SetDOFValues([query.q_robots_grasp[i]],
                                  [self._ndof])
    self.obj.SetTransform(query.v_start.SE3_config_end.T)

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
    query = self._query
    if (query.iteration_count - 1) % 2 == FW:
      cur_dir = FW
      cur_tree = query.tree_start
    else:
      cur_dir = BW
      cur_tree = query.tree_end

    status = TRAPPED
    nnindices = self._nearest_neighbor_indices(SE3_config, cur_dir)
    if reextend:
      nnindices = (reextend_info[4],)
    for index in nnindices:
      self._output_debug('index:{0}, nnindices:{1}'.format(index, nnindices))
      v_near = query.database[index]
      
      q_beg  = v_near.SE3_config_end.q
      qd_beg = v_near.SE3_config_end.qd
      p_beg  = v_near.SE3_config_end.p
      pd_beg = v_near.SE3_config_end.pd

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
                                      v_near.SE3_config_end.T,
                                      1.0 / np.pi, 1.0)
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
                                    query.interpolation_duration)
      translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

      # Check translational limit
      # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
      res = utils.check_translation_traj_str_limits(query.upper_limits, query.lower_limits, translation_traj_str)
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
              [rot_traj]), translation_traj, v_near.q_robots_end)
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

        if reextend:
          q_robots_real = np.array(new_q_robots) # post regrasp configs
          q_robots_real[reextend_info[0]] = reextend_info[1]
          v_new = CCVertex(q_robots_start=v_near.q_robots_end,
                           q_robots_inter=new_q_robots,
                           q_robots_end=q_robots_real,
                           SE3_config_start=v_near.SE3_config_end,
                           SE3_config_end=new_SE3_config)
          self._output_debug('Adding regrasping', 'yellow')
          v_new._add_regrasp_action()
        else:
          v_new = CCVertex(q_robots_start=v_near.q_robots_end,
                           q_robots_end=new_q_robots,
                           SE3_config_start=v_near.SE3_config_end,
                           SE3_config_end=new_SE3_config)

        cur_tree.add_vertex(v_new, v_near.index, rot_traj,
                            translation_traj, bimanual_wpts, timestamps)
        return status,
      else:
        if v_near.regrasp_count >= query.regrasp_limit \
          or self._is_bad_regrasp_T(robot_indices=[res[1]], 
                                    T_obj_regrasp=res[4]):
          status = TRAPPED
          continue

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
    query = self._query
    if (query.iteration_count - 1) % 2 == FW:
      cur_dir = FW
      cur_extend_dir = BW
      cur_tree_extend = query.tree_end
      cur_tree_goal = query.tree_start
    else:
      cur_dir = BW
      cur_extend_dir = FW
      cur_tree_extend = query.tree_start
      cur_tree_goal = query.tree_end

    v_test = query.database[cur_tree_goal.end_index]
    allowed_regrasp_count = query.regrasp_limit - v_test.regrasp_count
    nnindices = self._nearest_neighbor_indices(v_test.SE3_config_end,
                                               cur_extend_dir,
                                               allowed_regrasp_count)
    status = TRAPPED
    for index in nnindices:
      v_near = query.database[index]
      remain_regrasp_count = allowed_regrasp_count - v_near.regrasp_count

      while True:
        q_beg  = v_near.SE3_config_end.q
        qd_beg = v_near.SE3_config_end.qd
        p_beg  = v_near.SE3_config_end.p
        pd_beg = v_near.SE3_config_end.pd
        
        q_end  = v_test.SE3_config_end.q
        qd_end = v_test.SE3_config_end.qd
        p_end  = v_test.SE3_config_end.p
        pd_end = v_test.SE3_config_end.pd
        
        # Check distance
        SE3_dist = utils.SE3_distance(v_test.SE3_config_end.T, 
                                      v_near.SE3_config_end.T, 
                                      1.0 / np.pi, 1.0)
        if SE3_dist <= query.step_size: # connect directly
          # Interpolate the object trajectory
          R_beg = orpy.rotationMatrixFromQuat(q_beg)
          R_end = orpy.rotationMatrixFromQuat(q_end)
          rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end, 
                                        query.interpolation_duration)
          translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end,
            pd_beg, pd_end, query.interpolation_duration)

          # Check translational limit
          # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
          if not utils.check_translation_traj_str_limits(
                  query.upper_limits, query.lower_limits, 
                  translation_traj_str):
            self._output_debug('TRAPPED : SE(3) trajectory exceeds '
                               'translational limit', bold=False)
            break

          translation_traj = TrajectoryFromStr(translation_traj_str)

          # Check collision (object trajectory)
          if not self.is_collision_free_SE3_traj(
                  rot_traj, translation_traj, R_beg):
            self._output_debug('TRAPPED : SE(3) trajectory in collision', 
                               bold=False)
            break
          
          # Check reachability (object trajectory)
          res = self.check_SE3_traj_reachability(lie.LieTraj([R_beg, R_end], 
                  [rot_traj]), translation_traj, v_near.q_robots_end)
          if res[0] is False:
            passed, bimanual_wpts, timestamps = res[1:]
            if not passed:
              self._output_debug('TRAPPED : SE(3) trajectory not reachable', 
                                 bold=False)
              break

            # Check similarity of terminal IK solutions
            eps = 1e-3
            need_regrasp = False
            query.connecting_contain_endregrasp = False
            if v_test.contain_regrasp == NOREGRASP:
              for i in xrange(2):
                if not utils.distance(v_test.q_robots_end[i], 
                                      bimanual_wpts[i][-1]) < eps:
                  self._output_debug('IK discrepancy (robot {0})'.format(i), 
                                      bold=False)
                  need_regrasp = True
              if need_regrasp:
                if self._is_bad_regrasp_T([0,1], v_test.SE3_config_end.T)\
                  or remain_regrasp_count == 0:
                  break

                self._output_debug('Adding regrasping to v_test', 'yellow')
                q_robots_end = np.array([bimanual_wpts[0][-1],
                                         bimanual_wpts[1][-1]])
                v_test.q_robots_inter = v_test.q_robots_end
                v_test.q_robots_end = q_robots_end
                v_test._add_regrasp_action()
            elif v_test.contain_regrasp == ENDREGRASP:
              for i in xrange(2):
                if not utils.distance(v_test.q_robots_inter[i], 
                                      bimanual_wpts[i][-1]) < eps:
                  self._output_debug('IK discrepancy (robot {0})'.format(i), 
                                      bold=False)
                  need_regrasp = True
              if need_regrasp:
                if self._is_bad_regrasp_T([0,1], v_test.SE3_config_end.T):
                  break
                self._output_debug('Replacing regrasping in v_test', 'yellow')
                v_test.q_robots_end = np.array([bimanual_wpts[0][-1],
                                                bimanual_wpts[1][-1]])
            elif v_test.contain_regrasp == STARTREGRASP:
              for i in xrange(2):
                if not utils.distance(v_test.q_robots_end[i],
                                      bimanual_wpts[i][-1]) < eps:
                  self._output_debug('IK discrepancy (robot {0})'.format(i), 
                                      bold=False)
                  need_regrasp = True
              if need_regrasp:
                if self._is_bad_regrasp_T([0,1], v_test.SE3_config_end.T)\
                  or remain_regrasp_count == 0:
                  break
                self._output_debug('Adding regrasping to connect', 'yellow')
                query.connecting_contain_endregrasp = True
                query.connecting_q_robots_inter = np.array(
                  [bimanual_wpts[0][-1], bimanual_wpts[1][-1]])

            # Now the connection is successful
            cur_tree_extend.end_index         = v_near.index
            query.connecting_rot_traj         = rot_traj
            query.connecting_translation_traj = translation_traj
            query.connecting_bimanual_wpts    = bimanual_wpts
            query.connecting_timestamps       = timestamps
            query.connecting_dir              = cur_dir
            status = REACHED
            return status,
          else: # need regrasp
            if v_near.regrasp_count >= query.regrasp_limit \
              or remain_regrasp_count == 0 \
              or self._is_bad_regrasp_T(robot_indices=[res[1]], 
                                        T_obj_regrasp=res[4]):
              status = TRAPPED
              break

            self._output_debug('TRAPPED : SE(3) trajectory need regrasping', 
                               bold=False)
            status = NEEDREGRASP
            return status, res[1:], v_near.index

        else: # extend towards target by step_size
          if not utils._is_close_axis(q_beg, q_end):
            q_end = -q_end
          q_end = q_beg + query.step_size * (q_end - q_beg) / SE3_dist
          q_end /= np.sqrt(np.dot(q_end, q_end))
          p_end = p_beg + query.step_size * (p_end - p_beg) / SE3_dist
          new_SE3_config = SE3Config(q_end, p_end, qd_end, pd_end)

          # Check collision (SE3_config)
          res = self.is_collision_free_SE3_config(new_SE3_config)
          if not res:
            self._output_debug('TRAPPED : SE(3) config in collision',
                               bold=False)
            status = TRAPPED
            break

          # Check reachability (SE3_config)
          res = self.check_SE3_config_reachability(new_SE3_config)
          if not res:
            self._output_debug('TRAPPED : SE(3) config not reachable',
                               bold=False)
            status = TRAPPED
            break

          # Interpolate a SE3 trajectory for the object
          R_beg = orpy.rotationMatrixFromQuat(q_beg)
          R_end = orpy.rotationMatrixFromQuat(q_end)
          rot_traj = lie.InterpolateSO3(R_beg, R_end, qd_beg, qd_end,
                                        query.interpolation_duration)
          translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_beg, pd_end, query.interpolation_duration)

          # Check translational limit
          # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
          res = utils.check_translation_traj_str_limits(query.upper_limits, query.lower_limits, translation_traj_str)
          if not res:
            self._output_debug('TRAPPED : SE(3) trajectory exceeds translational limit', bold=False)
            status = TRAPPED
            break

          translation_traj = TrajectoryFromStr(translation_traj_str)

          # Check collision (object trajectory)
          res = self.is_collision_free_SE3_traj(rot_traj, translation_traj, 
                                                R_beg)
          if not res:
            self._output_debug('TRAPPED : SE(3) trajectory in collision', 
                               bold=False)
            status = TRAPPED
            break

          # Check reachability (object trajectory)
          res = self.check_SE3_traj_reachability(lie.LieTraj([R_beg, R_end], [rot_traj]), translation_traj, v_near.q_robots_end)
          if res[0] is False: # no need regrasp
            passed, bimanual_wpts, timestamps = res[1:]
            if not passed:
              self._output_debug('TRAPPED : SE(3) trajectory not reachable', 
                                 bold=False)
              status = TRAPPED
              break

            # Now this trajectory is alright.
            self._output_debug('Advanced : new vertex generated', 
                               color='green', bold=False)
            new_q_robots = [wpts[-1] for wpts in bimanual_wpts] 
            v_new = CCVertex(q_robots_start=v_near.q_robots_end,
                             q_robots_end=new_q_robots,
                             SE3_config_start=v_near.SE3_config_end,
                             SE3_config_end=new_SE3_config)
            cur_tree_extend.add_vertex(v_new, v_near.index, rot_traj,
                                       translation_traj, bimanual_wpts,
                                       timestamps)
            v_near = v_new
          else: # need regrasp
            if v_near.regrasp_count >= query.regrasp_limit \
              or remain_regrasp_count == 0 \
              or self._is_bad_regrasp_T(robot_indices=[res[1]], 
                                        T_obj_regrasp=res[4]):
              status = TRAPPED
              break

            self._output_debug('TRAPPED : SE(3) trajectory need regrasping', 
                               bold=False)
            status = NEEDREGRASP
            return status, res[1:], v_near.index
    return status,        

  def _is_bad_regrasp_T(self, robot_indices, T_obj_regrasp):
    for robot_index in robot_indices:
      for T in self._query.regrasp_T_blacklist[robot_index]:
        if utils.SE3_distance(T, T_obj_regrasp, 
                              1.0 / np.pi, 1.0) < self._query.step_size/2.0:
          self._output_info('Attempt to add regrasp but at bad T.',
                             bold=False)
          return True
    return False

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
                                  ref_sols):
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
            self._query.discr_check_timestep)

  def _nearest_neighbor_indices(self, SE3_config, treetype,
                                regrasp_count_limit=None):
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
    database = self._query.database

    if regrasp_count_limit is None:
      cur_tree_vertices = np.array([v for v in database.vertices 
                                    if v.type == treetype])
    else:
      cur_tree_vertices = np.array([v for v in database.vertices 
                                    if (v.type == treetype and 
                                    v.regrasp_count <= regrasp_count_limit)])
    distance_list = [utils.SE3_distance(SE3_config.T, v.SE3_config_end.T, 
                      1.0 / np.pi, 1.0) for v in cur_tree_vertices]
    try:
      distance_heap = heap.Heap(distance_list)
    except:
      print 'hahaha finally'
      embed()
        
    if (self._query.nn == -1):
      # to consider all vertices in the tree as nearest neighbors
      nn = len(cur_tree_vertices)
    else:
      nn = min(self._query.nn, len(cur_tree_vertices))
    cur_nnindices = np.argsort(distance_list)[:nn]
    nnindices = [cur_tree_vertices[i].index for i in cur_nnindices]
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

  def visualize_regrasp_traj(self, bimanual_regrasp_traj, speed=1.0):
    sampling_step = 0.01
    refresh_step  = sampling_step / speed

    if bimanual_regrasp_traj.order == LR:
      keys = (0, 1)
    else:
      keys = (1, 0)

    for key in keys:
      traj = bimanual_regrasp_traj.trajs[key]
      if traj is not None:
        robot = self.robots[key]
        manip = self.manips[key]
        taskmanip = self.taskmanips[key]
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
      if type(bimanual_traj) is not BimanualRegraspTrajectory:
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
        timestamp_index -= 1
        self.visualize_regrasp_traj(bimanual_traj, speed=speed)

  def shortcut(self, query, maxiters=[20, 20]):
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
    bimanual_trajs   = query.cctraj.bimanual_trajs

    wpt_traj_id = []
    for i, bimanual_traj in enumerate(bimanual_trajs):
      if type(bimanual_traj) is not BimanualRegraspTrajectory:
        wpt_traj_id.append(i)
    wpt_traj_count = len(wpt_traj_id)

    self.loose_gripper(query)
    t_begin = time()
    for _ in xrange(maxiters[0]):  
      self._output_debug('Iteration {0}'.format(i + 1), color='blue', bold=False)
      total_length = len(timestamps)
      wpt_traj_timestamps = {}
      wpt_traj_weight = {}
      prev_end_time_index = 0
      for i in wpt_traj_id:
        length = len(bimanual_trajs[i][0])
        wpt_traj_weight[i] = length / (total_length + wpt_traj_count - 1.0)
        wpt_traj_timestamps[i] = list(np.array(timestamps)[range(
                                  prev_end_time_index, 
                                  prev_end_time_index + length)])
        prev_end_time_index += length - 1
      
      # perform shorcut on selected trajectory(wpts) section
      cur_id = np.random.choice(wpt_traj_weight.keys(), 
                                p=wpt_traj_weight.values())
      cur_timestamps = wpt_traj_timestamps[cur_id][:]
      cur_timestamps_indices = range(len(cur_timestamps))
      cur_left_wpts  = bimanual_trajs[cur_id][0][:]
      cur_right_wpts = bimanual_trajs[cur_id][1][:]

      # Sample two time instants
      t0_index = _RNG.choice(cur_timestamps_indices[:-min_n_timesteps])
      t1_index = _RNG.choice(cur_timestamps_indices[t0_index + min_n_timesteps:])
      t0 = cur_timestamps[t0_index]
      t1 = cur_timestamps[t1_index]
      t0_global_index = timestamps.index(t0)
      t1_global_index = timestamps.index(t1)
      # Interpolate a new SE(3) trajectory segment
      new_R0 = lie_traj.EvalRotation(t0)
      new_R1 = lie_traj.EvalRotation(t1)
      new_rot_traj = lie.InterpolateSO3(new_R0, new_R1, 
                                        lie_traj.EvalOmega(t0), 
                                        lie_traj.EvalOmega(t1), 
                                        query.interpolation_duration)
      new_lie_traj = lie.LieTraj([new_R0, new_R1], [new_rot_traj])      
      neww_rot_traj = lie.InterpolateSO3(new_R1, new_R0, 
                                        -lie_traj.EvalOmega(t1), 
                                        -lie_traj.EvalOmega(t0), 
                                        query.interpolation_duration)
      neww_lie_traj = lie.LieTraj([new_R1, new_R0], [neww_rot_traj])

      new_translation_traj_str = utils.traj_str_3rd_degree(translation_traj.Eval(t0), translation_traj.Eval(t1), translation_traj.Evald(t0), translation_traj.Evald(t1), query.interpolation_duration)
      new_translation_traj = TrajectoryFromStr(new_translation_traj_str)  
      neww_translation_traj_str = utils.traj_str_3rd_degree(translation_traj.Eval(t1), translation_traj.Eval(t0), -translation_traj.Evald(t1), 
        -translation_traj.Evald(t0), query.interpolation_duration)
      neww_translation_traj = TrajectoryFromStr(neww_translation_traj_str) 
      
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
      res = self.check_SE3_traj_reachability(
              lie.LieTraj([new_R0, new_R1], 
            [new_rot_traj]), new_translation_traj, 
            [cur_left_wpts[t0_index], cur_right_wpts[t0_index]])
      if res[0]: # ignore request for regrasp in shortcut
        continue
      passed, bimanual_wpts, new_timestamps = res[1:]

      if not passed:
        not_reachable_count += 1
        self._output_debug('Not reachable', color='yellow', bold=False)
        continue

      # Check continuity between newly generated bimanual_wpts and original one
      eps = 5e-2 # Might be too big!!!
      if not (utils.distance(bimanual_wpts[0][-1], cur_left_wpts[t1_index]) < eps and utils.distance(bimanual_wpts[1][-1], cur_right_wpts[t1_index]) < eps):
        not_continuous_count += 1
        self._output_debug('Not continuous', color='yellow', bold=False)
        continue

      # Now the new trajectory passes all tests
      # Replace all the old trajectory segments with the new ones
      lie_traj = utils.replace_lie_traj_segment(lie_traj, 
                                                new_lie_traj.trajlist[0],
                                                t0, t1)            
      translation_traj = utils.replace_traj_segment(translation_traj,
                                                    new_translation_traj, 
                                                    t0, t1)
      
      first_timestamp_chunk       = timestamps[:t0_global_index + 1]
      last_timestamp_chunk_offset = timestamps[t1_global_index]
      last_timestamp_chunk        = [t - last_timestamp_chunk_offset for t in timestamps[t1_global_index:]]

      timestamps = utils.merge_timestamps_list([first_timestamp_chunk, new_timestamps, last_timestamp_chunk])
      bimanual_trajs[cur_id][0] = utils.merge_wpts_list(
        [cur_left_wpts[:t0_index + 1], bimanual_wpts[0], 
        cur_left_wpts[t1_index:]], eps=eps)            
      bimanual_trajs[cur_id][1] = utils.merge_wpts_list(
        [cur_right_wpts[:t0_index + 1], bimanual_wpts[1], 
        cur_right_wpts[t1_index:]], eps=eps)
      
      self._output_debug('Shortcutting successful.', color='green', 
                          bold=False)
      successful_count += 1

    t_end = time()
    self._output_info('Shortcutting for closed-chain motion done. Running time : {0} s.'. format(t_end - t_begin), 'green')
    self._output_debug('Successful: {0} times. In collision: {1} times. Not shorter: {2} times. Not reachable: {3} times. Not continuous: {4} times.'.format(successful_count, in_collision_count, not_shorter_count, not_reachable_count, not_continuous_count), 'yellow')

    query.cctraj = CCTrajectory(lie_traj, translation_traj, bimanual_trajs, 
                                timestamps)

    # smooth regrasp trajs
    t_begin = time()
    self.loose_gripper(query)
    for i, bimanual_traj in enumerate(bimanual_trajs):
      if type(bimanual_traj) is BimanualRegraspTrajectory:
        bimanual_wpts_0 = bimanual_trajs[i-1]
        bimanual_wpts_1 = bimanual_trajs[i+1]
        t = wpt_traj_timestamps[i-1][-1]
        # position everything
        T = np.eye(4)
        T[:3, :3] = lie_traj.EvalRotation(t)
        T[:3, 3] = translation_traj.Eval(t)
        self.obj.SetTransform(T)
        self.robots[0].SetActiveDOFValues(bimanual_wpts_0[0][-1])
        self.robots[1].SetActiveDOFValues(bimanual_wpts_0[1][-1])     
        if bimanual_traj.order == LR: 
          keys = (0, 1)
        else:
          keys = (1, 0)
        for index in keys:
          traj = bimanual_traj.trajs[index]
          robot = self.robots[index]
          if traj is not None:
            robot.SetDOFValues([0], [6])
            params = orpy.Planner.PlannerParameters()
            params.SetRobotActiveJoints(robot)
            params.SetExtraParameters(
              "<_nmaxiterations>{0}</_nmaxiterations>"
              "<_postprocessing></_postprocessing>".format(maxiters[1])
            )
            self.rave_smoother.InitPlan(robot, params)
            self.rave_smoother.PlanPath(traj)
            self.loose_gripper(query, [index])
            robot.SetActiveDOFValues(bimanual_wpts_1[index][0])
    t_end = time()
    self._output_info('Shortcutting for regrasp motion done. '
                      'Running time : {0} s.'. format(t_end - t_begin),
                      'green')
    
    self.reset_config(query)
    
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
    self._tol     = 1e-7

  def update_vmax(self):
    """
    Update attribute C{_vmax} in case velocity limit of the robot changed.
    """
    self._vmax = self.robots[0].GetDOFVelocityLimits()[0:self._ndof]

  def plan(self, lie_traj, translation_traj, bimanual_T_rel, q_robots_init, discr_timestep, discr_check_timestep):
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
    q_robots_prev = np.array(q_robots_init)
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
    return np.dot(error, error)
    
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