from . import utils
import numpy as np
import pickle
import copy
from TOPP import Trajectory
epsilon = 1e-8

class SE3Trajectory(object):
    """
    """
    
    def __init__(self, lie_traj, translation_traj, duration=None):
      if lie_traj is not None:
        assert(abs(lie_traj.duration - translation_traj.duration) <= epsilon)
      self.lie_traj = lie_traj
      self.translation_traj = translation_traj
      self.duration = duration if duration is not None else self.lie_traj.duration
      self._M = np.eye(4)
      self._reversed = False

    def init(self, lie_traj, translation_traj, duration=None):
      """Similar to __init__
      """
      if lie_traj is not None:
        assert(abs(lie_traj.duration - translation_traj.duration) <= epsilon)
      self.lie_traj = lie_traj
      self.translation_traj = translation_traj
      self.duration = duration if duration is not None else self.lie_traj.duration
      self._M = np.eye(4)
        
    @staticmethod
    def init_with_rotation_traj_list(rot_traj_list, rot_mat_list, translation_traj):
      lie_traj = lie.LieTraj(rot_mat_list, rot_traj_list)
      return SE3Trajectory(lie_traj, translation_traj)

    def duplicate(self):
      """Copy self and return an exact SE3Trajectory.
      """
      new_lie_traj = self.lie_traj.Duplicate()
      new_trans_traj = copy.deepcopy(self.translation_traj)
      new_se3_traj = SE3Trajectory(new_lie_traj, new_trans_traj, self.duration)
      return new_se3_traj

    def Eval(self, t):
      """E in Eval is capitalized to comply with other trajectory methods.
      """
      if self._reversed:
        t = self.duration - t
      self._M[0:3, 0:3] = self.lie_traj.EvalRotation(t)
      self._M[0:3, 3] = self.translation_traj.Eval(t)
      return self._M

    def reverse(self):
      """Perform change of variable: s --> duration - s.
      """
      self.lie_traj.Reverse()
      self.translation_traj = Trajectory.ReverseTrajectory(self.translation_traj)

    def trim_back(self, t):
      assert(0 <= t <= self.duration)
      self.lie_traj.trim_back(t)
      trans_traj = Trajectory.SubTraj(self.translation_traj, 0, t)
      self.translation_traj = trans_traj
      self.duration = t

    def trim_front(self, t):
      assert(0 <= t <= self.duration)
      self.lie_traj.trim_front(t)
      trans_traj = Trajectory.SubTraj(self.translation_traj, t)
      self.translation_traj = trans_traj
      self.duration -= t

    def cut(self, t):
      """Cut the trajectory into two halves. The left half is kept in self.
      The right half is returned.
      """
      if abs(t) <= epsilon:
        right_se3_traj = SE3Trajectory(self.lie_traj, self.translation_traj)
        self.init(None, None, 0.0)
        return right_se3_traj
      elif abs(self.duration - t) <= epsilon:
        return SE3Trajectory(None, None, 0.0)

      left_trans_traj = Trajectory.SubTraj(self.translation_traj, 0, t)
      right_lie_traj = self.lie_traj.Cut(t)
      right_trans_traj = Trajectory.SubTraj(self.translation_traj, t)

      self.init(self.lie_traj, left_trans_traj)
      right_se3_traj = SE3Trajectory(right_lie_traj, right_trans_traj)
      return right_se3_traj

class CCTrajectory(object):
  """
  Class of closed-chain trajectory, storing all information needed for 
  a trajectory in closed-chain motions.
  """
  def __init__(self, se3_traj, bimanual_wpts, timestamps, timestep=None):
    self.se3_traj      = se3_traj
    # Note: bimanual_wpts = [left_wpts, right_wpts]
    self.bimanual_wpts = bimanual_wpts
    self.timestamps    = timestamps[:]
    self.timestep      = timestep if timestep is not None else timestamps[1] - timestamps[0]
    self.duration      = self.timestamps[-1]

  @staticmethod
  def init_with_lie_trans_trajs(lie_traj, translation_traj, bimanual_wpts, timestamps,
                                timestep=None):
    """
    CCTrajectory constructor.

    @type  lie_traj: lie.LieTraj
    @param lie_traj: Trajectory of the manipulated object in SO(3) space.
    @type translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
    @param translation_traj: Translational trajectory of the manipulated object.
    @type  bimanual_wpts: list
    @param bimanual_wpts: Trajectory of bimanual robots in form of waypoints list.
    @type  timestamps: list
    @param timestamps: Timestamps for time parameterization of C{bimanual_wpts}.
    @type  timestep: float
    @param timestep: Time resolution of bimanual_wpts.
    """
    se3_traj = SE3Trajectory(lie_traj, translation_traj)
    return CCTrajectory(se3_traj, bimanual_wpts, timestamps, timestep)

  # @staticmethod
  def reverse(self):
    """
    Reverse the given CCTrajectory. Produce and return a new trajectory.
    """
    se3_traj = self.se3_traj.duplicate()
    se3_traj.reverse()
    bimanual_wpts = [self.bimanual_wpts[0][::-1], self.bimanual_wpts[1][::-1]]
    timestamps = self.timestamps[-1] - np.array(self.timestamps)
    timestamps = timestamps.tolist()[::-1]
    timestep = self.timestep
    return CCTrajectory(se3_traj, bimanual_wpts, timestamps, timestep)

  @staticmethod
  def serialize(traj):
    """
    Serialize CCTrajectory object into a string using cPickle.
    """
    return pickle.dumps(traj)

  @staticmethod
  def deserialize(traj_str):
    """
    Generate a CCTrajectory object from a serialized string.
    """
    return pickle.loads(traj_str)

  def cut_at_time_instant(self, index):
    """
    Cut the closed-chain trajectory into two halves. The time instant at which the
    trajectory is cut has to be one of the time stamps.
    """
    t = self.timestamps[index]
    left_lie_traj, right_lie_traj = self.lie_traj.Cut(t)
    left_trans_traj = Trajectory.SubTraj(self.translation_traj, 0, t)
    right_trans_traj = Trajectory.SubTraj(self.translation_traj, t)
    left_cc_traj = CCTrajectory(left_lie_traj, left_trans_traj,
                                [copy.deepcopy(self.bimanual_wtps[0][0:i + 1]),
                                 copy.deepcopy(self.bimanual_wtps[0][0:i + 1])],
                                self.timestamps[0: i + 1], self.timestep)
    right_cc_traj = CCTrajectory(right_lie_traj, right_trans_traj,
                                 [copy.deepcopy(self.bimanual_wtps[0][i:]),
                                  copy.deepcopy(self.bimanual_wtps[0][i:])],
                                 self.timestamps[i:], self.timestep)
    return left_cc_traj, right_cc_traj
