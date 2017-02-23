from . import utils
import pickle
from TOPP import Trajectory
epsilon = 1e-8

class CCTrajectory(object):
  """
  Class of closed-chain trajectory, storing all information needed for 
  a trajectory in closed-chain motions.
  """

  def __init__(self, lie_traj, translation_traj, bimanual_wpts, timestamps,
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
    self.lie_traj         = lie_traj
    self.translation_traj = translation_traj
    self.bimanual_wpts    = bimanual_wpts
    self.timestamps       = timestamps[:]
    self.timestep         = timestep if timestep is not None else timestamps[1] - timestamps[0]

  @staticmethod
  def reverse(traj):
    """
    Reverse the given CCTrajectory.
    """
    lie_traj = deepcopy(traj.lie_traj)
    lie_traj.reverse()
    translation_traj = utils.reverse_traj(traj.translation_traj)
    bimanual_wpts = [traj.bimanual_wpts[0][::-1], traj.bimanual_wpts[1][::-1]]
    timestamps = traj.timestamps[-1] - np.array(traj.timestamps)
    timestamps = timestamps.tolist()[::-1]
    timestep = traj.timestep
    return CCTrajectory(lie_traj, translation_traj, bimanual_wpts, 
                        timestamps, timestep)

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
                                self.bimanual_wpts[0:i + 1], self.timestamps[0: i + 1], self.timestep)
    right_cc_traj = CCTrajectory(right_lie_traj, right_trans_traj,
                                 self.bimanual_wpts[i:], self.timestamps[i:], self.timestep)
    return left_cc_traj, right_cc_traj
