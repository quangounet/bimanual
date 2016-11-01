"""
Utility functions used for motion planners in ikea_planner package.
"""

import openravepy as orpy
from pylab import *

from TOPP import Trajectory
from TOPP import Utilities
from time import sleep

import string
import numpy as np
import lie

INF = np.inf
EPS = 1e-12

############################ Trajectory related ############################
def poly_critical_points(p, interval=None):
  """
  Return the critical points of the given polynomial.

  @type         p: numpy.poly1d
  @param        p: Polynomial to be analyzed.
  @type  interval: list
  @param interval: Interested time interval of the polynomial.

  @rtype: list
  @return: A list of critical points of the polynomial.
  """
  pd = np.polyder(p)
  critical_points = pd.r
  point_list = []
  if interval is None:
    interval = [-INF, INF]
  for x in critical_points:
    if (abs(x.imag) < EPS):
      if (x.real <= interval[1]) and (x.real >= interval[0]):
        point_list.append(x.real)
  if not np.isinf(interval[0]):
    point_list.append(interval[0])
  if not np.isinf(interval[1]):
    point_list.append(interval[1])
  return point_list

def traj_str_3rd_degree(q_beg, q_end, qd_beg, qd_end, duration):
  """
  Return an interpolated 3rd degree polynomial trajectory string.
  It is up to 10 decimal places accuracy to guarantee continuity 
  at trajectory end.

  @type     q_beg: list
  @param    q_beg: Initial configuration.
  @type     q_end: list
  @param    q_end: Final configuration.
  @type    qd_beg: list
  @param   qd_beg: Time derivative of initial configuration.
  @type    qd_end: list
  @param   qd_end: Time derivative of final configuration.
  @type  duration: float
  @param duration: Time length of the interpolated trajectory.

  @rtype:  str
  @return: The interpolated polynomial trajectory string.
  """
  traj_str = ''
  ndof = len(q_beg)
  traj_str += "%f\n%d"%(duration, ndof)
  for k in range(ndof):
    a, b, c, d = Utilities.Interpolate3rdDegree(
                   q_beg[k], q_end[k], qd_beg[k], qd_end[k], duration)
    traj_str += "\n%.10f %.10f %.10f %.10f"%(d, c, b, a)
  return traj_str

def traj_str_5th_degree(q_beg, q_end, qd_beg, qd_end,
                        qdd_beg, qdd_end, duration):
  """
  Return an interpolated 5th degree polynomial trajectory string.
  It is up to 10 decimal places accuracy to guarantee continuity 
  at trajectory end.

  @type     q_beg: list
  @param    q_beg: Initial configuration.
  @type     q_end: list
  @param    q_end: Final configuration.
  @type    qd_beg: list
  @param   qd_beg: Time derivative of initial configuration.
  @type    qd_end: list
  @param   qd_end: Time derivative of final configuration.
  @type   qdd_beg: list
  @param  qdd_beg: 2nd order time derivative of initial configuration.
  @type   qdd_end: list
  @param  qdd_end: 2nd order time derivative of final configuration.
  @type  duration: float
  @param duration: Time length of the interpolated trajectory.

  @rtype:  str
  @return: The interpolated polynomial trajectory string.
  """
  traj_str = ''
  ndof = len(q_beg)
  traj_str += "%f\n%d"%(duration, ndof)
  for k in range(ndof):
    a, b, c, d, e, f = Utilities.Interpolate5thDegree(
                         q_beg[k], q_end[k], qd_beg[k], 
                         qd_end[k], qdd_beg[k], qdd_end[k], duration)
    traj_str += "\n%.10f %.10f %.10f %.10f %.10f %.10f"%(f, e, d, c, b, a)
  return traj_str

def check_config_DOF_limits(robot, q):
  """
  Check whether the configuration is within the robot's DOF limits. 

  @type  robot: openravepy.Robot
  @param robot: Robot to be checked.
  @type      q: list
  @param     q: Configuration of the robot to be checked

  @rtype:  bool
  @return: B{True} if the configuration is within limits.
  """
  lower_limits, upper_limits = robot.GetDOFLimits()

  for (i, val) in enumerate(q):
    if (val < lower_limits[i]) or (val > upper_limits[i]):
      return False
  return True

def check_traj_str_DOF_limits(robot, traj_str):
  """
  Check whether the trajectory is within the robot's DOF limits. 

  @type     robot: openravepy.Robot
  @param    robot: Robot to be checked.
  @type  traj_str: str
  @param traj_str: String representing the trajectory to be checked.

  @rtype:  bool
  @return: B{True} if the trajectory is within limits.
  """
  traj_info = string.split(traj_str, "\n")
  dur = float(traj_info[0])
  ndof = int(traj_info[1])
  lower_limits, upper_limits = robot.GetDOFLimits()
  velocity_limits = robot.GetDOFVelocityLimits()

  for i in range(ndof):
    coeff_list = [float(j) for j in string.split(traj_info[i + 2])]
    coeff_list.reverse()
    q = np.poly1d(coeff_list)
    qd = np.polyder(q)
    q_cri_points = poly_critical_points(q, [0.0, dur])
    qd_cri_points = poly_critical_points(qd, [0.0, dur])
  
    # check DOF values
    for x in q_cri_points:
      if (q(x) < lower_limits[i]) or (q(x) > upper_limits[i]):
        return False
    
    # check DOF velocities
    for x in qd_cri_points:
      if abs(qd(x)) > velocity_limits[i]:
        return False

  return True

def check_translation_traj_str_limits(upper_limits, lower_limits, traj_str):
  """
  Check whether the trajectory is within the given limits. 

  @type  upper_limits: list
  @param upper_limits: Upper limits.
  @type  lower_limits: list
  @param lower_limits: Lower limits.
  @type      traj_str: str
  @param     traj_str: String representing the trajectory to be checked.

  @rtype:  bool
  @return: B{True} if the trajectory is within limits.
  """
  traj_info = string.split(traj_str, "\n")
  dur = float(traj_info[0])
  ndof = int(traj_info[1])

  for i in range(ndof):
    coeff_list = [float(j) for j in string.split(traj_info[i + 2])]
    coeff_list.reverse()
    p = np.poly1d(coeff_list)
    p_cri_points = poly_critical_points(p, [0.0, dur])

    # check limits
    for x in p_cri_points:
      if not (lower_limits[i] <= p(x) <= upper_limits[i]):
        return False
   
  return True
 
def traj_str_from_traj_list(traj_list):
  """
  Return a single trajectory string by combining all trajectories in 
  the given list.

  @type  traj_list: list of TOPP.Trajectory.PiecewisePolynomialTrajectory
  @param traj_list: A list containing all trajectories.

  @rtype:  str
  @return: Trajectory string generated.
  """
  traj_str_list = [str(traj) for traj in traj_list]      
  traj_str = ""
  for i in range(len(traj_str_list)):
    traj_str += "\n"
    traj_str += str(traj_str_list[i])
  traj_str = string.lstrip(traj_str) # remove leading "\n"
  return traj_str

def replace_traj_segment(original_traj, traj_segment, t0, t1):
  """
  Replace the segment in time interval (C{t0}, C{t1}) in C{original_traj} 
  with the given C{traj_segment}.
  NB: The trajectory is of type TOPP.Trajectory.PiecewisePolynomialTrajectory.

  @type  original_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
  @param original_traj: Original trajectory.
  @type   traj_segment: TOPP.Trajectory.PiecewisePolynomialTrajectory
  @param  traj_segment: New trajectory segment.
  @type             t0: float
  @param            t0: Start time of the time interval.
  @type             t1: float
  @param            t1: End time of the time interval.

  @rtype:  str
  @return: New trajectory after replacement.
  """
  assert(t1 > t0)
  
  new_chunk_list = []
  i0, rem0 = original_traj.FindChunkIndex(t0)
  i1, rem1 = original_traj.FindChunkIndex(t1)
       
  ## check if t0 falls in the first chunk. 
  ## if not, insert chunk 0 to chunk i0 - 1 into new_chunk_list
  if i0 > 0:
    for c in original_traj.chunkslist[0: i0]:
      new_chunk_list.append(c)

  ## remainderchunk0
  rem_chunk0 = Trajectory.Chunk(
                 rem0, original_traj.chunkslist[i0].polynomialsvector)
  new_chunk_list.append(rem_chunk0)

  ## insert traj_segment
  for c in traj_segment.chunkslist:
    new_chunk_list.append(c)

  ## remainderchunk1
  new_poly_list = []
  for p in original_traj.chunkslist[i1].polynomialsvector:
    ## perform variable changing of p(x) = a_n(x)^n + a_(n-1)(x)^(n-1) + ...
    ## by x = y + rem1
    
    a = p.q ## coefficient vector with python convention (highest degree first)
    ## a is a poly1d object
    r = a.r ## polynomial roots
    for i in range(len(r)):
      r[i] = r[i] - rem1
    b = np.poly1d(r, True) ## reconstruct a new polynomial from roots
    ## b is a poly1d object
    b = b*a.coeffs[0] ## multiply back by a_n *** this multiplication does not commute
    
    new_poly = Trajectory.Polynomial(b.coeffs.tolist()[::-1]) ## TOPP convention is weak-term-first
    new_poly_list.append(new_poly)
  rem_chunk1 = Trajectory.Chunk(original_traj.chunkslist[i1].duration - rem1, new_poly_list)
  new_chunk_list.append(rem_chunk1)
  
  ## insert remaining chunks
  if i1 < len(original_traj.chunkslist) - 1:
    for c in original_traj.chunkslist[i1 + 1: len(original_traj.chunkslist)]:
      new_chunk_list.append(c)

  return Trajectory.PiecewisePolynomialTrajectory(new_chunk_list)

def replace_lie_traj_segment(original_lie_traj, lie_traj_segment, t0, t1):
  """
  Replace the lie trajectory segment in time interval (C{t0}, C{t1}) 
  in {original_lie_traj} with the given C{lie_traj_segment}.

  @type  original_traj: lie.LieTraj
  @param original_traj: Original lie trajectory.
  @type   traj_segment: lie.LieTraj
  @param  traj_segment: New lie trajectory segment.
  @type             t0: float
  @param            t0: Start time of the time interval.
  @type             t1: float
  @param            t1: End time of the time interval.

  @rtype:  str
  @return: New lie trajectory after replacement.
  """
  assert(t1 > t0)

  new_traj_list = []
  new_R_list = []
  i0, rem0 = original_lie_traj.FindTrajIndex(t0)
  i1, rem1 = original_lie_traj.FindTrajIndex(t1)

  ## check if t0 falls in the first traj. 
  ## if not, insert traj 0 to traj i0 - 1 into new_traj_list
  if i0 > 0:
    for i in range(0, i0):
      new_traj_list.append(original_lie_traj.trajlist[i])
      new_R_list.append(original_lie_traj.Rlist[i])
  ## remaindertraj0
  new_chunk_list = []
  ic0, remc0 = original_lie_traj.trajlist[i0].FindChunkIndex(rem0)
  # print "c0", original_lie_traj.trajlist[i0].FindChunkIndex(rem0) ##
  # check if rem0 falls in the first chunk, if not, ...
  if ic0 > 0:
    for c in original_lie_traj.trajlist[i0].chunkslist[0: ic0]:
      new_chunk_list.append(c)
      # remainderchunk0
  rem_chunk0 = Trajectory.Chunk(remc0, original_lie_traj.trajlist[i0].\
                                chunkslist[ic0].polynomialsvector)
  new_chunk_list.append(rem_chunk0)

  rem_traj0 = Trajectory.PiecewisePolynomialTrajectory(new_chunk_list) 
  new_traj_list.append(rem_traj0)
  new_R_list.append(original_lie_traj.Rlist[i0])


  ## insert lie_traj_segment
  new_traj_list.append(lie_traj_segment)
  new_R_list.append(original_lie_traj.EvalRotation(t0))


  ## For the traj right after the lie_traj_segment 
  ## remaindertraj1
  new_chunk_list = []
  ic1, remc1 = original_lie_traj.trajlist[i1].FindChunkIndex(rem1)
  new_poly_list = []
  for p in original_lie_traj.trajlist[i1].chunkslist[ic1].polynomialsvector:
    ## perform variable changing of p(x) = a_n(x)^n + a_(n-1)(x)^(n-1) + ...
    ## by x = y + remc1
    
    a = p.q ## coefficient vector with python convention (highest degree first)
    ## a is a poly1d object
    r = a.r ## polynomial roots
    for i in range(len(r)):
      r[i] = r[i] - remc1
    b = np.poly1d(r, True) ## reconstruct a new polynomial from roots
    ## b is a poly1d object
    b = b*a.coeffs[0] ## multiply back by a_n *** this multiplication does not commute
    
    new_poly = Trajectory.Polynomial(b.coeffs.tolist()[::-1]) ## TOPP convention is weak-term-first
    new_poly_list.append(new_poly)
  rem_chunk1 = Trajectory.Chunk(original_lie_traj.trajlist[i1].chunkslist[ic1].duration - remc1, new_poly_list)
  new_chunk_list.append(rem_chunk1)

  ## insert remaining chunk 
  if ic1 < len(original_lie_traj.trajlist[i1].chunkslist) - 1:
    for c in original_lie_traj.trajlist[i1].chunkslist[ic1 + 1: \
            len(original_lie_traj.trajlist[i1].chunkslist)]:
        new_chunk_list.append(c)
  ## insert 
  rem_traj1 = Trajectory.PiecewisePolynomialTrajectory(new_chunk_list)
  new_traj_list.append(rem_traj1)
  new_R_list.append(original_lie_traj.Rlist[i1])##ROTATION Should be at original_lie_traj.Rlist[i1] ##

  # insert the remainder trajectoris
  if i1 < len(original_lie_traj.trajlist)-1:
    R_index = i1+1
    for t in original_lie_traj.trajlist[i1+1: \
              len(original_lie_traj.trajlist)]:
      new_traj_list.append(t)
      new_R_list.append(original_lie_traj.Rlist[R_index])
      R_index += 1

  return lie.LieTraj(new_R_list, new_traj_list)


########################### Distance computation ###########################

def SO3_distance(R0, R1):
  """
  Return distance between the given two rotation matrices in SO3 space.

  @type  R0: numpy.ndarray
  @param R0: Rotation matrix.
  @type  R1: numpy.ndarray
  @param R1: Rotation matrix.

  @rtype:  float
  @return: Distance.
  """
  dR = lie.logvect(dot(R0.T, R1))
  return np.sqrt(dot(dR,dR))

def R3_distance(p0, p1):
  """
  Return distance between the given two translational 3-vector.

  @type  p0: numpy.ndarray
  @param p0: Translational 3-vector.
  @type  p1: numpy.ndarray
  @param p1: Translational 3-vector.

  @rtype:  float
  @return: Distance.
  """
  dp = p0-p1
  return np.sqrt(dot(dp,dp))

def SE3_distance(T0, T1, c=None, d=None):
  """
  Return distance between the given two transformation matrices in SE3 space.

  @type  T0: numpy.ndarray
  @param T0: 4x4 transformation matrix.
  @type  T1: numpy.ndarray
  @param T1: 4x4 transformation matrix.

  @rtype:  float
  @return: Distance.
  """
  R0 = T0[:3, :3]
  R1 = T1[:3, :3]
  p0 = T0[:3, 3]
  p1 = T1[:3, 3]
  if (c is None):
    c = 1
  else: c = c
  if (d is None):
    d = 1
  else: d = d
  return np.sqrt(c*(SO3_distance(R0, R1)**2) + d*(R3_distance(p0, p1)**2))

def distance(q1, q2, metrictype=1):
  """
  Return distance between the given two robot configurations (joint values)
  according to the specified distance metric.

  @type  q1: numpy.ndarray
  @param q1: Robot joint configuration.
  @type  q2: numpy.ndarray
  @param q2: Robot joint configuration.
  @type metrictype: int
  @param metrictype: Metric type used for computing distance. Possible values:
                     -  1 : L2 norm squared
                     -  2 : L2 norm
                     -  3 : L1 norm

  @rtype:  float
  @return: Distance.
  """
  delta_q = q1 - q2
  if (metrictype == 1):
    return np.dot(delta_q, delta_q)
  elif (metrictype == 2):
    return np.linalg.norm(delta_q)    
  elif (metrictype == 3):
    return np.sum(np.abs(delta_q))
  else:
    raise Exception("Unknown Distance Metric.")

def generate_accumulated_dist_list(traj, metrictype=1, discr_timestep=0.005):
  """
  Return a list of accumulated robot config distances at each timestamp of 
  the given trajectory.

  @type            traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
  @param           traj: Trajectory to be discretized.
  @type      metrictype: int
  @param     metrictype: Metric type used for computing distance. 
                         Possible values:
                         -  0 : L2 norm squared
                         -  1 : L2 norm
                         -  2 : L1 norm
  @type  discr_timestep: float
  @param discr_timestep: Time resolution for generating accumulated 
                         distance list.

  @rtype:  list
  @return: A list of accumulated distance at each timestamp.
  """
  delta_dist_list = [0.0]
  timestamps = np.append(np.arange(discr_timestep, traj.duration, discr_timestep), traj.duration)

  q_prev = traj.Eval(0)
  for t in timestamps:
    q_new = traj.Eval(t)
    delta_dist = np.sqrt(distance(q_prev, q_new, metrictype))
    delta_dist_list.append(delta_dist)
    q_prev = np.array(q_new)

  accumulated_dist_list = [sum(delta_dist_list[0:i+1]) for i in xrange(len(delta_dist_list))]
  return accumulated_dist_list

def compute_accumulated_SE3_distance(lie_traj, translation_traj, t0=None, t1=None, discr_timestep=0.005):
  """
  Return accumulated SE3 distance between configurations at C{t0} and C{t1} 
  in the given trajectory.
  This approach is much faster than using 
  C{generate_accumulated_SE3_dist_list()}
  when comparing accumulated distance between trajectories.

  @type          lie_traj: lie.LieTraj
  @param         lie_traj: Lie trajectory to be discretized.
  @type  translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
  @param translation_traj: Translational trajectory to be discretized.
  @type                t0: float
  @param               t0: Start time for computing accumulated distance.
  @type                t1: float
  @param               t1: End time for computing accumulated distance.
  @type    discr_timestep: float
  @param   discr_timestep: Time resolution for computing accumulated distance.

  @rtype:  float
  @return: Accumulated diatance.
  """
  
  if t0 is None: 
    t0 = 0    
  if t1 is None: 
    t1 = lie_traj.duration
  eps = discr_timestep * 0.2
  if not (0 - eps) <= t0 < t1 <= (lie_traj.duration + eps):
      raise Exception('Incorrect time stamp.')

  accumulated_dist = 0
  timestamps = np.append(np.arange(t0 + discr_timestep, 
                                   t1, discr_timestep), t1)

  Tprev = np.eye(4)
  Tcur = np.eye(4)
  Tprev[0:3, 0:3] = lie_traj.EvalRotation(t0)
  Tprev[0:3, 3] = translation_traj.Eval(t0)
  for t in timestamps:
    Tcur[0:3, 0:3] = lie_traj.EvalRotation(t)
    Tcur[0:3, 3] = translation_traj.Eval(t)
    
    delta_dist = SE3_distance(Tprev, Tcur)
    accumulated_dist += delta_dist
    
    Tprev = np.array(Tcur)

  return accumulated_dist

def generate_accumulated_SE3_dist_list(lie_traj, translation_traj, 
                                       discr_timestep, 
                                       discr_compute_timestep):
  """
  Return a list of accumulated SE3 distances at each timestamp of the 
  given trajectory.
  The trajectory is to be sliced into M{n} segments.

  @type                lie_traj: lie.LieTraj
  @param               lie_traj: Lie trajectory to be discretized.
  @type        translation_traj: TOPP.Trajectory.PiecewisePolynomialTrajectory
  @param       translation_traj: Translational trajectory to be discretized.
  @type          discr_timestep: float
  @param         discr_timestep: Time resolution for generating accumulated
                                 distance list.
  @type  discr_compute_timestep: float
  @param discr_compute_timestep: Real time resolition used to compute 
                                 distance; each contains a cycle of 
                                 discr_timestep. All timesteps within one 
                                 cycle will share the same accumulated 
                                 distance value to ensure speed.

  @rtype:  list, length = M{n+1}
  @return: A list of accumulated distance at each timestamp.
  """
  delta_dist_list = []
  Tprev = np.eye(4)
  Tcur = np.eye(4)
  Tprev[0:3, 0:3] = lie_traj.EvalRotation(0)
  Tprev[0:3, 3] = translation_traj.Eval(0)
    
  timestamps = np.arange(discr_compute_timestep, lie_traj.duration, 
                         discr_compute_timestep)
  if not np.isclose(timestamps[-1], lie_traj.duration):
    np.append(timestamps, lie_traj.duration)

  cycle_length = int(discr_compute_timestep / discr_timestep)
  
  for t in timestamps:
    Tcur[0:3, 0:3] = lie_traj.EvalRotation(t)
    Tcur[0:3, 3] = translation_traj.Eval(t)
    
    delta_dist = SE3_distance(Tprev, Tcur)
    delta_dist_list.append(delta_dist)
    
    Tprev = np.array(Tcur)

  accumulated_dist_list = [0.0]
  for i in xrange(len(delta_dist_list)):
    accumulated_dist_list += [sum(delta_dist_list[0:i+1])] * cycle_length

  return accumulated_dist_list

def _is_close_axis(axis, target_axis):
  """    
  Check wether the two given axis (extracted from their respective 
  quaternion) belong to the same class.

  @type         axis: list
  @param        axis: Axis vector.
  @type  target_axis: list
  @param target_axis: Axis vector.

  @rtype:  bool
  @return: B{True} is they are in the same class.
  """
  axis_neg = axis * -1.
  if sum(abs(axis - target_axis)) < sum(abs(axis_neg - target_axis)):
    return True
  return False

######################## Trajtectory manipulation ########################
def merge_timestamps_list(timestamps_list):
  """
  Return a list of timestamps by merging all lists of timestamps in the 
  given list.

  @type  timestamps_list: list
  @param timestamps_list: list containing all lists of timestamps to 
  be merged.

  @rtype:  list
  @return: A list of merged timstamps.
  """

  new_timestamps = None
  for T in timestamps_list:
    if new_timestamps is None:
      new_timestamps = []
      offset = 0
    else:
      T.pop(0)
      offset = new_timestamps[-1]

    newT = [t + offset for t in T]
    new_timestamps = new_timestamps + newT
  return new_timestamps

def merge_wpts_list(wpts_list, eps=1e-3):
  """
  Return a list of waypoints by merging all lists of waypoints in the 
  given list.

  @type  wpts_list: list
  @param wpts_list: list containing all lists of waypoints to be merged.

  @rtype:  list
  @return: A list of merged waypoints.
  """
  new_wpts = []
  for W in wpts_list:
    if not new_wpts == []:
      # Check soundness
      try:
        assert(distance(W[0], new_wpts[-1]) < eps)
      except:
        raise Exception('Waypoints not match')
      W.pop(0)

    new_wpts = new_wpts + W
    
  return new_wpts


def merge_bimanual_trajs_wpts_list(bimanual_trajs, eps=1e-3):
  """
  Merge lists of bimanual waypoints in the given bimanual trajectories, which
  may contain regrasp actions stored in dict.
  """
  new_bimanual_trajs = []
  traj_num = len(bimanual_trajs)
  before_regrasp = True
  i = 0
  while i < traj_num:
    while type(bimanual_trajs[i]) is dict:
      new_bimanual_trajs.append(bimanual_trajs[i])
      before_regrasp = False
      i += 1

    bimanual_wpts_list = [[], []]
    while (i < traj_num) and (type(bimanual_trajs[i]) is not dict):
      bimanual_wpts_list[0].append(bimanual_trajs[i][0])
      bimanual_wpts_list[1].append(bimanual_trajs[i][1])
      i += 1

    if before_regrasp:
      new_bimanual_trajs.append([merge_wpts_list(bimanual_wpts_list[0]),
                                 merge_wpts_list(bimanual_wpts_list[1])])
    else:
      new_bimanual_trajs.append([merge_wpts_list(bimanual_wpts_list[0])[1:],
                                 merge_wpts_list(bimanual_wpts_list[1])[1:]])
    
  return new_bimanual_trajs

def discretize_wpts(q_init, q_final, step_count):
  """
  Return a list of waypoints interpolated start and end waypoints
  (configurations), discretized according to C{step_count}.

  @type      q_init: numpy.ndarray
  @param     q_init: Start configuration.
  @type     q_final: numpy.ndarray
  @param    q_final: End configuration.
  @type  step_count: int
  @param step_count: Total number of waypoints to generate excluding q_init.

  @rtype:  list
  @return: A list of interpolated waypoints. This includes C{q_final}
           but not C{q_init}, with a length of C{step_count}.
  """
  assert(step_count > 0)
  q_delta = (q_final - q_init) / step_count
  wpts = []
  for i in xrange(step_count):
    wpts.append(q_init + q_delta*(i+1))
  return wpts

def arange(start, end, step):
  """
  An discretization function almost same as np.arange() when difference
  between C{start} and C{end} is multiple of C{step}, but fixing possible
  error caused by float rounding so that C{end} is always excluded.
  """
  array = np.arange(start, end, step)
  if np.isclose(array[-1], end):
    array = array[:-1]
  return array

########################### Manipulator related ###########################
def compute_endeffector_transform(manip, q):
  """
  Return the end-effector transform of the manipulator at given 
  the configuration.

  @type  manip: openravepy.Manipulator
  @param manip: Manipulator to be used.
  @type      q: numpy.ndarray
  @param     q: Configuration of the manipulator.

  @rtype:  numpy.ndarray
  @return: A 4x4 transformation matrix of the manipulator's end-effector.
  """
  robot = manip.GetRobot()
  with robot:
    robot.SetActiveDOFValues(q)
    T = manip.GetEndEffectorTransform()
  return T

def compute_bimanual_goal_configs(robots, obj, q_robots_cur, q_robots_grasp, 
                                  T_obj_cur, T_obj_goal, seeds=[None, None], 
                                  reset=True):
  """
  Return one set of IK solutions of the given robot grasping the 
  given object, when the object moves to a new transformation.

  @type          robots: list of openravepy.Robot
  @param         robots: A list of robots to be used.
  @type             obj: openravepy.KinBody
  @param            obj: Object grasped.
  @type    q_robots_cur: list
  @param   q_robots_cur: Current configuration of the robots.
  @type  q_robots_grasp: list
  @param q_robots_grasp: Current configuration of the grippers.
  @type       T_obj_cur: numpy.ndarray
  @param      T_obj_cur: Current transformation of the obj.
  @type      T_obj_goal: numpy.ndarray
  @param     T_obj_goal: Goal transformation of the obj.
  @type           seeds: list
  @param          seeds: Index to speficy which IK solution to return, if not
                         set, return the closest ones.
  @type           reset: bool
  @param          reset: Whether to restore all robots and objects to their
                         original pose after computation.

  @rtype:  list
  @return: Goal configurations of the robots.
  """
  def restore():
    obj.Enable(True)
    if reset: obj.SetTransform(T_obj_cur)
    for (robot, q_robot_cur, q_robot_grasp) in zip(robots, q_robots_cur,
                                                   q_robots_grasp):
      robot.Enable(True)
      if reset: robot.SetDOFValues(np.append(q_robot_cur, q_robot_grasp))

  obj.Enable(False)
  for robot in robots:
    robot.Enable(False)

  q_robots_new = []
  for (robot, q_robot_cur, q_robot_grasp, seed) in zip(robots, q_robots_cur,
                                                       q_robots_grasp, seeds):
    robot.Enable(True)
    robot.SetActiveDOFValues(q_robot_cur)
    manip = robot.GetActiveManipulator()
    T_ef_cur = compute_endeffector_transform(manip, q_robot_cur)
    T_ef_new = np.dot(T_obj_goal, np.dot(np.linalg.inv(T_obj_cur), T_ef_cur))
    if seed is None:
      q_robot_new = manip.FindIKSolution(
        T_ef_new, orpy.IkFilterOptions.CheckEnvCollisions)
    else:
      q_robot_new = manip.FindIKSolutions(
        T_ef_new, orpy.IkFilterOptions.CheckEnvCollisions)[seed]
    if q_robot_new is None:
      restore()
      print robot.GetName() + ': No IK solution exists.'
      return None
    q_robots_new.append(q_robot_new)
    robot.SetDOFValues(np.append(q_robot_new, q_robot_grasp * 0.7))

  obj.SetTransform(T_obj_goal)
  obj.Enable(True)
  if obj.GetEnv().CheckCollision(obj):
    restore()
    print 'Object in collision.'
    return None

  restore()
  return q_robots_new
  
########################### Visualization ###########################
def visualize_config_transition(robot, q_start, q_goal, step_num=50, 
                                timestep=0.05):
  q_start = np.array(q_start)
  q_goal = np.array(q_goal)
  delta = (q_goal - q_start)/step_num
  for i in xrange(step_num+1):
    robot.SetActiveDOFValues(q_start + delta * i)
    sleep(timestep)

########################### Output formatting ###########################
colors = dict()
colors['black'] = 0
colors['red'] = 1
colors['green'] = 2
colors['yellow'] = 3
colors['blue'] = 4
colors['magenta'] = 5
colors['cyan'] = 6
colors['white'] = 7
def colorize(string, color='white', bold=True):
  """
  Return a string by formatting the given one with specified color and 
  boldness.

  @type  string: str
  @param string: String to be formatted.
  @type   color: str
  @param  color: Color specification.
  @type    bold: bool
  @param   bold: B{True} if the string is to be formatted in bold.

  @rtype:  str
  @return: The formatted string.
  """
  new_string = '\033['
  new_string += (str(int(bold)) + ';')
  new_string += ('3' + str(colors[color]))
  new_string += 'm'
  new_string += string
  new_string += '\033[0m' # reset the subsequent text back to normal
  return new_string
