from openravepy import *
from pylab import *

from TOPP import Trajectory
from TOPP import Utilities

import string
import numpy as np
import lie

INF = np.inf
EPS = 1e-12

############################## Trajectory related ##############################
def poly_critical_points(p, interval=None):
    '''
    Return the critical points of the numpy.poly1d object p.
    '''
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
    '''
    Generate polynomial trajectory string up to 10 decimal places accuracy to guarantee continuity at trajectory end
    '''
    traj_str = ''
    ndof = len(q_beg)
    traj_str += "%f\n%d"%(duration, ndof)
    for k in range(ndof):
        a, b, c, d = Utilities.Interpolate3rdDegree(q_beg[k], q_end[k], qd_beg[k], qd_end[k], duration)
        traj_str += "\n%.10f %.10f %.10f %.10f"%(d, c, b, a)
    return traj_str

def traj_str_5th_degree(q_beg, q_end, qd_beg, qd_end, qdd_beg, qdd_end, duration):
    '''
    Generate polynomial trajectory string up to 10 decimal places accuracy to guarantee continuity at trajectory end
    '''
    traj_str = ''
    ndof = len(q_beg)
    traj_str += "%f\n%d"%(duration, ndof)
    for k in range(ndof):
        a, b, c, d, e, f = Utilities.Interpolate5thDegree(q_beg[k], q_end[k], qd_beg[k], qd_end[k], qdd_beg[k], qdd_end[k], duration)
        traj_str += "\n%.10f %.10f %.10f %.10f %.10f %.10f"%(f, e, d, c, b, a)
    return traj_str

def check_config_DOF_limits(robot, q):
    position_limits = robot.GetDOFLimits()[1]

    for i in len(q):
        if abs(q[i]) > position_limits[i]: 
            return False

    return True

def check_traj_str_DOF_limits(robot, traj_str):
    traj_info = string.split(traj_str, "\n")
    dur = float(traj_info[0])
    ndof = int(traj_info[1])
    position_limits = robot.GetDOFLimits()[1]
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
            if abs(q(x)) > position_limits[i]:
                return False
        
        # check DOF velocities
        for x in qd_cri_points:
            if abs(qd(x)) > velocity_limits[i]:
                return False

    return True

def check_translation_traj_str_limits(upper_limits, lower_limits, traj_str):
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
            if not lower_limits[i] <= abs(p(x)) <= upper_limits[i]:
                return False
     
    return True
 
def traj_str_from_traj_list(traj_list):
    traj_str_list = [str(traj) for traj in traj_list]      
    traj_str = ""
    for i in range(len(traj_str_list)):
        traj_str += "\n"
        traj_str += str(traj_str_list[i])
    traj_str = string.lstrip(traj_str) # remove leading "\n"
    return traj_str

def replace_traj_segment(original_traj, traj_segment, t0, t1):
    '''
    Replace the segment (t0, t1) in the (arbitrary degree) original_traj 
    with an (arbitrary degree) traj_segment.
    '''
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
    rem_chunk0 = Trajectory.Chunk(rem0, original_traj.chunkslist[i0].polynomialsvector)
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
    '''
    Replace the lie traj segment (t0, t1) in the (arbitrary degree) original_lie_traj 
    with an (arbitrary degree) lie_traj_segment.
    '''
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
    rem_chunk0 = Trajectory.Chunk(remc0, original_lie_traj.trajlist[i0].chunkslist[ic0].polynomialsvector)
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
        for c in original_lie_traj.trajlist[i1].chunkslist[ic1 + 1: len(original_lie_traj.trajlist[i1].chunkslist)]:
                new_chunk_list.append(c)
    ## insert 
    rem_traj1 = Trajectory.PiecewisePolynomialTrajectory(new_chunk_list)
    new_traj_list.append(rem_traj1)
    new_R_list.append(original_lie_traj.Rlist[i1])##ROTATION Should be at original_lie_traj.Rlist[i1] ##

    # insert the remainder trajectoris
    if i1 < len(original_lie_traj.trajlist)-1:
        R_index = i1+1
        for t in original_lie_traj.trajlist[i1+1: len(original_lie_traj.trajlist)]:
            new_traj_list.append(t)
            new_R_list.append(original_lie_traj.Rlist[R_index])
            R_index += 1

    return lie.LieTraj(new_R_list, new_traj_list)


############################## Distance computation ##############################

def SO3_distance(R0, R1): # bi-invariance
    return np.linalg.norm(lie.logvect(dot(R0.T, R1)))

def R3_distance(b0, b1):
        return np.linalg.norm(b1-b0)

def SE3_distance(T0, T1, c=None, d=None): # left invariance
    R0 = T0[:3, :3]
    R1 = T1[:3, :3]
    b0 = T0[:3, 3]
    b1 = T1[:3, 3]
    if (c is None):
        c = 1
    else: c = c
    if (d is None):
        d = 1
    else: d = d
    return np.sqrt(c*(SO3_distance(R0, R1)**2) + d*(R3_distance(b0, b1)**2))

def distance(q1, q2, metrictype=1):
    '''
    Return the distance between two robot configurations (joint values) according to the specified distance metric.
    
    metrictype
    0 : L2 norm squared
    1 : L2 norm
    2 : L1 norm
    '''
    delta_q = q1 - q2
    if (metrictype == 1):
        return np.dot(delta_q, delta_q)
    elif (metrictype == 2):
        return np.linalg.norm(delta_q)    
    elif (metrictype == 3):
        return np.linalg.norm(delta_q, 1)   
    else:
        raise Exception("Unknown Distance Metric.")

def generate_accumulated_dist_list(traj, metrictype=1, discr_timestep=0.005):
    '''
    Return a list of accumulated robot config distances at each timestamp of the given trajectory
    '''
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
    '''
    Return accumulated SE3 distance in the given trajectory by computing directly.
    This approach is much faster than using generate_accumulated_SE3_dist_list().
    '''
    
    if t0 is None: 
        t0 = 0    
    if t1 is None: 
        t1 = lie_traj.duration
    if not 0 <= t0 < t1 <= lie_traj.duration:
        raise Exception('Incorrect time stamp.')

    accumulated_dist = 0
    timestamps = np.append(np.arange(t0 + discr_timestep, t1, discr_timestep), t1)

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

def generate_accumulated_SE3_dist_list(lie_traj, translation_traj, discr_timestep, discr_compute_timestep):
    '''
    Return a list of accumulated SE3 distances at each timestamp of the given trajectory (including lie traj and trans traj)
    NB: The trajectory is to be sliced into n segments

    Parameters
    ----------
    discr_timestep: float, length = n
        Timestep between each item in the returned list
    discr_compute_timestep: float, length = n+1
        Real timestep used to compute distance, each contains a cycle of discr_timestep. All timesteps within one cycle will share the same accumulated distance value to ensure speed.
    '''
    delta_dist_list = []
    Tprev = np.eye(4)
    Tcur = np.eye(4)
    Tprev[0:3, 0:3] = lie_traj.EvalRotation(0)
    Tprev[0:3, 3] = translation_traj.Eval(0)
        
    timestamps = np.arange(discr_compute_timestep, lie_traj.duration, discr_compute_timestep)
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


########################### Trajtectory manipulation ###########################
def merge_timestamps_list(timestamps_list):
    '''
    merge_timestamps_list merges timestamps lists of the form [0, 1, 2,
    ..., T] together.
    '''
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
    new_wpts = None
    for W in wpts_list:
        if new_wpts is None:
            new_wpts = []
        else:
            # Check soundness
            try:
                assert(distance(W[0][0:6], new_wpts[-1][0:6]) < eps)
            except:
                print 'W', W[0]
                print 'new', new_wpts[-1]
                assert(False)
            W.pop(0)

        new_wpts = new_wpts + W
        
    return new_wpts

def discretize_wpts(q_init, q_final, step_count):
    '''
    Return a list of waypoints in-between initial and final waypoints, discretized according to step_count.
    q_init will not be in the list.

    Parameters
    ----------
    q_init: numpy.ndarray
    q_final: numpy.ndarray
    step_count: int
        Total number of waypoints to generate excluding q_init
    '''
    q_delta = (q_final - q_init) / step_count
    wpts = []
    for i in xrange(step_count):
        wpts.append(q_init + q_delta*(i+1))
    return wpts

############################## Manipulator related ##############################
def compute_endeffector_transform(manip, q):
    '''
    Return the end-effector transform of the manipulator at given configuration q
    '''
    robot = manip.GetRobot()
    q_orig = robot.GetActiveDOFValues()
    robot.SetActiveDOFValues(q)
    T = manip.GetEndEffectorTransform()
    robot.SetActiveDOFValues(q_orig)
    return T


def disable_gripper(robots):
    '''
    Set active DOFs of all robots in the given list to match their active manipulator
    '''
    for robot in robots:
        manip = robot.GetActiveManipulator()
        robot.SetActiveDOFs(manip.GetArmIndices())

def load_IK_model(robots):
    '''
    Load openrave IKFast model for all robots in the given list
    '''
    for robot in robots:    
        ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, iktype=IkParameterization.Type.Transform6D)    
        if (not ikmodel.load()):
            robot_name = robot.GetName()
            rospy.loginfo('Robot:[' + robot_name + '] IKFast not found. Generating IKFast solution...')
            ikmodel.autogenerate()

def scale_DOF_limits(robot, v=1, a=1):
    '''
    Adjust DOF limits of the given robot by specified multiplier
    '''
    robot.SetDOFVelocityLimits(robot.GetDOFVelocityLimits() * v)
    robot.SetDOFAccelerationLimits(robot.GetDOFAccelerationLimits() * a)

############################## Output formatting ##############################
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
    newstring = '\033['
    newstring += (str(int(bold)) + ';')
    newstring += ('3' + str(colors[color]))
    newstring += 'm'
    newstring += string
    newstring += '\033[0m' # reset the subsequent text back to normal
    return newstring
