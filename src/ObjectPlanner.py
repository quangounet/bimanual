import openravepy as orpy
import random
import numpy as np
import time
import copy

from pymanip.utils import Heap
from pymanip.utils import ObjectPreprocessing as op
from pymanip.utils import parabint_utilsformanip as pu
from pymanip.utils import Grasp as gr
from pymanip.utils.Grasp import (pX, pY, pZ, mX, mY, mZ)
from pymanip.utils import Utils
from pymanip.utils.Utils import Colorize
import TOPP

# SE3 planning for objects
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/rrtse3/')
import SE3RRT
import SE3Utils
import lie as Lie

# cvxopt for object tracking planner
import cvxopt
import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False # disable cvxopt output

# DEBUG
import pickle

############################################################
#                    Global Parameters
############################################################
# Planner parameters
FW = 0
BW = 1
REACHED = 0
ADVANCED = 1
TRAPPED = 2

# Unimanual
CG = 1
CP = 2
CGCP = 3
WORDS = ['', 'CG', 'CP', 'CGCP']
DICTIONARY = {'CG': 'CG', 
              'CP': 'CP',
              'CGCP': 'CGCP'}

# Manipulation trajectory types
TRANSIT = 0
TRANSFER = 1

EPSILON = 1e-10
DELTA = 0 # for patabint

CONTROLLERTIMEOUT = 5.
OPENRAVEPLANNERNAME = 'birrt'
            
    
############################################################
#                     Object Planner
############################################################
class ObjectPlanner(object):
    """
    This class is a planner for planning object motions in SE(3). The
    core planning implementation is done by Huy. Interface written by
    Puttichai.
    """
    def __init__(self, mobj):
        self.object = mobj
        self._print = True

        # For SE3 Planning
        self._objectplanningtimeout = 60. # max time allowed for object traj planning
        self._discrtimestep = 1e-3 # for TOPP
        self._objecttaumax = np.ones(3) # object 'virtual' actuator limits
        self._objectfmax = np.ones(3) # object 'virtual' force limits
        self._objectvmax = 10.*np.ones(6)
        self._shortcutiter = 300


    def Plan(self, Tinit, Tgoal):
        """
        Parameters
        ----------
        Tinit : 4x4 obejct transformation
        Tgoal : 4x4 obejct transformation

        Returns
        -------
        plannerstatus : bool
        se3traj
        Rlist
        """
        # Initial configuration
        R0 = Tinit[0:3, 0:3]
        quat0 = orpy.quatFromRotationMatrix(R0)
        p0 = Tinit[0:3, 3]
        w0 = np.zeros(3)
        v0 = np.zeros(3)
        # Goal configuration
        R1 = Tgoal[0:3, 0:3]
        quat1 = orpy.quatFromRotationMatrix(R1)
        p1 = Tgoal[0:3, 3]
        w1 = np.zeros(3)
        v1 = np.zeros(3)
        # Create vertices for the RRT planner
        vertex_init = SE3RRT.Vertex(SE3RRT.Config(quat0, p0, w0, v0), SE3RRT.FW)
        vertex_goal = SE3RRT.Vertex(SE3RRT.Config(quat1, p1, w1, v1), SE3RRT.BW)
        planner = SE3RRT.RRTPlanner(vertex_init, vertex_goal, self.object)

        # Set translational limits for sampling
        pinit = Tinit[0:3, 3]
        pgoal = Tgoal[0:3, 3]

        offsets = [0.1, 0.1, 0.1]
        upperlimits = [max(pinit[i], pgoal[i]) + offsets[i] for i in xrange(3)]
        lowerlimits = [min(pinit[i], pgoal[i]) - offsets[i] for i in xrange(3)]
        planner.SetTranslationalLimits(upperlimits, lowerlimits)

        planner.Run(self._objectplanningtimeout)

        plannersuccessful = planner.result
        if not plannersuccessful:
            if self._print:
                print '[PlanObjectTrajectory]',
                print Colorize('SE3RRT planner failed', 'red')
            return [False, None, None]
        
        if self._print:
            print '[PlanObjectTrajectory]',
            print Colorize('SE3RRT planner successful', 'green')
        
        # Prepare for running TOPP
        Rlist0 = planner.GenFinalRotationMatrixList()
        trajrotlist0 = planner.GenFinalTrajList()
        lietraj0 = Lie.LieTraj(Rlist0, trajrotlist0)
        transtraj0 = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString\
        (planner.GenFinalTrajTranString())
        rottraj0 = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString\
        (SE3Utils.TrajStringFromTrajList(trajrotlist0))
        se3traj0 = SE3Utils.SE3TrajFromTransandSO3(transtraj0, rottraj0)

        # Compute TOPP constraints
        a, b, c = SE3Utils.ComputeSE3Constraints(se3traj0, self._objecttaumax,
                                                 self._objectfmax, self._discrtimestep)

        # Run TOPP
        toppinst = TOPP.QuadraticConstraints(se3traj0, 
                                             self._discrtimestep, self._objectvmax, 
                                             list(a), list(b), list(c))
        toppsolver = toppinst.solver
        res = toppsolver.RunComputeProfiles(0.0, 0.0)
        if not (res == 1):
            if self._print:
                print '[PlanObjectTrajectory]',
                print Colorize('TOPP failed', 'red')
            return [False, None, None]
        if self._print:
            print '[PlanObjectTrajectory]',
            print Colorize('TOPP successful', 'green')

        toppsolver.ReparameterizeTrajectory()
        toppsolver.WriteResultTrajectory()
        se3traj1 = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString\
        (toppsolver.restrajectorystring)
        # transtraj1, rottraj1 = SE3Utils.TransRotTrajFromSE3Traj(se3traj1)
        # lietraj1 = Lie.SplitTraj2(Rlist0, rottraj1)

        # Shortcut
        se3traj2, Rlist2 = SE3Utils.SE3Shortcut(self.object, self._objecttaumax,
                                                self._objectfmax, self._objectvmax,
                                                se3traj1, Rlist0, self._shortcutiter)
        
        with open('debugse3.pkl', 'wb') as f:
            pickle.dump([se3traj2, Rlist2], f, pickle.HIGHEST_PROTOCOL)

        transtraj2, rottraj2 = SE3Utils.TransRotTrajFromSE3Traj(se3traj2)
        lietraj2 = Lie.SplitTraj2(Rlist2, rottraj2)
        if self._print:
            print '[PlanObjectTrajectory]',
            print Colorize('Shortcutting successful', 'green')

        # Compute TOPP constraints
        a, b, c = SE3Utils.ComputeSE3Constraints(se3traj2, self._objecttaumax,
                                                 self._objectfmax, self._discrtimestep)
        # Run TOPP
        toppinst = TOPP.QuadraticConstraints(se3traj2, 
                                             self._discrtimestep, self._objectvmax, 
                                             list(a), list(b), list(c))
        toppsolver = toppinst.solver
        res = toppsolver.RunComputeProfiles(0.0, 0.0)
        if not (res == 1):
            if self._print:
                print '[PlanObjectTrajectory]',
                print Colorize('TOPP(2) failed', 'red')
            return [False, None, None]
        if self._print:
            print '[PlanObjectTrajectory]',
            print Colorize('TOPP(2) successful', 'green')

        toppsolver.ReparameterizeTrajectory()
        toppsolver.WriteResultTrajectory()
        se3traj3 = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString\
        (toppsolver.restrajectorystring)
        # transtraj3, rottraj3 = SE3Utils.TransRotTrajFromSE3Traj(se3traj3)
        # lietraj3 = Lie.SplitTraj2(Rlist2, rottraj3)
        return [True, se3traj3, Rlist2]


    def PlayObjectTrajectory(self, lietraj, transtraj, dt=0.01, timemult=1.0):
        sleeptime = dt*timemult
        # # Release the object from robots (if they are grasping)
        # for robotindex in xrange(self.robots):
        #     self.taskmanips[robotindex].ReleaseFingers()
        #     self.robots[robotindex].WaitForController(CONTROLLERTIMEOUT)
        #     self.robots[robotindex].Release(self.object)

        T = np.eye(4) # dummy variable for object's transformation
        dur = lietraj.duration
        for t in np.arange(0, dur, dt):
            T[0:3, 0:3] = lietraj.EvalRotation(t)
            T[0:3, 3] = transtraj.Eval(t)
            self.object.SetTransform(T)
            time.sleep(sleeptime)
        T[0:3, 0:3] = lietraj.EvalRotation(dur)
        T[0:3, 3] = transtraj.Eval(dur)
        self.object.SetTransform(T)        
        
        
############################################################
#                 Object Tracking Planner
############################################################
class ObjectTrackingPlanner(object):

    def __init__(self, robot, manipulatorname, mobj, active_dofs=6):
        self.robot = robot
        self.manip = self.robot.SetActiveManipulator(manipulatorname)
        self.object = mobj
        self.active_dofs = active_dofs
        self._vmax = self.robot.GetDOFVelocityLimits()[0:self.active_dofs]
        
        self._print = True

        # For QP solver
        self._maxiter = 10000
        self._weight = 10.
        self._tol = 1e-5
        self._gain = 10.
        self._dt = 0.1 # time step for ik solver


    def Plan(self, lietraj, transtraj, qgrasp, q0=None, qd0=None, timestep=0.01):
        """
        Plan plans a trajectory for the robot following the object
        trajectory specified by lietraj and transtraj.

        Parameters
        ----------
        lietraj
        transtraj
        qgrasp : 4-vector
        q0 : n-vector, optional
            q0 can be provided as an initial configuration for the robot. 
            If this is not given, an IK solution will be calculated one.
        qd0 : n-vector, optional
            qd0 is an initial joint velocity for the robot.
            If not given, it will be set to a zero-vector.
        timestep : float, optional
            A time resolution for tracking.

        Returns
        -------
        plannerstatus : bool
        waypoints : list
            A list containing vectors (q, qd)
        timestamps : list
            A list containing timestamps of waypoints
        """
        graspedlink = self.object.GetLinks()[qgrasp[0]]
        extents = graspedlink.GetGeometries()[0].GetBoxExtents()
        dur = lietraj.duration
        Trel = np.dot(np.linalg.inv(self.object.GetTransform()),
                      graspedlink.GetTransform())
        
        # Trajectory tracking loop
        waypoints = [] # a list containing waypoints along the way
        timestamps = []
        if self._print:
            print '[ObjectTrackingPlanner::Plan]',
            print Colorize('t = {0}'.format(0), 'blue')
        M = np.eye(4) # dummy variable for the object's transformation
        M[0:3, 0:3] = lietraj.EvalRotation(0)
        M[0:3, 3] = transtraj.Eval(0)
        T = Utils.ComputeTGripper(np.dot(M, Trel), qgrasp, extents)

        if q0 is None:
            qprev = self.manip.FindIKSolution(T, 
                                              orpy.IkFilterOptions.CheckEnvCollisions)
            if qprev is None:
                print '[ObjectTrackingPlanner::Plan]'
                print Colorize('No IK solution at the initial configuration', 'red')
                return [False, None]
        else:
            qprev = q0
        if qd0 is None:
            qdprev = np.zeros(self.robot.GetActiveDOF())
        else:
            qdprev = qd0

        qnext, qdnext = self.ComputeIK(T, qprev, qdprev)
        waypoints.append(np.hstack([qnext, qdnext]))
        timestamps.append(0.)
        qprev = qnext
        qdprev = qdnext
        
        for t in np.arange(timestep, dur, timestep):
            if self._print:
                print '[ObjectTrackingPlanner::Plan]',
                print Colorize('t = {0}'.format(t), 'blue')
            M[0:3, 0:3] = lietraj.EvalRotation(t)
            M[0:3, 3] = transtraj.Eval(t)
            T = Utils.ComputeTGripper(np.dot(M, Trel), qgrasp, extents)
            qnext, qdnext = self.ComputeIK(T, qprev, qdprev)
            waypoints.append(np.hstack([qnext, qdnext]))
            timestamps.append(t)
            
            qprev = qnext
            qdprev = qdnext

        if self._print:
            print '[ObjectTrackingPlanner::Plan]',
            print Colorize('t = {0}'.format(dur), 'blue')
        M[0:3, 0:3] = lietraj.EvalRotation(dur)
        M[0:3, 3] = transtraj.Eval(dur)
        T = Utils.ComputeTGripper(np.dot(M, Trel), qgrasp, extents)
        qnext, qdnext = self.ComputeIK(T, qprev, qdprev)
        waypoints.append(np.hstack([qnext, qdnext]))
        timestamps.append(dur)
        
        return [True, waypoints, timestamps]

    
    def ComputeIK(self, T, q, qd):
        """
        ComputeIK computes an IK solution for a robot reaching a
        manipulator transformation T. q and qd are initial conditions.

        targetpose is a 7-vector, where the first 4 elements are from
        the quarternion of the rotation and the other 3 are from the
        translation vector.
        This implementation follows Stephane's pymanoid library.
        """
        R = T[0:3, 0:3]
        p = T[0:3, 3]
        targetpose = np.hstack([orpy.quatFromRotationMatrix(R), p])
        if targetpose[0] < 0:
            targetpose[0:4] *= -1.

        cur_obj = 5000. # some arbitrary number
        i = 0
        while i < self._maxiter:
            i += 1
            prev_obj = cur_obj
            cur_obj = self.ComputeObjective(targetpose, q, qd)
            if abs(cur_obj - prev_obj) < self._tol and cur_obj < self._tol:
                # Local minimum reached
                break
            
            qd = self.ComputeVelocity(targetpose, q, qd)
            qd = np.maximum(np.minimum(qd, self._vmax), -self._vmax)
            q = q + (qd * self._dt)
            
        if i == self._maxiter:
            print Colorize('[ComputeIK] max iteration exceeded', 'red')
        # Currently assume that it always converge
        return q, qd

    
    def ComputeObjective(self, targetpose, q, qd):
        error = self.ComputeError(targetpose, q, qd)
        return self._weight * np.dot(error, error)


    def ComputeError(self, targetpose, q, qd):
        with self.robot:
            self.robot.SetActiveDOFValues(q)
            currentpose = self.manip.GetTransformPose()
        if currentpose[0] < 0:
            currentpose[0:4] *= -1.
        error = targetpose - currentpose
        return error


    def ComputeVelocity(self, targetpose, q, qd):
        with self.robot:
            self.robot.SetActiveDOFValues(q)
            self.robot.SetActiveDOFVelocities(qd)
            # Jacobian
            J_trans = self.manip.CalculateJacobian()
            J_quat = self.manip.CalculateRotationJacobian()
            
            currentpose = self.manip.GetTransformPose()
        if currentpose[0] < 0:
            currentpose[0:4] *= -1.
            J_quat *= -1.
        
        # Full Jacobian
        J = np.vstack([J_quat, J_trans])

        # # CVXOPT
        # P = self._weight * np.dot(J.T, J)
        # error = targetpose - currentpose
        # r = -self._weight * self._gain * np.dot(error.T, J)
        # G = np.vstack([np.eye(6), -np.eye(6)])
        # h = np.hstack([self._vmax, self._vmax])

        # return cvxopt_solve_qp(P, r, G, h)

        weight = 1.0
        return weight * np.dot(np.linalg.pinv(J), (targetpose - currentpose))
        

############################################################
#                     Other Utilities
############################################################
#
# For CVXOPT (obtained from Stephane's pymanoid)
#
class OptimalNotFound(Exception):
    pass


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P_sym = .5 * (P + P.T)   # necessary for CVXOPT 1.1.7
    #
    # CVXOPT 1.1.7 only considers the lower entries of P
    # so we need to project on the symmetric part beforehand,
    # otherwise a wrong cost function will be used
    #
    M = cvxopt.matrix
    args = [M(P_sym), M(q)]
    if G is not None:
        args.extend([M(G), M(h)])
        if A is not None:
            args.extend([M(A), M(b)])
    sol = cvxopt.solvers.qp(*args)
    if not ('optimal' in sol['status']):
        raise OptimalNotFound(sol['status'])
    return np.array(sol['x']).reshape((P.shape[1], ))
