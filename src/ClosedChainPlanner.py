import openravepy as orpy
import numpy as np
import random
import time

from pymanip.utils.Grasp import (pX, pY, pZ, mX, mY, mZ)
from pymanip.utils import ObjectPreprocessing as op
from pymanip.utils import Utils
from pymanip.utils.Utils import Colorize
from pymanip.utils import Heap

# SE3 planning for objects
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/rrtse3/')
import SE3Utils
import lie as Lie

import TOPP

from CCTrajectory import CCTrajectory

from pymanip.utils import ObjectDatabase as od

"""
ClosedChainPlanner implements a new closed-chain motion planning which:
- incrementally plans an object trajectory with SE(3)RRT (work by Huy&Pham)
- ensures robot(s) reachability at each extension step

CC here refers to the word Closed-Chain.
"""
################################################################################
#                             Global Parameters
################################################################################
# Planner parameters
FW = 0
BW = 1
REACHED = 0
ADVANCED = 1
TRAPPED = 2

HOME = np.zeros(6)
IKCHECKCOLLISION = orpy.IkFilterOptions.CheckEnvCollisions

RNG = random.SystemRandom()

################################################################################
#                                  Config
################################################################################
class SE3Config(object):

    def __init__(self, q, p, qs=None, ps=None):
        quatlength = np.linalg.norm(q)
        self.q = q/quatlength
        if qs is None:
            self.qs = np.zeros(3)
        else:
            self.qs = qs

        self.p = p
        if ps is None:
            self.ps = np.zeros(3)
        else:
            self.ps = ps

        self.T = orpy.matrixFromPose(np.hstack([self.q, self.p]))

    @staticmethod
    def FromMatrix(T):
        quat = orpy.quatFromRotationMatrix(T[0:3, 0:3])
        p = T[0:3, 3]
        return SE3Config(quat, p)        


class CCConfig(object):

    def __init__(self, se3config, qrobots, qgrasps, configtype=FW):
        self.se3config = se3config
        self.qrobots = qrobots
        self.qgrasps = qgrasps
        self.configtype = configtype


class CCVertex(object):
    
    def __init__(self, config, vertextype=FW):
        self.config = config
        self.vertextype = vertextype
        
        # These parameters are to be assigned when the vertex is added
        # to the tree
        self.index = 0
        self.parentindex = None
        self.rotationaltraj = None # TOPP trajectory
        self.translationaltraj = None # TOPP trajectory
        self.robotwps = []
        self.timestamps = []
        self.level = 0


class CCTree(object):
    
    def __init__(self, vroot=None, treetype=FW):
        self.verticeslist = []
        self.length = 0
        if vroot is not None:
            self.verticeslist.append(vroot)
            self.length += 1

        self.treetype = treetype


    def __len__(self):
        return len(self.verticeslist)


    def __getitem__(self, i):
        return self.verticeslist[i]


    def AddVertex(self, vnew, parentindex, rottraj, transtraj, robottrajs, timestamps):
        vnew.parentindex = parentindex
        vnew.rotationaltraj = rottraj
        vnew.translationaltraj = transtraj
        vnew.robotwps = robottrajs
        vnew.timestamps = timestamps
        vnew.index = self.length
        vnew.level = self.verticeslist[parentindex].level + 1
        
        self.verticeslist.append(vnew)
        self.length += 1
        

    def GenerateRotationalTrajectoryList(self, startindex=-1):
        rottrajslist = []

        vertex = self.verticeslist[startindex]
        while (vertex.parentindex is not None):
            parent = self.verticeslist[vertex.parentindex]
            rottrajslist.append(vertex.rotationaltraj)
            vertex = parent

        if (self.treetype == FW):
            rottrajslist.reverse()

        return rottrajslist


    def GenerateRotationMatricesList(self, startindex=-1):
        rotmatriceslist = []

        vertex = self.verticeslist[startindex]
        while (vertex.parentindex is not None):
            parent = self.verticeslist[vertex.parentindex]
            rotmatriceslist.append(vertex.config.se3config.T[0:3, 0:3])
            vertex = parent
        rotmatriceslist.append(self.verticeslist[0].config.se3config.T[0:3, 0:3])

        if (self.treetype == FW):
            rotmatriceslist.reverse()

        return rotmatriceslist


    def GenerateTranslationalTrajectoryList(self, startindex=-1):
        transtrajslist = []
            
        vertex = self.verticeslist[startindex]
        while (vertex.parentindex is not None):
            parent = self.verticeslist[vertex.parentindex]
            transtrajslist.append(vertex.translationaltraj)
            vertex = parent

        if (self.treetype == FW):
            transtrajslist.reverse()

        return transtrajslist


    def GenerateRobotWaypointsDict(self, startindex=-1):
        waypointsdict = dict()
        for i in xrange(2):
            waypointsdict[i] = []
            
        vertex = self.verticeslist[startindex]
        while (vertex.parentindex is not None):
            parent = self.verticeslist[vertex.parentindex]
            for i in xrange(2):
                waypointsdict[i].append(vertex.robotwps[i])
            vertex = parent

        if (self.treetype == FW):
            for i in xrange(2):
                waypointsdict[i].reverse()
                
        return waypointsdict


    def GenerateTimeStampsList(self, startindex=-1):
        timestampslist = []
        
        vertex = self.verticeslist[startindex]
        while (vertex.parentindex is not None):
            parent = self.verticeslist[vertex.parentindex]
            timestampslist.append(vertex.timestamps)
            vertex = parent

        if (self.treetype == FW):
            timestampslist.reverse()

        return timestampslist            


class CCQuery(object):
    """
    Class Query stores everything related to a single query.
    """
    def __init__(self, vstart, vgoal):
        if not (vstart.config.qgrasps == vgoal.config.qgrasps):
            raise ValueError('terminal vertices must have the same qgrasps')
        
        # Initialize RRTs
        self.treestart = CCTree(vstart, FW)
        self.treeend = CCTree(vgoal, BW)

        # Connection information
        self.connectingrotationaltraj = None
        self.connectingtranslationaltraj = None
        self.connectingrobotwps = None
        self.connectingtimestamps = None

        # Statistics
        self.solved = False
        self.iterations = 0
        self.runningtime = 0.0
        
        # Parameters
        self.nn = -1
        self.stepsize = 0.7 # for tree extension
        self.interpolationdur = 0.5
        self.discrtimestep = 1e-2 # for collision checking
        self.print_ = True
        self.hastranslationallimits = False


    def SetTranslationalLimits(self, upper, lower=[]):
        self.upperlimits = upper
        if len(lower) == 0:
            self.lowerlimits = -1.0*self.upperlimits
        else:
            self.lowerlimits = lower

        self.hastranslationalLimits = True

        
    def ExtractFinalRotationalTrajectoryList(self):
        if not self.solved:
            print 'This query has not been solved yet.'
            return []

        rottrajslistfw = self.treestart.GenerateRotationalTrajectoryList()
        rottrajslistbw = self.treeend.GenerateRotationalTrajectoryList()
        if (self.connectingrotationaltraj is not None):
            rottrajslistfw.append(self.connectingrotationaltraj)
        return rottrajslistfw + rottrajslistbw


    def ExtractFinalRotationMatricesList(self):
        if not self.solved:
            print 'This query has not been solved yet.'
            return []

        rotmatriceslistfw = self.treestart.GenerateRotationMatricesList()
        rotmatriceslistbw = self.treeend.GenerateRotationMatricesList()
        return rotmatriceslistfw + rotmatriceslistbw
        
    
    def ExtractFinalTranslationalTrajectoryList(self):
        if not self.solved:
            print 'This query has not been solved yet.'
            return []

        transtrajslistfw = self.treestart.GenerateTranslationalTrajectoryList()
        transtrajslistbw = self.treeend.GenerateTranslationalTrajectoryList()
        if (self.connectingtranslationaltraj is not None):
            transtrajslistfw.append(self.connectingtranslationaltraj)
        return transtrajslistfw + transtrajslistbw


    def ExtractFinalWaypointsDict(self):
        if not self.solved:
            print 'This query has not been solved yet.'
            return []

        wpsdictfw = self.treestart.GenerateRobotWaypointsDict()
        wpsdictbw = self.treeend.GenerateRobotWaypointsDict()
        if (self.connectingrobotwps is not None):
            for i in xrange(2):
                wpsdictfw[i].append(self.connectingrobotwps[i])
                
        for i in xrange(2):
            wpsdictfw[i] = wpsdictfw[i] + wpsdictbw[i]
        return wpsdictfw


    def ExtractFinalTimeStampsList(self):
        if not self.solved:
            print 'This query has not been solved yet.'
            return []
        
        timestampslistfw = self.treestart.GenerateTimeStampsList()
        timestampslistbw = self.treeend.GenerateTimeStampsList()
        if (self.connectingtimestamps is not None):
            timestampslistfw.append(self.connectingtimestamps)
        return timestampslistfw + timestampslistbw


class CCPlanner(object):
    """
    Requirements:
    - two identical robots
    """
    
    def __init__(self, manip_object, robotslist, manipulatorname, objectdatabase=None):
        self.object = manip_object
        self.robotslist = robotslist
        self.manips = []
        for (i, robot) in enumerate(self.robotslist):
            self.manips.append(robot.SetActiveManipulator(manipulatorname))
            robot.SetActiveDOFs(self.manips[i].GetArmIndices())

        self.ots = [ObjectTracker1(robot, manipulatorname, manip_object)
                    for robot in self.robotslist]

        self._activedofs = self.manips[0].GetArmIndices()
        self._vmax = self.robotslist[0].GetDOFVelocityLimits()[self._activedofs]
        self._amax = self.robotslist[0].GetDOFAccelerationLimits()[self._activedofs]

        self.env = self.object.GetEnv()
        self.hasquery = False

        if (objectdatabase is None):
            self.objectdb = od.ObjectDatabase(self.object)
        else:
            self.objectdb = objectdatabase


    def SampleSE3Config(self):
        """
        SampleSE3Config randomly samples an object transformation.
        This function does not do any feasibility checking since when
        extending a vertex on a tree to this config, we do not use
        this config directly.
        """ 
        q_rand = Lie.RandomQuat()
        p_rand = np.asarray([RNG.uniform(self._query.lowerlimits[i], 
                                         self._query.upperlimits[i]) 
                             for i in xrange(3)])
        
        qs_rand = (1e-3)*np.ones(3)
        ps_rand = np.zeros(3)

        return SE3Config(q_rand, p_rand, qs_rand, ps_rand)
    

    def Solve(self, query, timeout):
        self._query = query
        if self._query.solved:
            print Colorize('This query has already been solved.', 'green')
            return self._query

        t = 0.0
        prev_it = self._query.iterations
        it = 0
        
        tbeg = time.time()
        if (self.Connect() == REACHED):
            it += 1
            self._query.iterations += 1
            tend = time.time()
            self._query.runningtime += (tend - tbeg)
            
            print Colorize('Path found', 'green')
            print Colorize('    Total number of iterations : {0}'.format\
                               (self._query.iterations), 'green')
            print Colorize('    Total running time : {0} sec.'.format\
                               (self._query.runningtime), 'green')
            self._query.solved = True
            return self._query
        elaspedtime = time.time() - tbeg
        t += elaspedtime
        self._query.runningtime += elaspedtime

        while (t < timeout):
            it += 1
            self._query.iterations += 1
            print Colorize('iteration : {0}'.format(it), 'blue')
            tbeg = time.time()

            se3config = self.SampleSE3Config()
            if (self.Extend(se3config) != TRAPPED):
                print Colorize('Tree start : {0}; Tree end : {1}'.\
                                   format(len(self._query.treestart.verticeslist), 
                                          len(self._query.treeend.verticeslist)),
                               'green')

                if (self.Connect() == REACHED):
                    tend = time.time()
                    self._query.runningtime += (tend - tbeg)
                    print Colorize('Path found', 'green')
                    print Colorize('    Total number of iterations : {0}'.format\
                                       (self._query.iterations), 'green')
                    print Colorize('    Total running time : {0} sec.'.format\
                                       (self._query.runningtime), 'green')
                    self._query.solved = True
                    return self._query
                
            elaspedtime = time.time() - tbeg
            t += elaspedtime
            self._query.runningtime += elaspedtime

        print Colorize('Allotted time {0} sec. is exhausted after {1} iterations'.\
                           format(allottedtime, self._query.iterations - prev_it))
        return self._query


    def Extend(self, se3config):
        if True:#(np.mod(self._query.iterations - 1, 2) == FW):
            return self.ExtendFW(se3config)
        else:
            return self.ExtendBW(se3config)


    def ExtendFW(self, se3config):
        print '[ExtendFW]'
        status = TRAPPED
        nnindices = self.NearestNeighborIndices(se3config, FW)
        for index in nnindices:
            vnear = self._query.treestart[index]
            
            # quaternion
            qbeg = vnear.config.se3config.q
            qsbeg = vnear.config.se3config.qs
            
            # translation
            pbeg = vnear.config.se3config.p
            psbeg = vnear.config.se3config.ps

            # Check if se3config is too far from vnear.se3config
            se3dist = SE3Distance(se3config, vnear.config.se3config)
            if se3dist <= self._query.stepsize:
                qend = se3config.q
                pend = se3config.p
                status = REACHED
            else:
                qend = qbeg + self._query.stepsize*(se3config.q - qbeg)/np.sqrt(se3dist)
                qend /= np.linalg.norm(qend)
                pend = pbeg + self._query.stepsize*(se3config.p - pbeg)/np.sqrt(se3dist)
                status = ADVANCED

            qsend = se3config.qs
            psend = se3config.ps

            newse3c = SE3Config(qend, pend, qsend, psend)

            # Check collision (se3config)
            if not self.IsSE3ConfigCollisionFree(newse3c):
                print '    [ExtendFW] TRAPPED : SE(3) config in collision'
                status = TRAPPED
                continue

            # Check reachability (se3config)
            [passed, iksols] = self.CheckSE3ConfigReachability\
            (newse3c, vnear.config.qgrasps, vnear.config.qrobots)
            if not passed:
                print '    [ExtendFW] TRAPPED : SE(3) config not reachable'
                status = TRAPPED
                continue
            
            # Interpolate the object trajectory
            Rbeg = orpy.rotationMatrixFromQuat(qbeg)
            Rend = orpy.rotationMatrixFromQuat(qend)
            rottraj = Lie.InterpolateSO3(Rbeg,
                                         orpy.rotationMatrixFromQuat(qend),
                                         qsbeg, qsend, 
                                         self._query.interpolationdur)
            transtrajstring = SE3Utils.TrajString3rdDegree\
            (pbeg, pend, psend, psend, self._query.interpolationdur)
            transtraj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString\
            (transtrajstring)

            # Check collision (object trajectory)
            if not self.IsSE3TrajCollisionFree(rottraj, transtraj, Rbeg):
                print '    [ExtendFW] TRAPPED : SE(3) trajectory in collision'
                status = TRAPPED
                continue
            
            # Check reachability (object trajectory)
            [passed, waypointslist, timestamps] = self.CheckSE3TrajReachability\
            (rottraj, transtraj, [Rbeg, Rend], 
             vnear.config.qgrasps, vnear.config.qrobots)
            if not passed:
                print '    [ExtendFW] TRAPPED : SE(3) trajectory not reachable'
                status = TRAPPED
                continue

            # Now this trajectory is alright.
            print '    [ExtendFW] successful'
            newqrobots = [wp[-1][0:6] for wp in waypointslist] # waypoint is (q, qd)
            newconfig = CCConfig(newse3c, newqrobots, vnear.config.qgrasps, FW)
            newvertex = CCVertex(newconfig, FW)
            self._query.treestart.AddVertex(newvertex, vnear.index, rottraj, transtraj,
                                            waypointslist, timestamps)
            return status
        return status
            

    def ExtendBW(self, se3config):
        status = TRAPPED
        return status


    def Connect(self):
        if True:#(np.mod(self._query.iterations - 1, 2) == FW):
            # Treestart has just been extended
            return self.ConnectFW()
        else:
            # Treeend has just been extended
            return self.ConnectBW()


    def ConnectFW(self):
        """
        ConnectFW tries to connect the newly added vertex on treestart
        (vtest) to other vertices on treeend (vnear).
        """
        vtest = self._query.treestart.verticeslist[-1]
        nnindices = self.NearestNeighborIndices(vtest.config.se3config, BW)
        status = TRAPPED
        for index in nnindices:
            vnear = self._query.treeend[index]
            
            # quaternion
            qbeg = vtest.config.se3config.q
            qsbeg = vtest.config.se3config.qs
            
            qend = vnear.config.se3config.q
            qsend = vnear.config.se3config.qs
            
            # translation
            pbeg = vtest.config.se3config.p
            psbeg = vtest.config.se3config.ps

            pend = vnear.config.se3config.p
            psend = vnear.config.se3config.ps
            
            # Interpolate the object trajectory
            Rbeg = orpy.rotationMatrixFromQuat(qbeg)
            Rend = orpy.rotationMatrixFromQuat(qend)
            rottraj = Lie.InterpolateSO3(Rbeg,
                                         orpy.rotationMatrixFromQuat(qend),
                                         qsbeg, qsend, 
                                         self._query.interpolationdur)
            transtrajstring = SE3Utils.TrajString3rdDegree\
            (pbeg, pend, psend, psend, self._query.interpolationdur)
            transtraj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString\
            (transtrajstring)

            # Check collision (object trajectory)
            if not self.IsSE3TrajCollisionFree(rottraj, transtraj, Rbeg):
                print '    [ConenctFW] TRAPPED : SE(3) trajectory in collision'
                continue
            
            # Check reachability (object trajectory)
            [passed, waypointslist, timestamps] = self.CheckSE3TrajReachability\
            (rottraj, transtraj, [Rbeg, Rend], 
             vtest.config.qgrasps, vtest.config.qrobots)
            if not passed:
                print '    [ConnectFW] TRAPPED : SE(3) trajectory not reachable'
                continue

            # Check similarity of terminal IK solutions
            eps = 1e-3
            for i in xrange(2):
                print vnear.config.qrobots[i]
                print waypointslist[i][-1][0:6]
                passed = (RobotConfigDistance(vnear.config.qrobots[i],
                                              waypointslist[i][-1][0:6]) < eps)
                if not passed:
                    break
            if not passed:
                print '    [ConnectFW] TRAPPED : IK sol discrepancy (robot {0})'.\
                format(i)
                continue

            # Now the connection is successful
            self._query.treeend.verticeslist.append(vnear)
            self._query.connectingrotationaltraj = rottraj
            self._query.connectingtranslationaltraj = transtraj
            self._query.connectingrobotwps = waypointslist
            self._query.connectingtimestamps = timestamps
            status = REACHED
            return status
        return status        


    def ConnectBW(self):
        status = TRAPPED
        return status

    
    def IsSE3ConfigCollisionFree(self, se3config):
        if True:# with self.env:
            self.SetHomeConfigurations()
            self.object.SetTransform(se3config.T)
            
            result = not self.env.CheckCollision(self.object)

        return result

    
    def IsSE3TrajCollisionFree(self, rottraj, transtraj, Rbeg):
        T = np.eye(4) # dummy transformation matrix
        with self.env:
            self.SetHomeConfigurations()

            for s in  np.arange(0, transtraj.duration, self._query.discrtimestep):
                T[0:3, 0:3] = Lie.EvalRotation(Rbeg, rottraj, s)
                T[0:3, 3] = transtraj.Eval(s)

                self.object.SetTransform(T)
                collisionfree = not self.env.CheckCollision(self.object)
                if not collisionfree:
                    return False
                
        return True        

    
    def CheckSE3ConfigReachability(self, se3config, qgrasps, refsols):
        """
        CheckSE3ConfigReachability checks whether both robots can
        grasp the object (at se3config.T), using qgrasps, with IK
        solutions in the same classes with refsols.
        
        This function returns a list containing the status, and the
        solutions.
        """
        with self.env:
            self.object.SetTransform(se3config.T)
            
            self.SetHomeConfigurations()
            
            sols = []
            for (i, robot) in enumerate(self.robotslist):
                ibox = qgrasps[i][0]
                Tbox = self.object.GetLinks()[ibox].GetTransform()
                Tgripper = Utils.ComputeTGripper(Tbox, qgrasps[i],
                                                 self.objectdb.boxinfos[ibox].extents)
                solslist = self.manips[i].FindIKSolutions(Tgripper, IKCHECKCOLLISION)
                sol = self.FindClosestIKSolution(solslist, refsols[i])
                if sol is None:
                    return [False, []]
                sols.append(sol)

        return [True, sols]


    def CheckSE3TrajReachability(self, rottraj, transtraj, rotmatriceslist, 
                                 qgrasps, refsols):
        """
        CheckSE3TrajReachability checks whether the two robots can
        follow the se3 traj. This function returns status,
        waypointslist, and timestamps.
        """
        lietraj = Lie.LieTraj(rotmatriceslist, [rottraj])

        waypointslist = []
        for i in xrange(2):
            [passed, wp, ts] = self.ots[i].Plan(lietraj, transtraj, qgrasps[i], 
                                                q0=refsols[i],
                                                timestep=self._query.discrtimestep)
            if not passed:
                break
            waypointslist.append(wp)

        if not passed:
            return [False, [], []]
        
        else:
            return [True, waypointslist, ts]


    def FindClosestIKSolution(self, solutionslist, refsol):
        if len(solutionslist) == 0:
            return None
        distanceslist = [RobotConfigDistance(sol, refsol) for sol in solutionslist]
        indiceslist = np.argsort(distanceslist)
        return solutionslist[indiceslist[0]]
    

    def NearestNeighborIndices(self, se3config, treetype):
        """
        NearestNeighborIndices returns indices of self.nn nearest
        neighbors of se3config on the tree specified by treetype.
        """
        if (treetype == FW):
            tree = self._query.treestart
        else:
            tree = self._query.treeend
        nv = len(tree)
            
        distancelist = [SE3Distance(se3config, v.config.se3config) 
                        for v in tree.verticeslist]
        distanceheap = Heap.Heap(distancelist)
                
        if (self._query.nn == -1):
            # to consider all vertices in the tree as nearest neighbors
            nn = nv
        else:
            nn = min(self._query.nn, nv)
        nnindices = [distanceheap.ExtractMin()[0] for i in range(nn)]
        return nnindices


    def VisualizeCCMotion(self, lietraj, transtraj,
                          lwaypoints, rwaypoints, timestamps, timemult=1.0):
        assert(len(lwaypoints) == len(timestamps))
        assert(len(rwaypoints) == len(timestamps))

        deltat = timestamps[1] - timestamps[0]
        sleeptime = deltat * timemult
        M = np.eye(4)
        for (ql, qr, t) in zip(lwaypoints, rwaypoints, timestamps):
            M[0:3, 0:3] = lietraj.EvalRotation(t)
            M[0:3, 3] = transtraj.Eval(t)
            self.object.SetTransform(M)
            self.robotslist[0].SetActiveDOFValues(ql)
            self.robotslist[1].SetActiveDOFValues(qr)
            time.sleep(sleeptime)

            
    def SetHomeConfigurations(self):
        for robot in self.robotslist:
            robot.SetActiveDOFValues(HOME)


    def CCShortcut(self, cctrajectory, maxiter):
        
        printtime = False#True
        
        # Object's physical limits
        fmax = np.ones(3)
        taumax = np.ones(3)
        vmax = 10.*np.ones(6)

        # Shortcutting parameters
        integrationtimestep = 1e-2
        reparamtimestep = 1e-2
        passswitchpointnsteps = 5
        discrtimestep = 1e-2
        interpolationdur = 0.5
        minntimesteps = 10

        # Statistics
        ncollision = 0
        nnotreachable = 0
        nnotretimeable = 0
        nnotshorter = 0
        nsuccessful = 0

        originaldur = cctrajectory.lietraj.duration
        dur = originaldur

        newlietraj = cctrajectory.lietraj
        newtranstraj = cctrajectory.transtraj
        newtimestamps = cctrajectory.timestamps[:]
        newlwaypoints = cctrajectory.waypointsdict[0][:]
        newrwaypoints = cctrajectory.waypointsdict[1][:]

        # Create an accumulated distance list
        accumulateddist = GenerateAccumulatedSE3DistancesList\
        (newlietraj, newtranstraj, self._query.discrtimestep)        

        for i in xrange(maxiter):
            if (dur < discrtimestep):
                print Colorize\
                ('[CCShortcut] trajectory duration < discrtimestep', 'yellow')
                break
            
            print '[CCShortcut] ' + Colorize('iteration {0}'.format(i + 1), 'blue')

            # Sample two time instants
            timestampindices = range(len(newtimestamps))
            tindex0 = RNG.choice(timestampindices[:-minntimesteps])
            tindex1 = RNG.choice(timestampindices[tindex0 + minntimesteps:])
            t0 = newtimestamps[tindex0]
            t1 = newtimestamps[tindex1]
            print 'tindex0 = {0}, t0 = {1}'.format(tindex0, t0)
            print 'tindex1 = {0}, t1 = {1}'.format(tindex1, t1)

            # Interpolate a new SE(3) trajectory segment
            R0 = newlietraj.EvalRotation(t0)
            R1 = newlietraj.EvalRotation(t1)
            rottraj = Lie.InterpolateSO3(R0, R1,
                                         newlietraj.EvalOmega(t0),
                                         newlietraj.EvalOmega(t1),
                                         interpolationdur)

            transtrajstring = SE3Utils.TrajString3rdDegree\
            (newtranstraj.Eval(t0), newtranstraj.Eval(t1),
             newtranstraj.Evald(t0), newtranstraj.Evald(t1), interpolationdur)
            transtraj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString\
            (transtrajstring)


            # rotmatriceslist = [R0, R1]
            # rottrajslist = [rottraj]
            # lietraj = Lie.LieTraj(rotmatriceslist, rottrajslist)
            # print 'check lietraj'
            # print 'newlietraj:', newlietraj.EvalRotation(t0)
            # print 'lietraj:', lietraj.EvalRotation(0)
            # print 'newlietraj:', newlietraj.EvalRotation(t1)
            # print 'lietraj:', lietraj.EvalRotation(lietraj.duration)

            # print 'check transtraj'
            # print 'newtranstraj', newtranstraj.Eval(t0)
            # print 'transtraj', transtraj.Eval(0)
            # print 'newtranstraj', newtranstraj.Eval(t1)
            # print 'transtraj', transtraj.Eval(transtraj.duration)


            # raw_input('[CCShortcut] visualize the interpolated trajectory')
            
            # Tdummy = np.eye(4)
            # for t in np.arange(0, interpolationdur + 1e-3, discrtimestep):
            #     Tdummy[0:3, 0:3] = lietraj.EvalRotation(t)
            #     Tdummy[0:3, 3] = transtraj.Eval(t)
            #     self.object.SetTransform(Tdummy)
            #     time.sleep(0.01)

            # Check collision (object trajectory)
            ts_coll = time.time()
            if not self.IsSE3TrajCollisionFree(rottraj, transtraj, R0):
                te_coll = time.time()
                if printtime:
                    print \
                    "checking collision (failed) {0} sec.".format(te_coll - ts_coll)
                    
                ncollision += 1
                continue
            te_coll = time.time()
            if printtime:
                print "checking collision (passed) {0} sec.".format(te_coll - ts_coll)

            # Check reachability (object trajectory)
            ts_reach = time.time()
            [passed, waypointslist, timestamps] = self.CheckSE3TrajReachability\
            (rottraj, transtraj, [R0, R1], cctrajectory.qgrasps, 
             [newlwaypoints[tindex0], newrwaypoints[tindex1]])
            te_reach = time.time()
            if printtime:
                print "checking reachability: {0} sec.".format(te_reach - ts_reach)

            if not passed:
                nnotreachable += 1
                continue

            # Check SE(3) trajectory length
            
            accumulateddist = GenerateAccumulatedSE3DistancesList\
            (newlietraj, newtranstraj, self._query.discrtimestep)
            print 'accu', len(accumulateddist), len(newtimestamps)

            rotmatriceslist = [R0, R1]
            rottrajslist = [rottraj]
            lietraj = Lie.LieTraj(rotmatriceslist, rottrajslist)
            newtrajaccumulateddist = GenerateAccumulatedSE3DistancesList\
            (lietraj, transtraj, self._query.discrtimestep)

            # print "newdist = {0}".format(newtrajaccumulateddist[-1])
            # print "olddist = {0}".format(accumulateddist[tindex1] - 
            #                              accumulateddist[tindex0])
            if (newtrajaccumulateddist[-1] >= 
                accumulateddist[tindex1] - accumulateddist[tindex0]):
                nnotshorter += 1
                continue

            # Now the new trajectory passes all tests
            # Replace all the old trajectory segments with the new ones

            # raw_input('old traj')
            # Tdummy = np.eye(4)
            # for t in np.arange(0, newtimestamps[-1] + 1e-3, discrtimestep):
            #     Tdummy[0:3, 0:3] = newlietraj.EvalRotation(t)
            #     Tdummy[0:3, 3] = newtranstraj.Eval(t)
            #     self.object.SetTransform(Tdummy)
            #     time.sleep(0.01)

            newlietraj = SE3Utils.ReplaceTrajectorySegment\
            (newlietraj, lietraj.trajlist[0], t0, t1)
            
            newtranstraj = SE3Utils.ReplaceTransTrajectorySegment\
            (newtranstraj, transtraj, t0, t1)

            firsttimestampchunk = newtimestamps[:tindex0 + 1]
            lasttimestampchunkoffset = newtimestamps[tindex1]
            lasttimestampchunk = [t - lasttimestampchunkoffset 
                                  for t in newtimestamps[tindex1:]]
            # print newtimestamps
            newtimestamps = MergeTimeStampsList([firsttimestampchunk,
                                                 timestamps,
                                                 lasttimestampchunk])

            # raw_input('new traj')
            # Tdummy = np.eye(4)
            # for t in np.arange(0, newtimestamps[-1] + 1e-3, discrtimestep):
            #     Tdummy[0:3, 0:3] = newlietraj.EvalRotation(t)
            #     Tdummy[0:3, 3] = newtranstraj.Eval(t)
            #     self.object.SetTransform(Tdummy)
            #     time.sleep(0.01)

            # print newtimestamps
            # print

            # print 'check lwayponits'
            # print 'newlwaypoints t0:', newlwaypoints[tindex0]
            # print 'waypoints:', waypointslist[0][0]
            # print 'newlwaypoints t1:', newlwaypoints[tindex1]
            # print 'waypoints:', waypointslist[0][-1]
            newlwaypoints = MergeWaypointsList([newlwaypoints[:tindex0 + 1],
                                                waypointslist[0],
                                                newlwaypoints[tindex1:]])
            
            # print 'check rwayponits'
            # print 'newrwaypoints t0:', newrwaypoints[tindex0]
            # print 'waypoints:', waypointslist[1][0]
            # print 'newrwaypoints t1:', newrwaypoints[tindex1]
            # print 'waypoints:', waypointslist[1][-1]
            newrwaypoints = MergeWaypointsList([newrwaypoints[:tindex0 + 1],
                                                waypointslist[1],
                                                newrwaypoints[tindex1:]])
            
            # raw_input('visualizing the traj')
            # self.VisualizeCCMotion(lietraj, transtraj, 
            #                        waypointslist[0], waypointslist[1], timestamps,
            #                        timemult=3.0)
            print Colorize('\tsuccessful', 'green')
            nsuccessful += 1

        print "nsuccessful = {0}".format(nsuccessful)
        print "ncollision = {0}".format(ncollision)
        print "nnotshorter = {0}".format(nnotshorter)
        print "nnotreachable = {0}".format(nnotreachable)
            
        return CCTrajectory(newlietraj, newtranstraj, newlwaypoints, newrwaypoints, 
                            newtimestamps, cctrajectory.qgrasps)
        

################################################################################
#                               Object Tracker
################################################################################
### THERE IS STILL NO COLLISION DETECTION IMPLEMENTED IN OBJECTTRACKER
class ObjectTracker(object):
    
    def __init__(self, robot, manipulatorname, mobj, active_dofs=6):
        self.robot = robot
        self.manip = self.robot.SetActiveManipulator(manipulatorname)
        self.object = mobj
        self.active_dofs = active_dofs

        self._vmax = self.robot.GetDOFVelocityLimits()[0:self.active_dofs]
        self._jmax = self.robot.GetDOFLimits()[1][0:self.active_dofs]
        self._maxiter = 10000
        self._weight = 10.
        self._tol = 0.5e-3
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
            A list containing vectors q
        timestamps : list
            A list containing timestamps of waypoints

        NB: qd in this case might not be meaningful. So we return to
        the user only q.
        """
        M = np.eye(4) # dummy variable for the object's transformation
        M[0:3, 0:3] = lietraj.EvalRotation(0)
        M[0:3, 3] = transtraj.Eval(0)
        self.object.SetTransform(M)
        graspedlink = self.object.GetLinks()[qgrasp[0]]
        extents = graspedlink.GetGeometries()[0].GetBoxExtents()
        dur = lietraj.duration
        Trel = np.dot(np.linalg.inv(self.object.GetTransform()),
                      graspedlink.GetTransform())
        
        # Trajectory tracking loop
        waypoints = [] # a list containing waypoints along the way
        timestamps = []
        
        T = Utils.ComputeTGripper(np.dot(M, Trel), qgrasp, extents)

        # Compute an initial solution if not provided
        if q0 is None:
            qprev = self.manip.FindIKSolution\
            (T, orpy.IkFilterOptions.CheckEnvCollisions)
            if qprev is None:
                print '[ObjectTrackingPlanner::Plan]'
                print Colorize('No IK solution at the initial configuration', 'red')
                return [False, waypoints, timestamps]
        else:
            qprev = q0
        if qd0 is None:
            qdprev = np.zeros(self.robot.GetActiveDOF())
        else:
            qdprev = qd0
            
        qnext, qdnext = self.ComputeIK(T, qprev, qdprev)
        if qnext is None and qdnext is None:
            return [False, waypoints, timestamps]
        
        waypoints.append(qnext)
        timestamps.append(0.)
        qprev = qnext
        qdprev = qdnext
        
        for t in np.arange(timestep, dur, timestep):
            M[0:3, 0:3] = lietraj.EvalRotation(t)
            M[0:3, 3] = transtraj.Eval(t)
            
            T = Utils.ComputeTGripper(np.dot(M, Trel), qgrasp, extents)
            
            qnext, qdnext = self.ComputeIK(T, qprev, qdprev)
            if qnext is None and qdnext is None:
                return [False, waypoints, timestamps]
            
            waypoints.append(qnext)
            timestamps.append(t)
            
            qprev = qnext
            qdprev = qdnext

        # Compute a solution at the end of the trajectory
        M[0:3, 0:3] = lietraj.EvalRotation(dur)
        M[0:3, 3] = transtraj.Eval(dur)

        T = Utils.ComputeTGripper(np.dot(M, Trel), qgrasp, extents)
        qnext, qdnext = self.ComputeIK(T, qprev, qdprev)
        if qnext is None and qdnext is None:
            return [False, waypoints, timestamps]
        waypoints.append(qnext)
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
        i = 0 # iteration counter
        reached = False
        while i < self._maxiter:
            prev_obj = cur_obj
            cur_obj = self.ComputeObjective(targetpose, q, qd)
            if abs(cur_obj - prev_obj) < self._tol and cur_obj < self._tol:
                # Local minimum reached
                reached = True
                break

            i += 1
            qd = self.ComputeVelocity(targetpose, q, qd) # KEY
            
            qd = np.maximum(np.minimum(qd, self._vmax), -self._vmax)
            q = q + (qd * self._dt)
            q = np.maximum(np.minimum(q, self._jmax), -self._jmax)
            
        if reached == False:
            print Colorize\
            ('[ComputeIK] max iteration ({0}) exceeded'.format(self._maxiter), 'red')
            return None, None
        
        return q, qd

    
    def ComputeObjective(self, targetpose, q, qd):
        error = self.ComputeError(targetpose, q, qd)
        obj = self._weight * np.dot(error, error)
        return obj


    def ComputeError(self, targetpose, q, qd):
        with self.robot:
            self.robot.SetActiveDOFValues(q)
            currentpose = self.manip.GetTransformPose()
        if currentpose[0] < 0:
            currentpose[0:4] *= -1.
        error = targetpose - currentpose
        
        return error


    def ComputeVelocity(self, targetpose, q, qd):
        raise NotImplementedError
    

class ObjectTracker1(ObjectTracker):

    def __init__(self, robot, manipulatorname, mobj, active_dofs=6):
        super(ObjectTracker1, self).__init__(robot, manipulatorname, mobj, active_dofs)


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

        weight = 10.0
        try:
            new_qd = weight * np.linalg.solve(J, (targetpose - currentpose))
        except:
            new_qd = weight * np.dot(np.linalg.pinv(J), (targetpose - currentpose))
        return new_qd


################################################################################
#                             Planning Utilities
################################################################################
def SE3Distance(se3config1, se3config2):
    return SE3Utils.SE3Distance(se3config1.T, se3config2.T, 1.0/np.pi, 1.0)
        
        
def RobotConfigDistance(q1, q2, choice=0):
    """
    RobotConfigDistance returns a distance between two robot
    configurations (joint values) according to the specified distance
    metric.
    
    choice
    0 : L2 norm squared
    1 : L2 norm
    2 : L1 norm
    """
    if choice == 0:
        # norm-2 squared
        return sum((q1 - q2)**2)
    
    elif choice == 1:
        # norm-2
        return np.sqrt(sum((q1 - q2)**2))

    elif choice == 2:
        # norm-1
        return sum(abs(q1 - q2))

    else:
        raise ValueError('unknown choice {0}'.format(choice))


def MergeTimeStampsList(timestampslist):
    """
    MergeTimeStampsList merges timestamps lists of the form [0, 1, 2,
    ..., T] together.
    """
    newtimestamps = None
    for T in timestampslist:
        if newtimestamps is None:
            newtimestamps = []
            offset = 0
        else:
            T.pop(0)
            offset = newtimestamps[-1]

        newT = [t + offset for t in T]
        newtimestamps = newtimestamps + newT
    return newtimestamps


def MergeWaypointsList(waypointslist, eps=1e-3):
    newwaypoints = None
    for W in waypointslist:
        if newwaypoints is None:
            newwaypoints = []
        else:
            # Check soundness
            try:
                assert(RobotConfigDistance(W[0][0:6], newwaypoints[-1][0:6]) < eps)
            except:
                print 'W', W[0]
                print 'new', newwaypoints[-1]
                assert(False)
            W.pop(0)

        newwaypoints = newwaypoints + W
        
    return newwaypoints


def GenerateAccumulatedSE3DistancesList(lietraj, transtraj, timestep):
    deltadistlist = [0.0]
    Tprev = np.eye(4)
    Tprev[0:3, 0:3] = lietraj.EvalRotation(0)
    Tprev[0:3, 3] = transtraj.Eval(0)
    Tcur = np.eye(4)
        
    timestamps = np.arange(timestep, lietraj.duration, timestep)
    timestamps = np.append(timestamps, lietraj.duration)
    
    for t in timestamps:
        Tcur[0:3, 0:3] = lietraj.EvalRotation(t)
        Tcur[0:3, 3] = transtraj.Eval(t)
        
        deltadist = SE3Utils.SE3Distance(Tprev, Tcur)
        deltadistlist.append(deltadist)
        
        Tprev = np.array(Tcur)

    accumulateddist = [sum(deltadistlist[0:i]) for i in xrange(len(deltadistlist))]
    return accumulateddist
        
