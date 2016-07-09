import openravepy as orpy
import numpy as np
import time
import random
import copy

# For plotting manipulation graphs
from matplotlib import pyplot as plt
from pylab import ion
ion()

import BaseBimanualManipulationPlanner as bbmp
from BaseBimanualManipulationPlanner import (FW, BW, REACHED, ADVANCED, TRAPPED,
                                             TRANSIT, TRANSFER, CG1CG2CP, CP,
                                             HOME)
import BimanualGPG as bgpg

from pymanip.utils import Utils
from pymanip.utils.Utils import Colorize

_RNG = random.SystemRandom()


################################################################################
#            Bimanual Grasp-Placement-Graph Manipulation Planner
################################################################################
class BimanualGPGMP(bbmp.BaseBimanualManipulationPlanner):

    def __init__(self, robots, manipulatornames, mobj):
        print Colorize('Initializing Bimanual GPG Manipulation Planner . . .', 'yellow')
        super(BimanualGPGMP, self).__init__(robots, manipulatornames, mobj)

        print Colorize('Gathering information of vertices of GPG', 'yellow')
        ts_gath = time.time()
        self.graphvertices = []
        for isurface in xrange(len(self.S) - 1):
            for ibox in xrange(len(self.object.GetLinks())):
                for appdir in self.boxinfos[ibox].possibleapproachingdir[isurface]:
                    vnew = (isurface, appdir + 6*ibox)
                    self.graphvertices.append(vnew)
        te_gath = time.time()
        print Colorize('  Created a set of GPG vertices in {0} sec.'.\
                           format(te_gath - ts_gath), 'yellow')

        # Assume identical Graphs
        print Colorize('Constructing Grasp-Placement Graph', 'yellow')
        ts_con = time.time()
        self.BGPG_original = bgpg.BimanualGraspPlacementGraph(self.graphvertices)
        te_con = time.time()
        print Colorize('  Construction time: {0} sec.'.\
                           format(te_con - ts_con), 'yellow')

        # Threshold value for removal of infeasible edges
        self._threshold = 15
        # Number of trials to try SampleCP & SampleCG
        self._ntrialsCP = 20
        self._ntrialsCG = 10
        
        print Colorize('Initialization completed', 'yellow')


    def SetSampleParameters(self, xobjlim, yobjlim, thetaobjlim, zoffset):
        self.xobjlim = xobjlim
        self.yobjlim = yobjlim
        self.thetaobjlim = thetaobjlim
        self.zoffset = zoffset # height of the table
        self._setsampleparameters = True


    def InitQuery(self, cstart, cgoal):
        self.vstart = self.Vertex(cstart.Clone(), FW)
        self.vgoal = self.Vertex(cgoal.Clone(), BW)
        # If all paths of length k are infeasible, then try paths of
        # length k + self._increment
        if (self.vstart.id.count(None) == 0) or (self.vgoal.id.count(None) == 0):
            # Either start or goal (or both) is in GP
            self._increment = 1
        else:
            self._increment = 2
            
        self._hasquery = True
        self._hassolution = False

        # A graph dictionary to be modified according to this query
        self.BGPG = copy.deepcopy(self.BGPG_original)

        # Shortest path length before edge elimination
        self._p = self.BGPG.FindShortestPathLength(self.vstart.id[1::], self.vgoal.id[1::])

        self.EliminateInfeasibleTerminalGrasps()


    def EliminateInfeasibleTerminalGrasps(self):
        """
        For each pair of grasp classes, we conduct Ntrials tests. If
        the probability of success is less than 0.5, the pair of grasp
        classes is eliminated.
        """
        ts = time.time()
        
        Ntrials = 20
        P = 0.5

        self._nremovededges = 0

        # Test with p_start (the initial placement)
        start_id = self.vstart.id[1:]
        if start_id.count(None) == 2:
            # Start configuration is in CP
            Tstart = self.vstart.config.tobj

            for key in self.BGPG.graphdict.keys():
                if not (key[0] == start_id[0]):
                    # Different placement class
                    continue

                if key.count(None) == 2:
                    continue

                isurface = key[0]
                approachingdirs = [key[1], key[2]]
                Npassed = 0
                for i in xrange(Ntrials):
                    # This loop is similar to SampleCG
                    sols = []
                    self.SetHOMEConfigurations()
                    self.object.SetTransform(Tstart)

                    for robotindex in xrange(self.nrobots):
                        ibox = int(approachingdirs[robotindex])/6
                        realapproachingdir = np.mod(approachingdirs[robotindex], 6)
                        boxinfo = self.boxinfos[ibox]
                        graspedlink = self.object.GetLinks()[ibox]

                        # Sample a sliding direction
                        slidingdir = _RNG.choice(boxinfo.possibleslidingdir[realapproachingdir])
                        # Resample a sliding direction if necessary
                        while (isurface, realapproachingdir, slidingdir) not in boxinfo.intervals:
                            slidingdir = \
                            _RNG.choice(boxinfo.possibleslidingdir[realapproachingdir])

                        # Sample a value for delta (where to grasp along the sliding direction)
                        delta = WeightedChoice2\
                        (boxinfo.intervals[isurface, realapproachingdir, slidingdir])

                        # Assign qgrasp
                        qgrasp = [ibox, approachingdirs[robotindex], slidingdir, delta]

                        # Compute Tgripper
                        Tgripper = Utils.ComputeTGripper(graspedlink.GetTransform(),
                                                         qgrasp, boxinfo.extents, unitscale=False)

                        # Compute an IK solution for robot robotindex
                        sol = self.manips[robotindex].FindIKSolution\
                        (Tgripper, orpy.IkFilterOptions.CheckEnvCollisions)
                        if sol is None:
                            break

                        sols.append(sol)

                        # Set the robot at the found configuration
                        self.robots[robotindex].SetActiveDOFValues(sol)

                    # Grasp checking takes some time so we do it after both IK
                    # solutions are found.
                    if len(sols) == self.nrobots:
                        Npassed += 1
                        # if self.CheckGrasp(0, sols[0], Tstart):
                        #     if self.CheckGrasp(1, sols[1], Tstart):
                        #         Npassed += 1

                if float(Npassed)/float(Ntrials) < P:
                    self._nremovededges += 1
                    self.BGPG.graphdict[start_id].remove(key)

        # Test with p_goal (the goal placement)
        goal_id = self.vgoal.id[1:]
        if goal_id.count(None) == 2:
            # Goal configuration is in CP
            Tgoal = self.vgoal.config.tobj

            for key in self.BGPG.graphdict.keys():
                if not (key[0] == goal_id[0]):
                    # Different placement class
                    continue

                if key.count(None) == 2:
                    continue

                isurface = key[0]
                approachingdirs = [key[1], key[2]]
                Npassed = 0
                for i in xrange(Ntrials):
                    # This loop is similar to SampleCG
                    sols = []
                    self.SetHOMEConfigurations()
                    self.object.SetTransform(Tgoal)

                    for robotindex in xrange(self.nrobots):
                        ibox = int(approachingdirs[robotindex])/6
                        realapproachingdir = np.mod(approachingdirs[robotindex], 6)
                        boxinfo = self.boxinfos[ibox]
                        graspedlink = self.object.GetLinks()[ibox]

                        # Sample a sliding direction
                        slidingdir = _RNG.choice(boxinfo.possibleslidingdir[realapproachingdir])
                        # Resample a sliding direction if necessary
                        while (isurface, realapproachingdir, slidingdir) not in boxinfo.intervals:
                            slidingdir = \
                            _RNG.choice(boxinfo.possibleslidingdir[realapproachingdir])

                        # Sample a value for delta (where to grasp along the sliding direction)
                        delta = WeightedChoice2\
                        (boxinfo.intervals[isurface, realapproachingdir, slidingdir])

                        # Assign qgrasp
                        qgrasp = [ibox, approachingdirs[robotindex], slidingdir, delta]

                        # Compute Tgripper
                        Tgripper = Utils.ComputeTGripper(graspedlink.GetTransform(),
                                                         qgrasp, boxinfo.extents, unitscale=False)

                        # Compute an IK solution for robot robotindex
                        sol = self.manips[robotindex].FindIKSolution\
                        (Tgripper, orpy.IkFilterOptions.CheckEnvCollisions)
                        if sol is None:
                            break

                        sols.append(sol)

                        # Set the robot at the found configuration
                        self.robots[robotindex].SetActiveDOFValues(sol)

                    # Grasp checking takes some time so we do it after both IK
                    # solutions are found.
                    if len(sols) == self.nrobots:
                        Npassed += 1
                        # if self.CheckGrasp(0, sols[0], Tgoal):
                        #     if self.CheckGrasp(1, sols[1], Tgoal):
                        #         Npassed += 1

                if float(Npassed)/float(Ntrials) < P:
                    self._nremovededges += 1
                    self.BGPG.graphdict[key].remove(goal_id)
            
        te = time.time()
        print Colorize('Eliminating {0} infeasible edges in {1} sec.'.\
                           format(self._nremovededges, te - ts), 'yellow')
        self.SetHOMEConfigurations()


    def Plan(self, timeout):
        if not self._hasquery:
            print Colorize('No query for the planner yet', 'red')
            return False
        if not self._setsampleparameters:
            print Colorize('Sample parameters have not been set yet', 'red')
            return False
        
        # Shortest path length (from the BGP Graph)
        # self._p = self.BGPG.FindShortestPathLength(self.vstart.id[1::], self.vgoal.id[1::])
        
        n = 0 # dummy variable keeping track of current path length
        self.timerecord = [] # keeps records of running time for each path length
        while (self._runningtime < timeout) and (not self._hassolution):
            ts_loop = time.time()
            
            self._k = self._p + self._increment*n # current shortest possible path length
            n += 1
            
            # Create new trees for planning
            vstart = self.vstart.Clone()
            vstart.level = 0
            vgoal = self.vgoal.Clone()
            vgoal.level = self._k
            vgoal.id = (vgoal.level, ) + vgoal.id[1::]
            self.treestart = self.Tree(vstart, FW)
            self.treeend = self.Tree(vgoal, BW)

            ts_graph = time.time() # start graph processing
            if self._print:
                print Colorize('Extracting paths of length {0}. . .'.format(self._k), 
                               'yellow')
            # pathslist is a list of possible paths of length self._k
            self._pathslist = self.BGPG.FindPathsOfLengthK2(self.vstart.id[1::],
                                                            self.vgoal.id[1::],
                                                            self._k)

            if len(self._pathslist) < 1:
                continue

            if self._print:
                print Colorize('Creating direction graphs. . .', 'yellow')
            # We use Q for a directed graph
            self.QFW = bgpg.CreateFWDirectedGraphFromPathsList(self._pathslist)
            self.QBW = bgpg.CreateBWDirectedGraphFromPathsList(self._pathslist)

            # We need to find out what the first trajtype is (then we
            # know the rest). For simplicity, we assume that either
            # the start configuration or the goal configuration is in
            # P. This implies there is no ambiguity in the first
            # trajtype.
            self.firsttrajtype = None
            for vid in self.QFW[vstart.id]:
                if ((vstart.id[1] == vid[1]) and (not (vstart.id[2] == vid[2]))):
                    self.firsttrajtype = TRANSIT
                    break
                elif ((vstart.id[1] == vid[1]) and (not (vstart.id[3] == vid[3]))):
                    self.firsttrajtype = TRANSIT
                    break
                elif ((vstart.id[2] == vid[2]) and (not (vstart.id[1] == vid[1]))):
                    self.firsttrajtype = TRANSFER
                    break
                else:
                    # Not enough information
                    continue
            assert(self.firsttrajtype is not None)

            if self._print:
                print Colorize('Creating an edge database for storing statistics. . .',
                               'yellow')
            # Each entry stores the number of IK failure. High number
            # of IK failure can probably reflect the actual kinematic
            # infeasibility.
            self.ED = dict() # Edge database
            for key in self.QFW.graphdict:
                for val in self.QFW.graphdict[key]:
                    edge = key + val
                    if edge not in self.ED:
                        self.ED[edge] = 0

            te_graph = time.time() # end graph processing
            if self._print:
                print Colorize('  Graph processing time : {0} sec.'.\
                                   format(te_graph - ts_graph), 'yellow')
            
            # Compute weights for start and goal vertices. A vertex
            # with more weight is more likely to be sampled. This
            # weight is then roughly a bias thatt we put in the
            # exploration.
            self._weighteddist = [[], []]
            self._weighteddist[FW].append((self.treestart[0].index, 
                                           ComputeWeight(self.treestart[0].level)))
            self._weighteddist[BW].append((self.treeend[0].index, 
                                           ComputeWeight(self.treeend[0].level)))

            # Run the planner until all possible paths (in
            # self._pathslist) are declared infeasible.
            self._hassolution = self._Run(timeout - self._runningtime)
            te_loop = time.time()
            self.timerecord.append(te_loop - ts_loop)

        return self._hassolution
            

    def _Run(self, timeout):
        if self._hassolution:
            print 'The planner has already found a solution.'
            return True

        t = 0.0
        it = 0

        while (t < timeout):
            it += 1
            if self._print:
                print Colorize('({0}) iteration : {1}'.format(self._k, it), 'blue')
            
            ts = time.time()
                
            # Eliminate edges that are declared infeasible (according
            # to self._threshold)
            modified = False # True if some edges are removed
            infeasiblekeys = []
            infeasiblepaths = []
            for key in self.ED:
                if self.ED[key] > self._threshold:
                    l1 = key[0]
                    v1 = key[1:4]
                    l2 = key[4]
                    v2 = key[5:8]
                    for ipath in xrange(len(self._pathslist)):
                        path = self._pathslist[ipath]
                        if (path[l1] == v1) and (path[l2] == v2):
                            infeasiblepaths.append(ipath)
                            if key not in infeasiblekeys:
                                if self._print:
                                    print Colorize('  Eliminating edge {0}'.format(key),
                                                   'magenta')
                                infeasiblekeys.append(key)
                            modified = True

            # If some edges are removed, we also need to remove paths
            # which contain those edges.
            if modified:
                oldpathslist = self._pathslist
                self._pathslist = []
                for ipath in xrange(len(oldpathslist)):
                    if ipath not in infeasiblepaths:
                        self._pathslist.append(oldpathslist[ipath])
                if len(self._pathslist) == 0:
                    # All paths have been eliminated
                    break
                else:
                    for key in infeasiblekeys:
                        self.ED.pop(key)
                    self.QFW = bgpg.CreateFWDirectedGraphFromPathsList(self._pathslist)
                    self.QBW = bgpg.CreateBWDirectedGraphFromPathsList(self._pathslist)

            treedirection = FW#np.mod(it - 1, 2)
            # Sample a vertex on the tree 'treedirection' to be
            # extended from
            index = self.SampleTree(treedirection)

            if treedirection == FW:
                # Forward extension
                status = self.ExtendFWFrom(index)
                if not (status == TRAPPED):
                    if (status == REACHED) or (self.ConnectFW() == REACHED):
                        te = time.time()
                        t += te - ts
                        self._runningtime += t
                        self._iterations += it
                        self._hassolution = True
                        if self._print:
                            print Colorize('Path found', 'green')
                            print Colorize('    Total number of iterations : {0}'.\
                                               format(self._iterations), 'green')
                            print Colorize('    Total running time : {0} sec.'.\
                                               format(self._runningtime), 'green')
                            
                        return True
            else:
                # Backward Extension
                status = self.ExtendBWFrom(index)
                if not (status == TRAPPED):
                    if (status == REACHED) or (self.ConnectBW() == REACHED):
                        te = time.time()
                        t += te - ts
                        self._runningtime += t
                        self._iterations += it
                        self._hassolution = True
                        if self._print:
                            print Colorize('Path found', 'green')
                            print Colorize('    Total number of iterations : {0}'.\
                                               format(self._iterations),
                                           'green')
                            print Colorize('    Total running time : {0} sec.'.\
                                               format(self._runningtime),
                                           'green')
                            
                        return True
                    
            te = time.time()
            t += te - ts
            
        # End while loop
        if len(self._pathslist) == 0:
            print Colorize('All possible paths of length {0} are declared infeasible'.\
                               format(self._k), 'red')
        else:
            print Colorize('Allotted time ({0} sec.) is exhausted after {1} iterations'.\
                               format(timeout, it), 'red')
        self._runningtime += t
        self._iterations += it
        return False


    def SampleTree(self, treedirection):
        return WeightedChoice(self._weighteddist[treedirection])

    
    def ExtendFWFrom(self, vindex):
        if self._print:
            print '  [ExtendFWFrom] vindex = {0}'.format(vindex)
        status = TRAPPED

        vnear = self.treestart[vindex]
        prevtype = vnear.trajtype

        # Look into the dictionary for possible extensions
        try:
            possibleextensions = self.QFW.graphdict[vnear.id]
        except KeyError:
            # Possible paths have been removed (due to kinematic infeasibility)
            oldweighteddist = self._weighteddist[FW]
            if self._print:
                print '    Sampled vertex is not on any feasible path'
                print '    Removeing vindex = {0} from treestart'.format(vindex)
            self._weighteddist[FW] = [d for d in oldweighteddist if d[0] != vindex]
            return status

        # Sample an id of a vertex to extend to
        vnext_id = _RNG.choice(possibleextensions)
        # Unpack the id
        [nextlevel, isurface, approachingdir1, approachingdir2] = vnext_id

        # The cases when vnear is either 1 step or 2 steps away from
        # the goal have to be handled separately.
        
        # Check whether vnear is one step away from vgoal
        onesteptogoal = (vnext_id == self.treeend[0].id)
        # Check whether vnear is two steps away from vgoal
        twostepstogoal = False
        nexttwosteps = self.QFW.graphdict[vnext_id] # set of two-step-away vertices
        if (len(nexttwosteps) == 1) and (nexttwosteps[0] == self.treeend[0].id):
            twostepstogoal = True
            
        if self._print:
            print '  [ExtendFWFrom] vnear = {0}; vnext = {1}'.\
            format(vnear.id, vnext_id)

        # Check whether this upcoming edge is TRANSIT or TRANSFER
        # If any of these condition is satisfied, the next trajtyps is TRANSIT
        cond1 = (prevtype == TRANSFER)
        cond2 = ((vnear.id[1] == vnext_id[1]) and (not (vnear.id[2] == vnext_id[2])))
        cond3 = ((vnear.id[1] == vnext_id[1]) and (not (vnear.id[3] == vnext_id[3])))
        cond4 = ((self.firsttrajtype == TRANSIT) and (np.mod(vnear.level, 2) == 0))
        cond5 = ((self.firsttrajtype == TRANSFER) and (np.mod(vnear.level, 2) == 1))

        if cond1 or cond2 or cond3 or cond4 or cond5:
            if self._print:
                print '  TRANSIT extension'
            # TRANSIT
            if onesteptogoal:
                # TODO: code for planning TRANSIT trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]
                if not (plannerstatus == 1):
                    return status

                # Now we have reached the goal.
                status = REACHED

                # Stack the vertex vindex on top of treestart
                self.treestart.verticeslist.append(self.treestart[vindex])
                # Stack the vertex goal on top of treeend
                self.treeend.verticeslist.append(self.treeend[0])

                self.connectingtrajectorystrings = trajectorystrings
                self.connectingtrajectorytype = TRANSIT
                return status

            if twostepstogoal:
                # We need to transit to the same grasp(s) as in vertex goal
                qgrasps = self.treeend[0].config.qgrasps
                Tobj = vnear.config.tobj
                qrefs = self.treeend[0].config.qrobots
                sols = []
                for robotindex in xrange(self.nrobots):
                    sol = self.ComputeGraspConfiguration\
                    (robotindex, qgrasps[robotindex], Tobj, 
                     qrobot_ref=qrefs[robotindex])
                    if sol is None:
                        if self._print:
                            print '    No IK solution for robot {0}'.format(robotindex)
                        self.ED[vnear.id + vnext_id] += 1
                        return status
                    sols.append(sol)

            else:
                # Usual TRANSIT extension
                # Sample new grasps
                approachingdirs = [approachingdir1, approachingdir2]
                [passed, sols, qgrasps] = self.SampleCG(vnear.config.tobj, isurface,
                                                        approachingdirs)
                if not passed:
                    # Infeasible grasp is probably because it is
                    # kinematically unreachable
                    self.ED[vnear.id + vnext_id] += 1
                    return status

            cnew = self.Config(sols, vnear.config.tobj, qgrasps, CG1CG2CP, isurface)
            vnew = self.Vertex(cnew, FW, level=nextlevel)
            # TODO: code for planning TRANSIT trajectories
            [plannerstatus, trajectorystrings] = [True, ['', '']]
            
            if not (plannerstatus == 1):
                return status

            # Now successfully extended
            self.treestart.AddVertex(vindex, vnew, trajectorystrings, TRANSIT)
            self._weighteddist[FW].append((self.treestart[-1].index, 
                                           ComputeWeight(self.treestart[-1].level)))
            status = ADVANCED
            return status

        else:
            if self._print:
                print '  TRANSFER extension'
            # TRANSFER
            if onesteptogoal:
                # TODO: code for planning TRANSFER trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]
                if not (plannerstatus == 1):
                    return status
                # Now we have reached the goal. 
                status = REACHED
                
                # Stack the vertex vindex on top of treestart
                self.treestart.verticeslist.append(self.treestart[vindex])
                # Stack the vertex goal on top of treeend
                self.treeend.verticeslist.append(self.treeend[0])

                self.connectingtrajectorystrings = trajectorystrings
                self.connectingtrajectorytype = TRANSFER
                return status

            if twostepstogoal:
                # We need to transfer the object to the same Tobj as in vertex goal
                Tobj = self.treeend[0].config.tobj
                qgrasps = vnear.config.qgrasps
                qrefs = vnear.config.qrobots
                sols = []
                for robotindex in xrange(self.nrobots):
                    sol = self.ComputeGraspConfiguration\
                    (robotindex, qgrasps[robotindex], Tobj, 
                     qrobot_ref=qrefs[robotindex])
                    if sol is None:
                        if self._print:
                            print '    No IK solution for robot {0}'.format(robotindex)
                        self.ED[vnear.id + vnext_id] += 1
                        return status
                    sols.append(sol)

            else:
                # Usual TRANSFER extension
                [passed, sols, Tobj] = self.SampleCP(isurface, vnear.config.qgrasps,
                                                     vnear.config.qrobots)

                if not passed:
                    # Infeasible placement is probably because it is
                    # kinematically unreachable
                    self.ED[vnear.id + vnext_id] += 1
                    return status

            cnew = self.Config(sols, Tobj, vnear.config.qgrasps, CG1CG2CP, isurface)
            vnew = self.Vertex(cnew, FW, level=nextlevel)
            # TODO: code for planning TRANSFER trajectories
            [plannerstatus, trajectorystrings] = [True, ['', '']]
            
            if not (plannerstatus == 1):
                return status

            # Now successfully extended
            self.treestart.AddVertex(vindex, vnew, trajectorystrings, TRANSFER)
            self._weighteddist[FW].append((self.treestart[-1].index, 
                                           ComputeWeight(self.treestart[-1].level)))
            status = ADVANCED
            return status
        

    def ExtendBWFrom(self, vindex):
        return TRAPPED


    def ConnectFW(self):
        if self._print:
            print '  [ConnectFW]'
        status = TRAPPED

        vfw = self.treestart[-1] # newly added vertex
        cfw = vfw.config
        curtype = vfw.trajtype
        onestepfromvfwlist = self.QFW.graphdict[vfw.id]
        
        # Check wheter the newly added vertex on treestart is one step
        # away from goal
        cond1 = (len(onestepfromvfwlist) == 1)
        cond2 = (onestepfromvfwlist[0] == self.treeend[0].id)
        if cond1 and cond2:
            # The next vertex is vertex goal
            if curtype == TRANSIT:
                # TRANSFER
                # TODO: code for planning TRANSFER trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]
                if not (plannerstatus == 1):
                    return status

                # Now we have reached the start. 
                status = REACHED
                
                # # Stack the vertex start on top of treestart
                # self.treestart.verticeslist.append(self.treestart[0])
                
                # Stack the vertex vindex on top of treeend
                self.treeend.verticeslist.append(self.treeend[0])

                self.connectingtrajectorystrings = trajectorystrings
                self.connectingtrajectorytype = TRANSFER
                return status
            else:
                # TRANSIT
                # TODO: code for planning TRANSIT trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]
                if not (plannerstatus == 1):
                    return status

                # Now we have reached the goal.
                status = REACHED

                # # Stack the vertex vindex on top of treestart
                # self.treestart.verticeslist.append(self.treestart[vindex])
                
                # Stack the vertex goal on top of treeend
                self.treeend.verticeslist.append(self.treeend[0])

                self.connectingtrajectorystrings = trajectorystrings
                self.connectingtrajectorytype = TRANSIT
                return status

        # Create a list of vertices which are two steps away from vfw
        temp = []
        for v_id in onestepfromvfwlist:
            temp += self.QFW.graphdict[v_id]
        twostepsfromvfwlist = []
        # Append only non-redundant vertices to twostepsfromvfwlist
        for v_id in temp:
            if v_id not in twostepsfromvfwlist:
                twostepsfromvfwlist.append(v_id)

        # Now examine all existing vertices on treeend which are two
        # steps away from vfw
        idslist = [v.id for v in self.treeend]
        for v_id in twostepsfromvfwlist:
            if v_id not in idslist:
                continue

            vbw_index = idslist.index(v_id)
            vbw = self.treeend[vbw_index]
            cbw = vbw.config
            
            # Now execute a two-step extension
            if curtype == TRANSIT:
                # TRANSFER --> TRANSIT
                # TRANSFER
                qgrasps = cfw.qgrasps
                Tobj = cbw.tobj
                qrefs = cfw.qrobots
                sols = []
                for robotindex in xrange(self.nrobots):
                    sol = self.ComputeGraspConfiguration\
                    (robotindex, qgrasps[robotindex], Tobj, 
                     qrobot_ref=qrefs[robotindex])
                    if sol is None:
                        if self._print:
                            print '    No IK solution for robot {0}'.format(robotindex)
                        vnext_id = (vfw.level + 1, cbw.isurface, 
                                    cfw.qgrasps[0][1], cfw.qgrasps[1][1])
                        self.ED[vfw.id + vnext_id] += 1
                        return status
                    sols.append(sol)

                cnew = self.Config(sols, Tobj, qgrasps, CG1CG2CP, cbw.isurface)
                vnew = self.Vertex(cnew, FW, level=vfw.level + 1)
                             
                # TODO: code for planning TRANSFER trajectories
                [plannerstatus1, trajectorystrings1] = [True, ['', '']]
                if not (plannerstatus1 == 1):
                    continue

                # Now successfully extended
                self.treestart.AddVertex(vfw.index, vnew, trajectorystrings1, TRANSFER)
                self._weighteddist[FW].append((self.treestart[-1].index, 
                                               ComputeWeight(self.treestart[-1].level)))

                # TRANSIT
                # TODO: code for planning TRANSIT trajectories
                [plannerstatus2, trajectorystrings2] = [True, ['', '']]
                if not (plannerstatus2 == 1):
                    continue

                # Now successfully connected
                self.treeend.verticeslist.append(vbw)
                self.connectingtrajectorystrings = trajectorystrings2
                self.connectingtrajectorytype = TRANSIT
                status = REACHED
                return status

            else:
                # TRANSIT --> TRANSFER
                # TRANSIT
                qgrasps = cbw.qgrasps
                Tobj = cfw.tobj
                qref = cbw.qrobots
                sols = []
                for robotindex in xrange(self.nrobots):
                    sol = self.ComputeGraspConfiguration\
                    (robotindex, qgrasps[robotindex], Tobj, 
                     qrobot_ref=qrefs[robotindex])
                    if sol is None:
                        if self._print:
                            print '    No IK solution for robot {0}'.format(robotindex)
                        vnext_id = (vfw.level + 1, cfw.isurface, 
                                    cbw.qgrasps[0][1], cbw.qgrasps[1][1])
                        self.ED[vfw.id + vnext_id] += 1
                        return status
                    sols.append(sol)

                cnew = self.Config(sols, Tobj, qgrasps, CG1CG2CP, cfw.isurface)
                vnew = self.Vertex(cnew, FW, level=vfw.level + 1)
                
                # TODO: code for planning TRANSIT trajectories
                [plannerstatus1, trajectorystrings1] = [True, ['', '']]
                if not (plannerstatus1 == 1):
                    continue

                # Now successfully extended
                self.treestart.AddVertex(vfw.index, vnew, trajectorystrings1, TRANSIT)
                self._weighteddist[FW].append((self.treestart[-1].index, 
                                               ComputeWeight(self.treestart[-1].level)))

                # TRANSFER
                [plannerstatus2, trajectorystring2] = [True, ['', '']]
                if not (plannerstatus2 == 1):
                    continue
                
                # Now successfully extended
                self.treeend.verticeslist.append(vbw)
                self.connectingtrajectorystrings = trajectorystrings2
                self.connectingtrajectorytype = TRANSFER
                status = REACHED
                return status                
            
        # TRAPPED
        return status
                
                
    def ConnectBW(self):
        return TRAPPED


    def SampleCP(self, isurface, qgrasps, qrobot_refs):
        """
        SampleCP samples a configuration in CP.
        
        Returns
        -------
        passed : bool
        sols : list of n-vectors
            A list containing configurations of robots grasping 
            the object with qgrasps
        Tobj : 4x4 transformation matrix
        """
        passed = False
        
        print "    [SampleCP]"
        for _ in xrange(self._ntrialsCP):
            print "      trial {0} : ".format(_ + 1),
            objincollision = True
            while objincollision:
                xobj = _RNG.uniform(self.xobjlim[0], self.xobjlim[1])
                yobj = _RNG.uniform(self.yobjlim[0], self.yobjlim[1])
                thetaobj = _RNG.uniform(self.thetaobjlim[0], self.thetaobjlim[1])

                qobj = [xobj, yobj, thetaobj, isurface]
                Tobj = Utils.ComputeTObject(qobj, self.S, self.zoffset)

                if False:
                    # TODO: add external implementation of placement checking
                    if not self.CheckPlacement(self.object, Tobj, isurface, self.S):
                        continue

                self.object.SetTransform(Tobj)
                self.SetHOMEConfigurations()
                objincollision = self.env.CheckCollision(self.object)

            # Compute IK solutions for robots
            sols = []
            for robotindex in xrange(self.nrobots):
                ibox = qgrasps[robotindex][0]
                graspedlink = self.object.GetLinks()[ibox]
                extents = self.boxinfos[ibox].extents
                robot = self.robots[robotindex]
                manip = self.manips[robotindex]

                Tgripper = Utils.ComputeTGripper(graspedlink.GetTransform(),
                                                 qgrasps[robotindex], extents,
                                                 unitscale=False)

                with robot:
                    robot.SetActiveDOFValues(qrobot_refs[robotindex])
                    sol = manip.FindIKSolution\
                    (Tgripper, orpy.IkFilterOptions.CheckEnvCollisions)
                if sol is None:
                    print "no ik solution for robot {0}".format(robotindex)
                    break

                # A naive method to check whether two configurations
                # are in the same class is to look at the sign of each
                # element.
                qrobot_ref = qrobot_refs[robotindex]
                checksolution = [qrobot_ref[i]*sol[i] >= 0 for i in xrange(len(sol))]
                # if not np.alltrue(checksolution):
                if checksolution.count(False) > 2:
                    # print 'ik solution is possibly in a different class from ref.'
                    # self.robots[robotindex].SetActiveDOFValues(qrobot_ref)
                    # self.object.SetTransform
                    # print qrobot_ref
                    # raw_input()
                    # self.robots[robotindex].SetActiveDOFValues(sol)
                    # print sol
                    # raw_input()
                    break

                sols.append(sol)

            # Grasp checking takes some time so we do it after both IK
            # solutions are found.
            if len(sols) == self.nrobots:
                if not self.CheckGrasp(0, sols[0], Tobj):
                    print "invalid grasp for robot 0"
                    break

                if not self.CheckGrasp(1, sols[1], Tobj):
                    print "invalid grasp for robot 1"
                    break                
   
                print "passed"
                passed = True
                break
            
            else:
                continue

        if passed:
            return [passed, sols, Tobj]
        else:
            return [False, None, None]            


    def SampleCG(self, Tobj, isurface, approachingdirs):
        """
        SampleCG samples a configuration in CG.
        
        Returns
        -------
        passed : bool
        sols : list of n-vectors
            A list containing configurations of robots grasping 
            the object with qgrasps
        qgrasps : list of 4-vectors
        """
        passed = False
        self.object.SetTransform(Tobj)
        
        print "    [SampleCG]"
        for _ in xrange(self._ntrialsCP):
            print "      trial {0} : ".format(_ + 1),
            sols = []
            qgrasps = []

            self.SetHOMEConfigurations()

            for robotindex in xrange(self.nrobots):
                ibox = int(approachingdirs[robotindex])/6
                realapproachingdir = np.mod(approachingdirs[robotindex], 6)
                boxinfo = self.boxinfos[ibox]
                graspedlink = self.object.GetLinks()[ibox]
                
                # Sample a sliding direction
                slidingdir = _RNG.choice(boxinfo.possibleslidingdir[realapproachingdir])
                # Resample a sliding direction if necessary
                while (isurface, realapproachingdir, slidingdir) not in boxinfo.intervals:
                    slidingdir = \
                    _RNG.choice(boxinfo.possibleslidingdir[realapproachingdir])

                # Sample a value for delta (where to grasp along the sliding direction)
                delta = WeightedChoice2\
                (boxinfo.intervals[isurface, realapproachingdir, slidingdir])

                # Assign qgrasp
                qgrasp = [ibox, approachingdirs[robotindex], slidingdir, delta]
            
                # Compute Tgripper
                Tgripper = Utils.ComputeTGripper(graspedlink.GetTransform(),
                                                 qgrasp, boxinfo.extents, unitscale=False)

                # Compute an IK solution for robot robotindex
                sol = self.manips[robotindex].FindIKSolution\
                (Tgripper, orpy.IkFilterOptions.CheckEnvCollisions)
                if sol is None:
                    print "no ik solution for robot {0}".format(robotindex)
                    break
                
                sols.append(sol)
                qgrasps.append(qgrasp)

                # Set the robot at the found configuration
                self.robots[robotindex].SetActiveDOFValues(sol)

            # Grasp checking takes some time so we do it after both IK
            # solutions are found.
            if len(sols) == self.nrobots:
                if not self.CheckGrasp(0, sols[0], Tobj):
                    print "invalid grasp for robot 0"
                    break

                if not self.CheckGrasp(1, sols[1], Tobj):
                    print "invalid grasp for robot 1"
                    break                
   
                print "passed"
                passed = True
                break
            
            else:
                continue

        if passed:
            return [passed, sols, qgrasps]
        else:
            return [False, None, None]


    def SetHOMEConfigurations(self):
        for robot in self.robots:
            robot.SetActiveDOFValues(HOME)

            
    def ExtractVerticesSequence(self):
        if not self._hassolution:
            print Colorize('The planner has not yet found any solution', 'red')
            return []

        vseq = []
        v = self.treestart[-1].Clone()
        while v.parentindex is not None:
            vseq.insert(0, v)
            v = self.treestart[v.parentindex].Clone()            
        vseq.insert(0, self.treestart[0].Clone())
        
        for i in xrange(1, len(vseq)):
            vseq[i - 1].trajtype = vseq[i].trajtype
            
        if vseq[-2].trajtype == TRANSIT:
            vseq[-1].trajtype = TRANSFER
        else:
            vseq[-1].trajtype = TRANSIT        

        v = self.treeend[-1].Clone()
        while v.parentindex is not None:
            vseq.append(v)
            v = self.treeend[v.parentindex].Clone()
        vseq.append(self.treeend[0].Clone())

        return vseq


############################################################
#                       Utilities
############################################################
def ComputeWeight(l):
    """
    ComputeWeight computes a weight for an l^th-level vertex 
    """
    w = 2**(0.5*l)
    return w


def WeightedChoice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        upto += w
        if upto >= r:
            return c
    assert False


def WeightedChoice2(choices):
    total = sum(w for c, w in choices)
    s = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        upto += w
        if upto >= s:
            return c + (upto - s)
    assert False
