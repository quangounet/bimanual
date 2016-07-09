import openravepy as orpy
import numpy as np
import time
import random
import copy

# For plotting manipulation graphs
from matplotlib import pyplot as plt
from pylab import ion
ion()

import BaseBimanualManipulationPlanner3 as bbmp
from BaseBimanualManipulationPlanner3 import (FW, BW, REACHED, ADVANCED, TRAPPED,
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
        for isurface in xrange(len(self.objectdb.S) - 1):
            for ibox in xrange(len(self.object.GetLinks())):
                for appdir in \
                    self.objectdb.boxinfos[ibox].possibleapproachingdir[isurface]:
                    vnew = (isurface, appdir + 6*ibox)
                    self.graphvertices.append(vnew)
        te_gath = time.time()
        print Colorize('  Created a set of GPG vertices in {0} sec.'.\
                           format(te_gath - ts_gath), 'yellow')

        # Assume identical Graphs
        graphdict = bgpg.LoadGraphDict(self.object.GetKinematicsGeometryHash())
        if graphdict is None:
            print Colorize('Constructing Grasp-Placement Graph', 'yellow')
            ts_con = time.time()
            self.BGPG_original = bgpg.BimanualGraspPlacementGraph(self.graphvertices)
            te_con = time.time()
            print Colorize('  Construction time: {0} sec.'.\
                               format(te_con - ts_con), 'yellow')
        else:
            print Colorize('Graphdict found', 'yellow')
            self.BGPG_original = bgpg.BimanualGraspPlacementGraph(graphdict=graphdict)
            
        # Threshold value for removal of infeasible edges
        self._threshold = 30#15
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
        self.vstart = bbmp.Vertex(cstart.Clone(), FW)
        self.vgoal = bbmp.Vertex(cgoal.Clone(), BW)
        self.treestart = bbmp.Tree(self.vstart, FW)
        self.treeend = bbmp.Tree(self.vgoal, BW)
        
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

        # We need to find out what the first trajtype is (then we know
        # the rest). For simplicity, we assume that either the start
        # configuration or the goal configuration is in P. This
        # implies there is no ambiguity in the first trajtype.
        self.firsttrajtype = None
        for vid in self.BGPG.graphdict[self.vstart.id]:
            if ((self.vstart.id[0] == vid[0]) and (not (self.vstart.id[1] == vid[1]))):
                self.firsttrajtype = TRANSIT
                break
            elif ((self.vstart.id[0] == vid[0]) and (not (self.vstart.id[2] == vid[2]))):
                self.firsttrajtype = TRANSIT
                break
            elif ((self.vstart.id[1] == vid[1]) and (not (self.vstart.id[0] == vid[0]))):
                self.firsttrajtype = TRANSFER
                break
            else:
                # Not enough information
                continue
        assert(self.firsttrajtype is not None)

        self.EliminateInfeasibleTerminalGrasps()
        
        # Remove vertices which can only connected to others by
        # TRANSIT
        toberemoved = []
        for key in self.BGPG.graphdict:
            if key[1] is None and key[2] is None:
                continue
            ok = False
            for val in self.BGPG.graphdict[key]:
                if not (val[0] == key[0]):
                    # This vertex can be connected to other placements
                    ok = True
                    break
            if not ok:
                toberemoved.append(key)
        for tbrkey in toberemoved:
            self.BGPG.graphdict.pop(tbrkey, None)
            
            for key in self.BGPG.graphdict:
                try:
                    self.BGPG.graphdict[key].remove(tbrkey)
                except:
                    continue
                
        # self._dist is a dictionary of distances to vgoal
        self._dist, prev = self.BGPG.RunDijkstra(self.vgoal.id)


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
        start_id = self.vstart.id
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
                        boxinfo = self.objectdb.boxinfos[ibox]
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
        goal_id = self.vgoal.id
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
                        boxinfo = self.objectdb.boxinfos[ibox]
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

        if self._print:
            print Colorize('Creating an edge database for storing statistics. . .',
                           'yellow')
        # Each entry stores the number of IK failure. High number
        # of IK failure can probably reflect the actual kinematic
        # infeasibility.
        self.ED = dict() # Edge database
        for key in self.BGPG.graphdict:
            for val in self.BGPG.graphdict[key]:
                edge = key + val
                if edge not in self.ED:
                    self.ED[edge] = 0

        self.tbredges = [] # to-be-removed edges

        # Compute weights for start and goal vertices. A vertex
        # with more weight is more likely to be sampled. This
        # weight is then roughly a bias thatt we put in the
        # exploration.
        self._weighteddist = [[], []]
        self._weighteddist[FW].append((self.treestart[0].index, 
                                       ComputeWeight(self.treestart[0].level)))
        self._weighteddist[BW].append((self.treeend[0].index, 
                                       ComputeWeight(self.treeend[0].level)))
                    
        # Planning loop
        t = 0.0 # running time
        it = 0  # number of iterations
        
        while (self._runningtime < timeout) and (not self._hassolution):
            it += 1
            if self._print:
                print Colorize('iteration : {0}'.format(it), 'blue')
            ts = time.time()
                
            # Eliminate edges that are declared infeasible (according
            # to self._threshold)
            notransfer = [] # keep track of vertices with no
                            # connection to other placements
            for edge in self.tbredges:
                vid1 = edge[0:3]
                vid2 = edge[3:6]
                self.BGPG.graphdict[vid1].remove(vid2)
                
                if len(self.BGPG.graphdict[vid1]) == 0:
                    print 'All neighbors of vid {0} have been removed. Removing {0}'.\
                    format(vid1)
                    self.BGPG.graphdict.pop(vid1, None)
                    
                    for key in self.BGPG.graphdict:
                        try:
                            self.BGPG.graphdict[key].remove(vid1)
                        except:
                            continue

                else:
                    # See if the vertex vid1 has any connection to other
                    # placements
                    if vid1.count(None) > 0:
                        continue
                    
                    ok = False
                    for val in self.BGPG.graphdict[vid1]:
                        if not (val[0] == vid1[0]):
                            ok = True
                            break
                    if not ok:
                        self.BGPG.graphdict.pop(vid1, None)

                        for key in self.BGPG.graphdict:
                            try:
                                self.BGPG.graphdict[key].remove(vid1)
                            except:
                                continue
                
                self.ED.pop(edge, None)
            self.tbredges = []
            
            treedirection = FW # subject ot change
            # Sample a node from the tree
            index = self.SampleTree(treedirection)

            if treedirection == FW:
                # Forward extension
                status = self.ExtendFWFrom(index)
                if not (status == TRAPPED):
                    if self._print:
                        print Colorize('Extension successful', 'green')
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
                    if self._print:
                        print Colorize('Extension successful', 'green')
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
        print Colorize('Allotted time ({0} sec.) is exhausted after {1} iterations'.\
                           format(timeout, it), 'red')
        self._runningtime += t
        self._iterations += it
        return False            
            
                
    def SampleTree(self, treedirection):
        return WeightedChoice(self._weighteddist[treedirection])


    def ExtendFWFrom(self, vindex):
        
        vfrom = self.treestart[vindex]
        cfrom = vfrom.config
        prevtype = vfrom.trajtype
        if self._print:
            print '  [ExtendFWFrom] vindex = {0}, level = {1}, dist = {2}'.\
            format(vindex, vfrom.level, self._dist[vfrom.id])
        status = TRAPPED

        # possibleextensions contains all vertices adjacent to vfrom
        try:
            possibleextensions = self.BGPG.graphdict[vfrom.id]
        except KeyError:
            # vfrom has already been removed from the dictionary since
            # it has no connection to other placements
            print '  vindex = {0} has no connection to other placements'.\
            format(vfrom.id)
            print '  Removing vindex from the tree'
            self._weighteddist[FW][vindex] = (vindex, 0) # assign 0 weight to the vertex
            return status

        # legalextensions contains all adjacent vertices which will
        # not cause manipulation path redundancy
        cond1 = ((self.firsttrajtype == TRANSIT) and (np.mod(vfrom.level, 2) == 0))
        cond2 = ((self.firsttrajtype == TRANSFER) and (np.mod(vfrom.level, 2) == 1))
        if cond1 or cond2:
            # the next extension is TRANSIT
            legalextensions = [vid for vid in possibleextensions if
                               (vfrom.id[0] == vid[0]) and 
                               (not (vfrom.id == vid)) and
                               (vid.count(None == 0))]
            try:
                assert(len(legalextensions) > 0)
            except:
                print 'Something is wrong here. No legal extension for TRANSIT'
                print 'vfrom.id = {0}'.format(vfrom.id)
                print 'len(possibleextensions) = {0}'.format(len(possibleextensions))
                assert(False)
        else:
            legalextensions = [vid for vid in possibleextensions if 
                               (not (vid[0] == vfrom.id[0])) or (vfrom.id == vid)]
            try:
                assert(len(legalextensions) > 0)
            except:
                print 'No logal extension for TRANSFER'
                print 'CHECK YOUR CODE!'
                assert(False)
            
        choices = self.ComputeProbDist(legalextensions) # assign weights to vertices

        k = 2 # k-NN extension
        for i in xrange(k):
            # Sample an id of a vertex to extend to
            vnext_id = WeightedChoice(choices)
            # Unpack the id
            [isurface, approachingdir1, approachingdir2] = vnext_id

            if self._print:
                print '  [ExtendFWFrom] extending to vnext_id = {0}'.\
                format(vnext_id)

            # Check whether the upcoming edge should be TRANSIT or TRANSFER
            # if any of these conditions is satisfied, the next edge is TRANSIT
            cond1 = (prevtype == TRANSFER)
            cond2 = ((vfrom.id[0] == vnext_id[0]) and
                     (not (vfrom.id[1] == vnext_id[1])))
            cond3 = ((vfrom.id[0] == vnext_id[0]) and 
                     (not (vfrom.id[2] == vnext_id[2])))
            cond4 = ((self.firsttrajtype == TRANSIT) and 
                     (np.mod(vfrom.level, 2) == 0))
            cond5 = ((self.firsttrajtype == TRANSFER) and
                     (np.mod(vfrom.level, 2) == 1))
            if cond1 or cond2 or cond3 or cond4 or cond5:
                if self._print:
                    print '  TRANSIT extension'
                # TRANSIT
                try:
                    assert(approachingdir1 is not None) # check soundness
                    assert(approachingdir2 is not None) # check soundness
                except:
                    continue
                approachingdirs = [approachingdir1, approachingdir2]
                [passed, sols, qgrasps] = self.SampleCG(vfrom.config.tobj, isurface,
                                                        approachingdirs)

                if not passed:
                    # Unable to sample new feasible grasps
                    edge = vfrom.id + vnext_id
                    self.ED[edge] += 1
                    if self.ED[edge] > self._threshold:
                        if edge not in self.tbredges:
                            self.tbredges.append(edge)
                    continue
                
                cnew = bbmp.Config(sols, vfrom.config.tobj, qgrasps, CG1CG2CP, isurface)
                nextlevel = vfrom.level + 1
                vnew = bbmp.Vertex(cnew, FW, level=nextlevel)
                # TODO: code for planning TRANSIT trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]

                if not (plannerstatus == 1):
                    continue

                # Now successfully extended
                self.treestart.AddVertex(vindex, vnew, trajectorystrings, TRANSIT)
                self._weighteddist[FW].append((self.treestart[-1].index, 
                                               ComputeWeight(self.treestart[-1].level)))
                status = ADVANCED
                return status
            else:
                if self._print:
                    print 'TRANSFER extension'
                # TRANSFER               
                [passed, sols, Tobj] = self.SampleCP(isurface, vfrom.config.qgrasps,
                                                     vfrom.config.qrobots,
                                                     Tprev=vfrom.config.tobj)
                    
                if not passed:
                    # Unable to sample a new feasible placement
                    edge = vfrom.id + vnext_id
                    self.ED[edge] += 1
                    if self.ED[edge] > self._threshold:
                        if edge not in self.tbredges:
                            self.tbredges.append(edge)
                    continue
                    
                cnew = bbmp.Config(sols, Tobj, vfrom.config.qgrasps, CG1CG2CP, isurface)
                nextlevel = vfrom.level + 1
                vnew = bbmp.Vertex(cnew, FW, level=nextlevel)
                # TODO: code for planning TRANSFER trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]

                if not (plannerstatus == 1):
                    continue

                # Now successfully extended
                self.treestart.AddVertex(vindex, vnew, trajectorystrings, TRANSFER)
                self._weighteddist[FW].append((self.treestart[-1].index, 
                                               ComputeWeight(self.treestart[-1].level)))
                status = ADVANCED
                return status

        return status
            

    def ExtendBWFrom(self, index):
        return TRAPPED


    def ConnectFW(self):
        status = TRAPPED
        if self._print:
            print '  [ConnectFW]'
        # Treestart has just been extended. Try connecting the newly
        # added vertex (on treestart) with treeend.
        vfrom = self.treestart[-1]
        cfrom = vfrom.config
        curtype = vfrom.trajtype

        if not (self._dist[vfrom.id] == 2):
            print '  vgoal is not 2-step reachable'
            return status
        
        ts = time.time()
        if curtype == TRANSIT:
            # The next step is TRANSFER
            onestep = [vid for vid in self.BGPG.graphdict[vfrom.id] if 
                       (not (vfrom.id[0] == vid[0])) or (vfrom.id == vid)]
        else:
            # The next step is TRANSIT
            onestep = [vid for vid in self.BGPG.graphdict[vfrom.id] if
                       (vfrom.id[0] == vid[0]) and (not (vfrom.id == vid))]
        twosteps = []
        for v1_id in onestep:
            if curtype == TRANSIT:
                # The next two steps is TRANSIT
                temp = [vid for vid in self.BGPG.graphdict[v1_id] if
                        (v1_id[0] == vid[0]) and (not (v1_id == vid))]
            else:
                # The next two steps is TRANSFER
                temp = [vid for vid in self.BGPG.graphdict[v1_id] if
                        (not (v1_id[0] == vid[0])) or (v1_id == vid)]
            for v2_id in temp:
                if v2_id not in twosteps:
                    twosteps.append(v2_id)
        te = time.time()
        print '  exploring two-step reachable vertices in {0} sec.'.format(te - ts)

        nnindices = self.NearestNeighborIndices(cfrom, BW)
        print '  len(nnindices) = {0}'.format(len(nnindices))
        for vindex in nnindices:
            vnext = self.treeend[vindex]
            cnext = vnext.config
            if vnext.id not in twosteps:
                # This vertex is not two-step reachable
                continue

            if curtype == TRANSIT:
                # TRANSFER --> TRANSIT
                # TRANSFER
                sols = []
                for rindex in xrange(len(self.robots)):
                    sol = self.ComputeGraspConfiguration\
                    (rindex, cfrom.qgrasps[rindex], cnext.tobj, cfrom.qrobots[rindex])
                    if sol is None:
                        print '    No IK solution for robot {0}'.format(rindex)
                        edge = vfrom.id + (vnext.id[0], vfrom.id[1], vfrom.id[2])
                        self.ED[edge] += 1
                        if self.ED[edge] > self._threshold:
                            if edge not in self.tbredges:
                                self.tbredges.append(edge)
                        break
                    else:
                        sols.append(sol)
                if len(sols) < 2:
                    continue
                
                cnew = bbmp.Config(sols, cnext.tobj, cfrom.qgrasps, CG1CG2CP, 
                                   cnext.isurface)
                nextlevel = vfrom.level + 1
                vnew = bbmp.Vertex(cnew, FW, level=nextlevel)
                # TODO: code for planning TRANSFER trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]

                if not (plannerstatus == 1):
                    return status

                # Now successfully extended
                self.treestart.AddVertex(vfrom.index, vnew, trajectorystrings, TRANSFER)
                self._weighteddist[FW].append((self.treestart[-1].index, 
                                               ComputeWeight(self.treestart[-1].level)))
                status = ADVANCED
                
                # TRANSIT
                # TODO: code for planning TRANSIT trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]
                
                if not plannerstatus:
                    return status

                # Now successfully extended
                self.treeend.verticeslist.append(vnext)
                self.connectingtrajectorystrings = trajectorystrings
                self.connectingtrajectorytype = TRANSIT
                status = REACHED
                return status
            
            else:
                # TRANSIT --> TRANSFER
                # TRANSIT
                sols = []
                for rindex in xrange(len(self.robots)):
                    sol = self.ComputeGraspConfiguration\
                    (rindex, cnext.qgrasps[rindex], cfrom.tobj, cfrom.qrobots[rindex])
                    if sol is None:
                        print '    No IK solution for robot {0}'.format(rindex)
                        edge = vfrom.id + (vfrom.id[0], vnext.id[1], vnext.id[2])
                        self.ED[edge] += 1
                        if self.ED[edge] > self._threshold:
                            if edge not in self.tbredges:
                                self.tbredges.append(edge)
                        break
                    else:
                        sols.append(sol)
                if len(sols) < 2:
                    continue
                
                cnew = bbmp.Config(sols, cfrom.Tobj, cnext.qgrasps, CG1CG2CP, 
                                   cfrom.isurface)
                nextlevel = vfrom.level + 1
                vnew = bbmp.Vertex(cnew, FW, level=nextlevel)
                # TODO: code for planning TRANSIT trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]

                if not plannerstatus:
                    return status

                # Now successfully extended
                self.treestart.AddVertex(vfrom.index, vnew, trajectorystrings, TRANSIT)
                self._weighteddist[FW].append((self.treestart[-1].index, 
                                               ComputeWeight(self.treestart[-1].level)))
                status = ADVANCED
                
                # TRANFER
                # TODO: code for planning TRANSFER trajectories
                [plannerstatus, trajectorystrings] = [True, ['', '']]
                
                if not plannerstatus:
                    return status

                # Now successfully extended
                self.treeend.verticeslist.append(vnext)
                self.connectingtrajectorystrings = trajectorystrings
                self.connectingtrajectorytype = TRANSFER
                status = REACHED
                return status

        if self._print:
            print '  [ConnectFW] failed: running out of neighbors'
        return status
                

    def ConnectBW(self):
        return TRAPPED


    def SampleCP(self, isurface, qgrasps, qrobot_refs, Tprev):
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
        ts = time.time()
        _printtime = False
        _printsamplecp = False
        passed = False
        
        if _printsamplecp:
            print "    [SampleCP]"
        for _ in xrange(self._ntrialsCP):
            if _printsamplecp:
                print "      trial {0} : ".format(_ + 1),
            objincollision = True
            while objincollision:
                xobj = _RNG.uniform(self.xobjlim[0], self.xobjlim[1])
                yobj = _RNG.uniform(self.yobjlim[0], self.yobjlim[1])
                thetaobj = _RNG.uniform(self.thetaobjlim[0], self.thetaobjlim[1])

                qobj = [xobj, yobj, thetaobj, isurface]
                Tobj = Utils.ComputeTObject(qobj, self.objectdb.S, self.zoffset)

                if False:
                    # TODO: add external implementation of placement checking
                    if not self.CheckPlacement(self.object, Tobj, isurface, 
                                               self.objectdb.S):
                        continue

                self.object.SetTransform(Tobj)
                self.SetHOMEConfigurations()
                objincollision = self.env.CheckCollision(self.object)

            # Compute IK solutions for robots
            sols = []
            for robotindex in xrange(self.nrobots):
                ibox = qgrasps[robotindex][0]
                graspedlink = self.object.GetLinks()[ibox]
                extents = self.objectdb.boxinfos[ibox].extents
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
                    if _printsamplecp:
                        print "no ik solution for robot {0}".format(robotindex)
                    break

                # # A naive method to check whether two configurations
                # # are in the same class is to look at the sign of each
                # # element.
                # qrobot_ref = qrobot_refs[robotindex]
                # checksolution = [qrobot_ref[i]*sol[i] >= 0 for i in xrange(len(sol))]
                # # if not np.alltrue(checksolution):
                # if checksolution.count(False) > 2:
                #     # print 'ik solution is possibly in a different class from ref.'
                #     # self.robots[robotindex].SetActiveDOFValues(qrobot_ref)
                #     # self.object.SetTransform
                #     # print qrobot_ref
                #     # raw_input()
                #     # self.robots[robotindex].SetActiveDOFValues(sol)
                #     # print sol
                #     # raw_input()
                #     break

                sols.append(sol)

            # Grasp checking takes some time so we do it after both IK
            # solutions are found.
            if len(sols) == self.nrobots:
                if not self.CheckGrasp(0, sols[0], Tobj):
                    if _printsamplecp:
                        print "invalid grasp for robot 0"
                    break

                if not self.CheckGrasp(1, sols[1], Tobj):
                    if _printsamplecp:
                        print "invalid grasp for robot 1"
                    break                
   
                if _printsamplecp:
                    print "passed"
                passed = True
                break
            
            else:
                continue
            
        te = time.time()
        if _printtime:
            print "[SampleCP] time = {0} sec.".format(te - ts)
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
        ts = time.time()
        _printtime = False
        _printsamplecg = False

        passed = False
        self.object.SetTransform(Tobj)
        
        if _printsamplecg:
            print "    [SampleCG]"
        for _ in xrange(self._ntrialsCG):
            if _printsamplecg:
                print "      trial {0} : ".format(_ + 1),
            sols = []
            qgrasps = []

            self.SetHOMEConfigurations()

            for robotindex in xrange(self.nrobots):
                ibox = int(approachingdirs[robotindex])/6
                realapproachingdir = np.mod(approachingdirs[robotindex], 6)
                boxinfo = self.objectdb.boxinfos[ibox]
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
                    if _printsamplecg:
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
                    if _printsamplecg:
                        print "invalid grasp for robot 0"
                    continue

                if not self.CheckGrasp(1, sols[1], Tobj):
                    if _printsamplecg:
                        print "invalid grasp for robot 1"
                    continue
   
                if _printsamplecg:
                    print "passed"
                passed = True
                break
            
            else:
                continue

        te = time.time()
        if _printtime:
            print "[SampleCG] time = {0} sec.".format(te - ts)
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


    def ComputeProbDist(self, legalextensions):
        _printtime = False
        ts = time.time()
        base = 10
        # Initialize a dictionary
        temp = dict()
        maxdist = max(self._dist.values())
        summaxdist = sum(range(maxdist + 1))
        for i in xrange(maxdist + 1):
            temp[i] = 0
        for vid in legalextensions:
            # Count the number of vertices with a specific distance
            temp[self._dist[vid]] += 1

        choices = []
        for vid in legalextensions:
            prob = base**(-self._dist[vid])/temp[self._dist[vid]]
            choices.append((vid, prob))

        te = time.time()
        if _printtime:
            print "[ComputeProbDist] time = {0} sec.".format(te - ts)
        return choices
            

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
    _RNG.shuffle(choices)
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    # print "weightedchoice r = {0}".format(r)
    # print "choice = {0}".format(choices)
    assert False


def WeightedChoice2(choices):
    total = sum(w for c, w in choices)
    s = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= s:
            return c + (upto + w - s)
        upto += w
    assert False
