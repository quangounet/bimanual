import numpy as np
import copy
import time
from pymanip.utils import GraspPlacementGraph as GPG

# For plotting manipulation graphs
from matplotlib import pyplot as plt
from pylab import ion
from mpl_toolkits.mplot3d import Axes3D
ion()


TRANSIT = 0
TRANSFER = 1

"""
Assume first that there are only configurations in P and G1G2P ==
either both robots grasping or no robot grasping.

"""

################################################################################
#                        Bimanual Grasp-Placement Graph
################################################################################
class BimanualGraspPlacementGraph(GPG.GraspPlacementGraph):

    def __init__(self, verticeslist2d=[], graphdict=None, autogenerate=True):
        """
        verticeslist is a list of 2D vertices. Assuming that the two
        robots are identical, self.verticeslist, which contains 3D
        vertices of the form v = (p, g1, g2), can be obtained from 2D
        vertices.
        """
        if graphdict is None:
            # The Bimanual GPG is created anew.
            self._availableplacements = []

            self.verticeslist = []
            for v1 in verticeslist2d:
                if v1[0] not in self._availableplacements:
                    self._availableplacements.append(v1[0])

                for v2 in verticeslist2d:
                    if (v1[0] == v2[0]):
                        v = (v1[0], v1[1], v2[1])
                        if v not in self.verticeslist:
                            self.verticeslist.append(v)

            self.graphdict = dict()
            self.nvertices = len(self.verticeslist)

            if autogenerate:
                self.GenerateGraphDictionary()

        else:
            # The Bimanual GPG is loaded from somewhere.
            self.graphdict = copy.deepcopy(graphdict)
            self.verticeslist = graphdict.keys()
            self._availableplacements = []
            for v in self.verticeslist:
                if v[0] not in self._availableplacements:
                    self._availableplacements.append(v[0])

            self.nvertices = len(self.verticeslist)


    def GenerateGraphDictionary(self):
        ts = time.time()
        if (self.nvertices > 0):
            # Add additional vertices (vertices in P)
            for p in self._availableplacements:
                vnew = (p, None, None)
                if vnew not in self.verticeslist:
                    self.verticeslist.append(vnew)

            self.verticeslist.sort()
            self.nvertices = len(self.verticeslist)

            # Initialize graph dictionary's entries
            for v in self.verticeslist:
                self.graphdict[v] = []

            # Add edges of the graph
            for i in xrange(self.nvertices):
                vi = self.verticeslist[i]
                for j in xrange(i + 1, self.nvertices):
                    vj = self.verticeslist[j]
                    
                    if (vi[0] == vj[0]):
                        # TRANSIT edge
                        if (vj not in self.graphdict[vi]):
                            self.graphdict[vi].append(vj)
                        if (vi not in self.graphdict[vj]):
                            self.graphdict[vj].append(vi)

                    elif ((vi[1] == vj[1]) and (vi[2] == vj[2]) and
                          (vi[1] is not None) and (vi[2] is not None)):
                        # TRANSFER edge
                        if (vj not in self.graphdict[vi]):
                            self.graphdict[vi].append(vj)
                        if (vi not in self.graphdict[vj]):
                            self.graphdict[vj].append(vi)
            
            # Add self-cycle
            for key in self.graphdict.keys():
                if (key[1] is not None) and (key[2] is not None):
                    self.graphdict[key].append(key)
                    self.graphdict[key].sort()

        te = time.time()
        self._graphdictgentime = te - ts


    #
    # Dynamic programming for finding all paths of given length
    #
    def FindPathsOfLengthK(self, vstart, vgoal, k, sols=[], removeinfpaths=True):
        """
        FindPathsOfLengthK returns all paths connecting vstart and
        vgoal which have length k.
        """
        if len(sols) == 0:
            sols.append([vstart])

        # print "sols = {0}".format(sols)
        if (k == 0):
            return []

        if (k == 1):
            if vgoal in self.graphdict[vstart]:
                for path in sols:
                    path.append(vgoal)
                return sols
            else:
                return []

        newsols = []
        ## search in its neighbors
        for v in self.graphdict[vstart]:
            temp_ = copy.deepcopy(sols)
            for path in temp_:
                path.append(v)
                
            # t0 = time.time()
            # temp = self.RemoveInfeasiblePaths(temp_)
            # t1 = time.time()
            # print "removed {0} infeasible paths in {1} sec.".format(len(temp_) - len(temp), t1 - t0)
            
            paths = self.FindPathsOfLengthK(v, vgoal, k - 1, temp_)
            if len(paths) == 0:
                continue

            for path in paths:
                newsols.append(path)
                
        if len(newsols) == 0:
            return []
        else:
            return self.RemoveInfeasiblePaths(newsols)


    def FindPathsOfLengthK2(self, vstart, vgoal, k):
        """
        FindPathsOfLengthK2 divides the problem into 3
        subproblems. First, it finds length-2 paths starting from
        vstart. Then it finds length-2 paths ending at vgoal. Finally,
        it tries to find paths of length k - 4 connecting those paths
        found earlier.
        """
        if k < 4:
            return self.FindPathsOfLengthK(vstart, vgoal, k, [])

        pathsFW = dict()
        pathsBW = dict()

        # Find all 2-step paths from vstart
        for v in self.graphdict:
            resFW = self.FindPathsOfLengthK(vstart, v, 2, [])
            if len(resFW) > 0:
                pathsFW[v] = resFW

        # Find all 2-step paths to vgoal
        for v in self.graphdict:
            resBW = self.FindPathsOfLengthK(v, vgoal, 2, [])
            if len(resBW) > 0:
                pathsBW[v] = resBW

        # Establish connections
        paths = [] # a list of all paths of length k from vstart to vgoal
        for u in pathsFW: # u is a 2-step reachable vertex from vstart
            for v in pathsBW: # vgoal is 2-step reachable from v
                if k == 4:
                    if (u == v):
                        for path1 in pathsFW[u]:
                            for path2 in pathsBW[v]:                                
                                paths.append(path1[:-1] + path2)
                else:
                    res = self.FindPathsOfLengthK(u, v, k - 4, [])
                    if len(res) > 0:
                        for path1 in pathsFW[u]:
                            for path2 in pathsBW[v]:
                                for path in res:
                                    paths.append(path1[:-1] + path + path2[1:])

        return self.RemoveInfeasiblePaths(paths)


    # def FindPathsOfLengthK3(self, vstart, vgoal, k, sols0=[]):
    #     sols = copy.deepcopy(sols0)

    #     if len(sols) == 0:
    #         sols.append([vstart])

    #     if ( (vstart.count(None) == 2) and (vgoal.count(None) == 2) and 
    #          (np.mod(k, 2) == 0) ):
    #         # Both vstart and vgoal are in CP. There is no
    #         # non-redundant path of even length.
    #         return []

    #     if (k == 0):
    #         return []

    #     if (k == 1):
    #         if vgoal in self.graphdict[vstart]:
    #             if len(sols[0]) > 1:
    #                 # sols contains paths of length more than 1. Now
    #                 # we need to check whether we can add vgoal to
    #                 # each path.
    #                 newsols = []
    #                 for path in sols:
    #                     curindex = -1
    #                     curnode = path[curindex]
    #                     prevnode = path[curindex - 1]
    #                     while curnode == prevnode:
                            
    #             else:
    #                 for path in sols:
    #                     path.append(vgoal)
    #                 return sols
    #         else:
    #             return []
                
        

    def RemoveInfeasiblePaths(self, paths0):
        TRANSIT = 0
        TRANSFER = 1
        """
        Parameters
        ----------
        paths0 : a list
            A list containing manipulation paths of length k.

        Returns
        -------
        sols : a list
            A list containing non-redundant manipulation paths of 
            length k.
            Note that sols \subset paths0.
        """
        if len(paths0) == 0:
            return []
        paths = copy.deepcopy(paths0)
        l = len(paths[0])
        sols = []

        for path in paths:
            ## examine each path
            OK = True

            for i in xrange(1, l):
                ## examine each segment
                try:
                    curnode = path[i]
                except:
                    print "remove infeasible paths"
                    print paths
                    print
                    raw_input()
                if (i == 1):
                    ## the first segment can be anything
                    prevnode = path[i - 1]
                    if (not (len(prevnode) == len(curnode))):
                        prevtype = TRANSIT
                    else:
                        if (prevnode[0] == curnode[0]):
                            prevtype = TRANSIT
                        else:
                            prevtype = TRANSFER
                    prevnode = curnode
                else:
                    ## otherwise, in order to be valid (not redundant),
                    ## the currenttype has to be different
                    if (not (len(prevnode) == len(curnode))):
                        curtype = TRANSIT
                    else:
                        if ((prevnode[0] == curnode[0]) and 
                            (prevnode[1] == curnode[1]) and 
                            (prevnode[2] == curnode[2])):
                            # Self-loop
                            curtype = np.mod(prevtype + 1, 2)
                        elif (prevnode[0] == curnode[0]):
                            curtype = TRANSIT
                        else:
                            curtype = TRANSFER
                    if (curtype == prevtype):
                        ## got a problem here
                        OK = False
                        break

                    ## if no problem, continue
                    prevtype = curtype
                    prevnode = curnode
            if OK:
                sols.append(path)

        return sols



    def Plot(self, fignum, grid=False, verticesoffset=1):
        offset = verticesoffset # grasp and placement classes start from offset
        V = [v for v in self.graphdict if v.count(None) == 0]
        
        # These sets contain all available grasp and placement classes
        grasp1classes = []
        grasp2classes = []
        placementclasses = []
        
        for v in V:
            if v[0] not in placementclasses:
                placementclasses.append(v[0])
            if v[1] not in grasp1classes:
                grasp1classes.append(v[1])
            if v[2] not in grasp2classes:
                grasp2classes.append(v[2])

        placementclasses.sort()
        grasp1classes.sort()
        grasp2classes.sort()
        
        nplacements = max(placementclasses) + 1
        ngrasp1 = max(grasp1classes) + 1
        ngrasp2 = max(grasp2classes) + 1
        
        if grid:
            pass

        # UNOPTIMIZED CODE
        TF = dict()
        TS1 = dict() # transit (changing grasp 1)
        TS2 = dict() # transit (changing grasp 2)

        for v1 in V:
            for v2 in V:
                if v1 == v2:
                    continue
                if (v1[1:] == v2[1:]):
                    if v1 not in TF:
                        TF[v1] = []
                    if v2 not in TF[v1]:
                        TF[v1].append(v2)
                    if v2 not in TF:
                        TF[v2] = []
                    if v1 not in TF[v2]:
                        TF[v2].append(v1)

                if (v1[0] == v2[0]) and (v1[2] == v2[2]):
                    if v1 not in TS1:
                        TS1[v1] = []
                    if v2 not in TS1[v1]:
                        TS1[v1].append(v2)
                    if v2 not in TS1:
                        TS1[v2] = []
                    if v1 not in TS1[v2]:
                        TS1[v2].append(v1)

                if (v1[0] == v2[0]) and (v1[1] == v2[1]):
                    if v1 not in TS2:
                        TS2[v1] = []
                    if v2 not in TS2[v1]:
                        TS2[v1].append(v2)
                    if v2 not in TS2:
                        TS2[v2] = []
                    if v1 not in TS2[v2]:
                        TS2[v2].append(v1)
                        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for key in TF:
            temp = TF[key]
            ax.plot([temp[0][0], temp[-1][0]], 
                    [temp[0][1], temp[-1][1]], 
                    [temp[0][2], temp[-1][2]])

        for key in TS1:
            temp = TS1[key]
            ax.plot([temp[0][0], temp[-1][0]], 
                    [temp[0][1], temp[-1][1]], 
                    [temp[0][2], temp[-1][2]])

        for key in TS1:
            temp = TS2[key]
            ax.plot([temp[0][0], temp[-1][0]], 
                    [temp[0][1], temp[-1][1]], 
                    [temp[0][2], temp[-1][2]])

        # Plot vertices
        for v in V:
            ax.scatter(v[0], v[1], v[2])

        ax.set_xlabel('placement index')
        ax.set_ylabel('grasp1 index')
        ax.set_zlabel('grasp2 index')

            
############################################################
#                    Graph Utilities
############################################################
def CreateFWDirectedGraphFromPathsList(pathslist):
    """
    Gnew will contain vertices of the form (pathlevel, isurface,
    appdir) instead of (isurface, appdir)
    """
    Gnew = BimanualGraspPlacementGraph(autogenerate=False)
    k = len(pathslist[0])
    vstart = (0, ) + pathslist[0][0]
    Gnew.graphdict[vstart] = []
    Gnew.verticeslist.append(vstart)

    for path in pathslist:
        for pathlevel in xrange(1, k):
            prevkey = (pathlevel - 1, ) + path[pathlevel - 1]
            curkey = (pathlevel, ) + path[pathlevel]
            
            if (curkey not in Gnew.graphdict[prevkey]):
                Gnew.graphdict[prevkey].append(curkey)
                
            if (curkey not in Gnew.graphdict):
                Gnew.graphdict[curkey] = []
                Gnew.verticeslist.append(curkey)
                
    return Gnew


def CreateBWDirectedGraphFromPathsList(pathslist):
    """
    Gnew will contain vertices of the form (pathlevel, isurface,
    appdir) instead of (isurface, appdir)
    """
    Gnew = BimanualGraspPlacementGraph(autogenerate=False)
    k = len(pathslist[0])
    vgoal = (k - 1, ) + pathslist[0][k - 1]
    Gnew.graphdict[vgoal] = []
    Gnew.verticeslist.append(vgoal)

    for path in pathslist:
        for pathlevel in xrange(k - 1, 0, -1):
            curkey = (pathlevel, ) + path[pathlevel]
            prevkey = (pathlevel - 1, ) + path[pathlevel - 1]
            
            if (prevkey not in Gnew.graphdict[curkey]):
                Gnew.graphdict[curkey].append(prevkey)
                
            if (prevkey not in Gnew.graphdict):
                Gnew.graphdict[prevkey] = []
                Gnew.verticeslist.append(prevkey)
                
    return Gnew


def SaveGraphDict(graphdict, objecthash, graphtype='bgpg', path='../data/'):
    import pickle
    filename = path + objecthash + '.' + graphtype + '.graphdict.pkl' 
    with open(filename, 'wb') as f:
        pickle.dump(graphdict, f, pickle.HIGHEST_PROTOCOL)
    print 'The graphdict has been successfully saved to {0}'.format(filename)


def LoadGraphDict(objecthash, graphtype='bgpg', path='../data/'):
    import pickle
    filename = path + objecthash + '.' + graphtype + '.graphdict.pkl'
    try:
        with open(filename, 'rb') as f:
            graphdict = pickle.load(f)
        print 'The graphdict has been successfully loaded from {0}'.format(filename)
    except:
        print 'Something is wrong while loading the graphdict.'
        print 'Check the information again'
        graphdict = None
    return graphdict
