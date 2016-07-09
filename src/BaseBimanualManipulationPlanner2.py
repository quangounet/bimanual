import openravepy as orpy
import numpy as np
import time
import copy
from parabint import utilities as pu

from ObjectPlanner import ObjectPlanner

from pymanip.planners.ObjectTracker import ObjectTracker
from pymanip.utils.Grasp import (pX, pY, pZ, mX, mY, mZ)
from pymanip.utils import ObjectPreprocessing as op
from pymanip.utils import Utils
from pymanip.utils.Utils import Colorize

from toppso3 import Utils as SE3Utils
from toppso3 import lie as Lie
import TOPP

from CCTrajectory import CCTrajectory
import ClosedChainPlanner as CCP


"""
BaseBimanualManipulationPlanner2

- Restructured the planner. Config, Vertex, and Tree are now not
  nested inside the planner.

- Integrated the closed-chain motion planner to plan transfer motions.

"""

############################################################
#                    Global Parameters
############################################################
# Planner parameters
FW = 0
BW = 1
REACHED = 0
ADVANCED = 1
TRAPPED = 2

# Robot indices
ROBOT1 = 0
ROBOT2 = 1

## Mode of motions
CG1 = 1
CG2 = 2
CG1CG2 = 3

CP = 4 ## let CP be such that both robots are in their HOME configs

CG1CP = 5
CG2CP = 6

CG1CG2CP = 7

WORDS = ['', 'CG1', 'CG2', 'CG1CG2', 'CP', 'CG1CP', 'CG2CP', 'CG1CG2CP']
DICTIONARY = dict()
for w in WORDS:
    DICTIONARY[w] = w

# Manipulation trajectory types
# TRANSIT1 = 0
# TRANSFER1 = 1
# TRANSIT2 = 2
# TRANSFER2 = 3
TRANSIT = 0
TRANSFER = 1

EPSILON = 1e-10
DELTA = 0 # for patabint

CONTROLLERTIMEOUT = 5.
CONTROLLERSLEEPDUR = 0.001
OPENRAVEPLANNERNAME = 'birrt'

HOME = np.zeros(6)

############################################################
#                         Config
############################################################
class Config(object):
    """
        Class Config

        Parameters
        ----------
        qrobots : list
            A list containing robot configurations (excluding the grippers).
        tobj : 4x4 transformation matrix
            A transformation matrix of the movable object at the current 
            configuration.
        qgrasps : list
            qgrasps[x] must be in the format [ibox, approachingdir, slidingdir, delta].
            qgrasps[x] must be None if the current configtype is CP.
            ibox : an index of the box being grasped (a link index).
            approachingdir : pX, pY, pZ, mX, mY, or mZ
            slidingdir : pX, pY, or, pZ
            delta : a position of the gripper along the sliding direction.
        configtype
            A type of this composite configuration.
        isurface : int
            An index of the contact surface.
            If isurface is unknown, we need to create a Config via 
            MRRTPlanner.CreateConfig
            in order to specify isurface automatically.
        """

    def __init__(self, qrobots, tobj, qgrasps, configtype, isurface):

        self.qrobots = qrobots
        self.tobj = tobj
        self.qgrasps = qgrasps
        self.type = configtype
        self.isurface = isurface

        self.approachingdirs = []
        for robotindex in xrange(len(qrobots)):
            if qgrasps[robotindex] is None:
                self.approachingdirs.append(None)
            else:
                self.approachingdirs.append(qgrasps[robotindex][1])


    def __str__(self):
        string = "configtype = {0}\n".format(DICTIONARY[WORDS[self.type]])
        string += "qrobot0 = {0}\n".format(self.qrobots[0])
        string += "qrobot1 = {0}\n".format(self.qrobots[1])
        string += "tobj = {0}\n".format(self.qobj)
        string += "qgrasp0 = {0}".format(self.qgrasps[0])
        string += "qgrasp1 = {0}".format(self.qgrasps[1])
        return string


    def Clone(self):
        qrobots = copy.copy(self.qrobots)
        tobj = copy.copy(self.tobj)
        qgrasps = copy.copy(self.qgrasps)
        configtype = copy.copy(self.type)
        isurface = copy.copy(self.isurface)
        return Config(qrobots, tobj, qgrasps, configtype, isurface)


############################################################
#                         Vertex
############################################################
class Vertex(object):
    """
        Class Vertex

        Parameters
        ----------
        config : Config
        parentindex : int
            Index of its parent
        vertextype : FW or BW
            Indicates in which tree the vertex is
        trajectories : list
            A list containing trajectorystrings for two robots
        objecttrajectory : trajectorystring
            This is assigned when the system executes closed chain motions.
        level : int
            The number of levels from the root.
            The root is at level 0.
        trajtype : TRANSIT or TRANSFER
        index : int
            The index of this vertiex in tree.verticeslist.
            This number will be assigned when added to a tree.
        id : tuple
            A vertex id is (pathlevel, isurface, approachingdir).
            id has to contain pathlevel since now the graph will possibly 
            have cycles.
        """    

    def __init__(self, config, vertextype=FW, trajtype=None, level=0):
        self.config = config
        self.parentindex = None
        self.vertextype = vertextype
        self.trajectories = [None, None]
        self.objecttrajectory = None
        self.level = level
        self.trajtype = trajtype
        self.index = 0 ## to be assigned when added to the tree
        self.id = (self.level, self.config.isurface, 
                   self.config.approachingdirs[0], self.config.approachingdirs[1])


    def __str__(self):
        string = "vindex = {0}\n".format(self.index)
        string += str(self.config)
        return string


    def Clone(self):
        vnew = Vertex(self.config.Clone())

        vnew.id = copy.deepcopy(self.id)
        ## float and string are safe
        vnew.parentindex = self.parentindex
        vnew.vertextype = self.vertextype
        vnew.trajectories = [self.trajectories[0], self.trajectories[1]]
        vnew.objecttrajectory = self.objecttrajectory
        vnew.level = self.level
        vnew.trajtype = self.trajtype
        vnew.index = self.index

        return vnew


############################################################
#                          Tree
############################################################
class Tree(object):

    def __init__(self, vroot, treetype=FW):
        self.verticeslist = []
        self.verticeslist.append(vroot)
        self.treetype = treetype
        self.length = 1


    def __len__(self):
        return len(self.verticeslist)


    def __getitem__(self, index):
        return self.verticeslist[index]


    def Clone(self):
        tnew = Tree(None, self.treetype)
        tnew.verticeslist = [v.Clone() for v in self.verticeslist]
        tnew.length = self.length
        return tnew


    def AddVertex(self, parentindex, vnew, trajectories, trajtype, 
                  objecttrajectory=None):
        parent = self.verticeslist[parentindex]
        vnew.parentindex = parentindex
        # vnew.level = parent.level + 1
        vnew.trajectories = trajectories
        vnew.trajtype = trajtype
        vnew.objecttrajectory = objecttrajectory
        vnew.index = self.length
        self.verticeslist.append(vnew)
        self.length += 1


    def GenerateManipulationTrajectory(self, vindex = -1):
        trajslist = []
        trajtypeslist = []
        vertex = self.verticeslist[vindex]
        while (vertex.parentindex != None):
            parent = self.verticeslist[vertex.parentindex]
            trajslist.append(vertex.trajectory)
            trajtypeslist.append(vertex.trajtype)
            vertex = parent

        if (self.treetype == FW):
            trajslist = trajslist[::-1]
            trajtypeslist = trajtypeslist[::-1]

        return [trajslist, trajtypeslist]


    def GenerateObjectTransformationsList(self, vindex = -1):
        tobjlist = []
        vertex = self.verticeslist[vindex]
        while (vertex.parentindex != None):
            parent = self.verticeslist[vertex.parentindex]
            tobjlist.append(vertex.config.tobj)
            vertex = parent
        tobjlist.append(self.verticeslist[0].config.tobj)

        if (self.treetype == FW):
            tobjlist = tobjlist[::-1]

        return tobjlist


############################################################
#            Base Bimanual Manipulation Planner
############################################################
class BaseBimanualManipulationPlanner(object):
    """
    This class implements basic utilities required for bimanual
    manipulation planning.
    """
    def __init__(self, robots, manipulatornames, mobj):
        self.robots = robots
        self._vmaxset = []
        self._amaxset = []
        self.manips = []
        self.taskmanips = []
        self.object = mobj
        # Activate only the first active_dofs joints
        for robotindex, robot in enumerate(self.robots):
            self.manips.append(robot.SetActiveManipulator\
                                   (manipulatornames[robotindex]))
            
            self.taskmanips.append(orpy.interfaces.TaskManipulation(robot))
            
            robot.SetActiveDOFs(self.manips[robotindex].GetArmIndices())
                        
            self._vmaxset.append(robot.GetDOFVelocityLimits()[0:robot.GetActiveDOF()])
            
            self._amaxset.append(robot.GetDOFAccelerationLimits()[0:robot.GetActiveDOF()])
            
        self.env = self.robots[0].GetEnv()
        
        self.nrobots = len(self.robots)
        
        self._print = True
        self._openravemaxiter = 1000 # max iterations for OpenRAVE planners
        self._postprocessing = False # OpenRAVE default postprocessing
        self._timestep = 0.002 # for object tracker

        self._setsampleparameters = False
        self._hasquery = False
        self._hassolution = False
        self._runningtime = 0.0
        self._iterations = 0
        
        self.PreprocessObjectInfo()

        self.ccplanner = CCP.CCPlanner(mobj, self.robots, manipulatornames[0])


    def PreprocessObjectInfo(self):
        """
        Stores information about object's placement & grasp classes
        """
        # S contains stable contact surfaces's transformation frame
        # with respect to its COM (except S[0] which is the relative
        # transformation between COM and the first link's frame). see
        # ObjectPreprocess.py for more detail about S
        self.S = op.PlacementPreprocess(self.object)

        TCOM = np.array(self.S[0])
        # The z-axis of each surface frame is pointing out of the
        # object. However, what we need is a frame pointing into the
        # object (so that the z-axis is aligned with that of the
        # world).
        Toffset = Utils.ComputeTRot(pX, np.pi)
        # transformationset contains transformation T such that when
        # assigning self.object.SetTransform(T), self.object is
        # resting at a stable placement.
        transformationset = [np.dot(Toffset, np.linalg.inv(np.dot(TCOM, T)))
                             for T in self.S[1:]]

        if self._print:
            print Colorize('Processing the object geometries', 'yellow')
        self.boxinfos = []
        for ibox in xrange(len(self.object.GetLinks())):
            ts = time.time()
            boxinfo = op.BoxInfo(self.object, ibox)
            boxinfo.GetPossibleSlidingDirections()
            boxinfo.Preprocess(transformationset)
            te = time.time()
            if self._print:
                print Colorize('  box {0} took {1} sec.'.format(ibox, te - ts), 'yellow')
            boxinfo.env.Destroy()
            self.boxinfos.append(boxinfo)

        
    def CheckGrasp(self, robotindex, qrobot, Tobj, holdobject=False):
        """
        Check whether self.robot at qrobot can correctly grasp the
        object at Tobj.  If holdobject is True and the grasp is valid,
        the robot will still continue to hold the object.
        """
        robot = self.robots[robotindex]
        taskmanip = self.taskmanips[robotindex]

        isgrasping = False
        robot.SetActiveDOFValues(qrobot)
        self.object.SetTransform(Tobj)
        taskmanip.CloseFingers()
        robot.WaitForController(CONTROLLERTIMEOUT)
        if self.env.CheckCollision(robot, self.object):
            isgrasping = True

        # This grabbing is required regardless of holdobject.
        robot.Grab(self.object)
        
        if isgrasping and holdobject:
            return isgrasping
            
        taskmanip.ReleaseFingers()
        robot.WaitForController(CONTROLLERTIMEOUT)
        robot.Release(self.object)
        return isgrasping


    def ComputeGraspConfiguration(self, robotindex, qgrasp, Tobj, qrobot_ref=None):
        """
        Returns a robot configuration (nearest to qrobot_ref) that can
        grasp the object placed at Tobj with a grasp identified by
        qgrasp.
        """
        robot = self.robots[robotindex]
        manip = self.manips[robotindex]
        
        with self.env:
            self.object.SetTransform(Tobj)
            if qrobot_ref is not None:
                robot.SetActiveDOFValues(qrobot_ref)
            else:
                qrobot_ref = robot.GetActiveDOFValues()
            ibox = qgrasp[0]
            Tgripper = Utils.ComputeTGripper(self.object.GetLinks()[ibox].\
                                                 GetTransform(),
                                             qgrasp, self.boxinfos[ibox].extents)
            sol = manip.FindIKSolution(Tgripper, 
                                       orpy.IkFilterOptions.CheckEnvCollisions)
        if sol is None:
            return None

        # checksolution = [qrobot_ref[i]*sol[i] >= 0 for i in xrange(len(sol))]
        # if not np.alltrue(checksolution):
        #     return None
        
        if not self.CheckGrasp(robotindex, sol, Tobj):
            # Invalid grasp
            return None
        return sol

    
    def PlanClosedChainMotion(self, qgrasps, Tinit, Tgoal, ikinit=[], refinit=[]):
        """
        PlanClosedChainMotion plans a closed chain motion for two
        robots holding the object with qgrasps[0] and
        qgrasps[1]. The object starts at Tinit and finishes at Tgoal.
        
        Parameters
        ----------
        qgrasps : list
            A list containing grasp configurations for both robots
            Note : qgrasp = [ibox, approachingdir, slidingdir, delta]
        Tinit : 4x4 object transformation
            The initial transformation of the object
        Tgoal : 4x4 object transformation
            The goal transformation of the object
        ikinit : list of arrays, optional
            A list containing IK solutions for both robots grasping the
            object at Tinit with qgrasp[0] and qgrasp[1]
        refinit : list of arrays, optional
            A list containing robot configurations to be set before solving
            for IK solutions.

        Returns
        -------
        plannerstatus : bool
        cctraj : CCTrajectory
        """
        _funcname = '[PlanClosedChainMotion] '

        # Ungrasp if grasping
        for robotindex in xrange(len(self.robots)):
            self.taskmanips[robotindex].ReleaseFingers()
            while not self.robots[robotindex].GetController().IsDone():
                time.sleep(CONTROLLERSLEEPDUR)
            self.robots[robotindex].Release(self.object)

        self.object.SetTransform(Tinit)

        # Find initial IK solutions (if not provided)
        if len(ikinit) < 2:
            for robotindex in xrange(len(self.robots)):
                # Set reference configuration
                try:
                    qref = refinit[robotindex]
                    self.robots[robotindex].SetActiveDOFValues(qref)
                except:
                    pass
                
                # Compute Tgripper & corresponding IK solution
                ibox = qgrasps[robotindex][0]
                Tgripper = Utils.ComputeTGripper(self.object.GetLinks()[ibox].\
                                                     GetTransform(), 
                                                 qgrasps[robotindex], 
                                                 self.boxinfos[ibox].extents)
                
                sol = self.manips[robotindex].FindIKSolution\
                (Tgripper, orpy.IkFilterOptions.CheckEnvCollisions)

                if sol is None:
                    if self._print:
                        message =  Colorize('IK solution does not exist for robot {0}'.\
                                                format(robotindex), 'red')
                        print _funcname + message
                    return [False, None]

                ikinit.append(sol)

        # Find goal IK solutions
        ikgoal = []
        self.object.SetTransform(Tgoal)
        for robotindex in xrange(len(self.robots)):
            # Set reference configuration
            qref = ikinit[robotindex]

            # Compute Tgripper & corresponding IK solution
            ibox = qgrasps[robotindex][0]
            Tgripper = Utils.ComputeTGripper(self.object.GetLinks()[ibox].\
                                                 GetTransform(), 
                                             qgrasps[robotindex], 
                                             self.boxinfos[ibox].extents)

            sol = self.manips[robotindex].FindIKSolution\
            (Tgripper, orpy.IkFilterOptions.CheckEnvCollisions)

            if sol is None:
                if self._print:
                    message =  Colorize('goal IK solution does not exist for robot {0}'.\
                                            format(robotindex), 'red')
                    print _funcname + message
                return [False, None]

            ikgoal.append(sol)

        self.object.SetTransform(Tinit)
                        
        for robotindex in xrange(len(self.robots)):
            self.robots[robotindex].SetActiveDOFValues(HOME)

        #
        # Actual closed-chain motion planning
        #            
        poseinit = orpy.poseFromMatrix(Tinit)
        se3configinit = CCP.SE3Config(poseinit[:4], poseinit[4:])
        cccinit = CCP.CCConfig(se3configinit, ikinit, qgrasps, FW)
        ccvinit = CCP.CCVertex(cccinit, FW)
        
        posegoal = orpy.poseFromMatrix(Tgoal)
        se3configgoal = CCP.SE3Config(posegoal[:4], posegoal[4:])
        cccgoal = CCP.CCConfig(se3configgoal, ikgoal, qgrasps, FW)
        ccvgoal = CCP.CCVertex(cccgoal, FW)

        query = CCP.CCQuery(ccvinit, ccvgoal)
        upperlimits = [0.6, 0.25, 1.5]
        lowerlimits = [0.28, -0.35, 0.726]
        query.SetTranslationalLimits(upperlimits, lowerlimits)

        res = self.ccplanner.Solve(query, 600)

        # Extract the solution
        Q = self.ccplanner._query
        if not Q.solved:
            if self._print:
                message =  Colorize('Closed-chain motion planning failed', 'red')
                print _funcname + message
            return [False, None]

        rottrajslist = Q.ExtractFinalRotationalTrajectoryList()
        rotmatriceslist = Q.ExtractFinalRotationMatricesList()
        transtrajslist = Q.ExtractFinalTranslationalTrajectoryList()

        transtrajstringslist = [str(transtraj) for transtraj in transtrajslist]

        lietraj = Lie.LieTraj(rotmatriceslist, rottrajslist)
        transtraj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString\
        (SE3Utils.TrajStringFromTrajList(transtrajstringslist))

        timestampslist = Q.ExtractFinalTimeStampsList()
        waypointsdict = Q.ExtractFinalWaypointsDict()

        timestamps = CCP.MergeTimeStampsList(timestampslist)
        lwp = CCP.MergeWaypointsList(waypointsdict[0])
        rwp = CCP.MergeWaypointsList(waypointsdict[1])

        cctraj = CCTrajectory(lietraj, transtraj, lwp, rwp, timestamps, qgrasps)
        
        # Shortcut the closed-chain trajectory
        maxiter = 50
        cctraj_shortcut = self.ccplanner.CCShortcut(cctraj, maxiter)

        return [True, cctraj_shortcut]


    #
    # Extension & connection utilities
    #
    def Distance(self, c0, c1, metrictype=1):
        if (metrictype == 1):
            delta_qrobot0 = c1.qrobot[0] - c0.qrobot[0]
            dist_qrobot0 = np.dot(delta_qrobot0, delta_qrobot0)

            delta_qrobot1 = c1.qrobot[1] - c0.qrobot[1]
            dist_qrobot1 = np.dot(delta_qrobot1, delta_qrobot1)
            
            ## distance in SE(3)
            p0 = c0.tobj[0:3, 3]
            p1 = c1.tobj[0:3, 3]
            R0 = c0.tobj[0:3, 0:3]
            R1 = c1.tobj[0:3, 0:3]
            
            delta_x = p1 - p0
            rvect = logvect(np.dot(R0.T, R1))

            """
            Object's rotation costs more since the robot needs to move
            a lot more when the object rotates than when it translates.
            """

            wt = 0.2 ## weight for translational distance
            wr = 0.8 ## weight for rotational distance
            
            dist_tobj = wt*np.dot(delta_x, delta_x) + wr*np.dot(rvect, rvect)
            
            return dist_qrobot0 + dist_qrobot1 + dist_tobj

        else:
            raise InputError('metrictype', metrictype)

        
    def NearestNeighborIndices(self, c_rand, treetype, newnn = None):     
        if (treetype == FW):
            vlist = self.treestart.verticeslist
        else:
            vlist = self.treeend.verticeslist
        nv = len(vlist)

        distancelist = [self.Distance(c_rand, v.config, self.metrictype) for v in vlist]
        distanceheap = Heap.Heap(distancelist)

        if (newnn == None):
            if (self.nn < 0):
                nn = nv
            else:
                nn = min(self.nn, nv)
        else:
            if (newnn < 0):
                nn = newnn
            else:
                nn = min(newnn, nv)

        nnindices = [distanceheap.ExtractMin()[0] for i in range(nn)]
        return nnindices








    ################################################################################

    #
    # Local planners
    #
    def Steer(self, robotindex, qinit, qgoal, nmaxiterations, postprocessing=False):
        """
        Steer is a local planner that plans a path between qinit and
        qgoal.
        Currecnt implementation uses OpenRAVE RRT.

        Parameters
        ----------
        robotindex : int
            The index of the robot for which we plan the motion for
        qinit : n-vector
        qgoal : n-vector
        nmaxiterations : int
        postprocessing : bool, optional
            If True, OpenRAVE planner will run parabolic smoothing on 
            the path obtained. Otherwise, OpenRAVE planner will just 
            returns waypoints.

        Returns
        -------
        plannerstatus
        openravetraj
        """
        robot = self.robots[robotindex]
        # Prepare planning parameters
        params = orpy.Planner.PlannerParameters()
        params.SetRobotActiveJoints(robot)
        params.SetInitialConfig(qinit)
        params.SetGoalConfig(qgoal)
        extraparams = ''
        extraparams += '<_nmaxiterations>' + str(nmaxiterations) + '</_nmaxiterations>'
        if not postprocessing:
            # Force no post-processing
            extraparams += '<_postprocessing></_postprocessing>'
        params.SetExtraParameters(extraparams)
        
        with self.env:
            # Intialize openrave planner
            planner = orpy.RaveCreatePlanner(self.env, OPENRAVEPLANNERNAME)
            planner.InitPlan(robot, params)

            # Create an empty trajectory
            openravetraj = orpy.RaveCreateTrajectory(self.env, '')

            # Start planning        
            plannerstatus = planner.PlanPath(openravetraj)
        
        return [plannerstatus, openravetraj]


    def _PlanTransitTrajectory(self, robotindex, vinit, vgoal):
        """
        PlanTransitTrajectory plans a one-step transit trajectory from
        vstart to vgoal.
        
        Parameters
        ----------
        vstart : Vertex
        vgoal : Vertex

        Returns
        -------
        plannerstatus
        trajectorystring : TOPP format
        """
        qinit = vinit.config.qrobots[robotindex]
        qgoal = vgoal.config.qrobots[robotindex]
        Tobj = vinit.config.tobj
        return self.PlanTransitTrajectory(robotindex, qinit, Tobj, qgoal=qgoal)


    def _PlanTransferTrajectory(self, robotindex, vinit, vgoal):
        """
        PlanTransferTrajectory plans a one-step transfer trajectory
        from vstart to vgoal.
        
        Parameters
        ----------
        vstart : Vertex
        vgoal : Vertex

        Returns
        -------
        plannerstatus
        trajectorystring : TOPP format
        """
        qinit = vinit.config.qrobots[robotindex]
        qgrasp = vinit.config.qgrasps[robotindex]
        qgoal = vgoal.config.qrobots[robotindex]
        Tobj = vinit.config.tobj
        return self.PlanTransferTrajectory(robotindex, qinit, qgrasp, Tobj, qgoal=qgoal)


    def PlanTransitTrajectory(self, robotindex, qinit, T, qgrasp_goal=None, qgoal=None):
        """
        PlanTransitTrajectory plans a one-step transit trajectory from
        qinit to qgoal.
        
        Parameters
        ----------
        qinit : n-vector (initial robot configuration)
        T : 4x4 object transformation
        qgrasp_goal : list; optional
        qgoal : n-vector (goal robot configuration); optional

        Either qgrasp_goal or qgoal has to be provided.

        Returns
        -------
        plannerstatus
        trajectorystring : TOPP format
        """
        if (qgrasp_goal is None) and (qgoal is None):
            print Colorize('[PlanTransitTrajectory] Not enough information;', 
                           'red')
            print Colorize('    Either qgrasp_goal or qgoal needs to be given',
                           'red')
            return [orpy.PlannerStatus.Failed, '']
        
        robot = self.robots[robotindex]
        taskmanip = self.taskmanips[robotindex]
        manip = self.manips[robotindex]

        # Release the grabbed object (if any)
        if (robot.IsGrabbing(self.object) is not None):
            taskmanip.ReleaseFingers()
            while not robot.GetController().IsDone():
                time.sleep(CONTROLLERSLEEPDUR)
            robot.Release(self.object)
        
        # Set up the scene
        self.object.SetTransform(T)
        robot.SetActiveDOFValues(qinit)
        
        # Compute qgrasp_goal if not given
        if (qgrasp_goal is not None) and (qgoal is None):
            ibox = qgrasp_goal[0]
            graspedlink = self.object.GetLinks()[ibox]
            extents = self.boxinfos[ibox].extents
            Tgripper = Utils.ComputeTGripper(graspedlink.GetTransform(),
                                             qgrasp_goal, extents, unitscale=False)
            qgoal = manip.FindIKSolution\
            (Tgripper, orpy.IkFilterOptions.CheckEnvCollisions)
            if qgoal is None:
                print Colorize('[PlanTransitTrajectory] qgrasp_goal not feasible',
                               'red')
                return [orpy.PlannerStatus.Failed, '']
            
        # Plan a path using Steer
        [plannerstatus, transittraj] = self.Steer(robotindex, qinit, qgoal,
                                                  self._openravemaxiter,
                                                  self._postprocessing)
        
        if not (plannerstatus == 1):
            if self._print:
                print '[PlanTransitTrajectory]',
                print Colorize('OpenRAVE planner failed', 'red')
            return [plannerstatus, '']

        if self._print:
            print '[PlanTransitTrajectory]',
            print Colorize('Successful', 'green')
        
        # Convert the OpenRAVE trajectory to RampsListND
        rampslist = pu.ConvertOpenRAVETrajToRampsListND(transittraj,
                                                        self._vmaxset[robotindex], 
                                                        self._amaxset[robotindex], DELTA)
        
        return [plannerstatus, str(rampslist)]


    def PlanTransferTrajectory(self, robotindex,  qinit, qgrasp, Tinit, 
                               qgoal=None, Tgoal=None):
        """
        PlanTransferTrajectory plans a one-step transfer trajectory from
        qinit to qgoal.
        
        Parameters
        ----------
        qinit : n-vector (initial robot configuration)
        qgrasp : list
        Tinit : 4x4 object transformation
        qgoal : n-vector; optional
        Tgoal : 4x4 object transformation; optional

        Either qgoal or Tgoal has to be provided

        Returns
        -------
        plannerstatus
        trajectorystring : TOPP format
        """
        if (qgoal is None) and (Tgoal is None):
            print Colorize('[PlanTransferTrajectory] Not enough information;', 
                           'red')
            print Colorize('    Either qgoal or Tgoal needs to be given',
                           'red')
            return [orpy.PlannerStatus.Failed, '']

        robot = self.robots[robotindex]
        taskmanip = self.taskmanips[robotindex]
        manip = self.manips[robotindex]

        # Release the grabbed object (if any)
        if (robot.IsGrabbing(self.object) is not None):
            taskmanip.ReleaseFingers()
            while not robot.GetController().IsDone():
                time.sleep(CONTROLLERSLEEPDUR)
            robot.Release(self.object)

        # Compute qgoal if not given
        if (qgoal is None) and (Tgoal is not None):
            ibox = qgrasp[0]
            graspedlink = self.object.GetLinks()[ibox]
            extents = self.boxinfos[ibox].extents
            self.object.SetTransform(Tgoal)
            Tgripper = Utils.ComputeTGripper(graspedlink.GetTransform(),
                                             qgrasp, extents, unitscale=False)
            
            # Set robot reference configuration before solving for IK solutions
            self.object.SetTransform(Tinit)
            robot.SetActiveDOFValues(qinit)
            qgoal = manip.FindIKSolution\
            (Tgripper, orpy.IkFilterOptions.CheckEnvCollisions)
            if qgoal is None:
                print Colorize('[PlanTransitTrajectory] qgrasp_goal not feasible',
                               'red')
                return [orpy.PlannerStatus.Failed, '']
        
        # Set up the scene
        self.object.SetTransform(Tinit)
        robot.SetActiveDOFValues(qinit)
            
        # Grab the object
        holdobject = True
        isgrasping = self.CheckGrasp(robotindex, qinit, Tinit, holdobject)
        if not isgrasping:
            print '[PlanTransferTrajectory]',
            print Colorize('G R A S P  E R R O R', 'red')
            raw_input() # pause

        # Plan a path using Steer
        [plannerstatus, transfertraj] = self.Steer(robotindex, qinit, qgoal,
                                                   self._openravemaxiter,
                                                   self._postprocessing)
        
        # Relase the grabbed object
        taskmanip.ReleaseFingers()
        while not robot.GetController().IsDone():
            time.sleep(CONTROLLERSLEEPDUR)
        robot.Release(self.object)
        
        if not (plannerstatus == 1):
            if self._print:
                print '[PlanTransferTrajectory]',
                print Colorize('OpenRAVE planner failed', 'red')            
            return [plannerstatus, '']

        if self._print:
            print '[PlanTransferTrajectory]',
            print Colorize('Successful', 'green')
        
        # Convert the OpenRAVE trajectory to RampsListND
        rampslist = pu.ConvertOpenRAVETrajToRampsListND(transfertraj,
                                                        self._vmaxset[robotindex], 
                                                        self._amaxset[robotindex], DELTA)
        
        return [plannerstatus, str(rampslist)]

