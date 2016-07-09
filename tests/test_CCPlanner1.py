import numpy as np
import openravepy as orpy

from pymanip.utils import Utils
from pymanip.utils import ObjectPreprocessing as prep
from pymanip.utils.Grasp import (pX, pY, pZ, mX, mY, mZ)

# SE3 planning for objects
import os
import sys
sys.path.append('../src/rrtse3/')
import SE3Utils
import lie as Lie

import sys
sys.path.append('../src/')
import BimanualGPGMP as bgpgmp
from BaseBimanualManipulationPlanner import (CG1CG2CP, CP, TRANSIT, TRANSFER)
from GenerateRectangularCage import *
from CCTrajectory import CCTrajectory

import TOPP
import time
import random
import pickle

"""
test_CCPlanner1

plan closed chain motions for the given start and goal closed chain
configurations.
+ shortcutting
"""

################################################################################
# LOADING ALL THE STUFF WE NEED
################################################################################
# Global parameters
viewername = 'qtcoin'
showviewer = 1
collisioncheckername = 'ode'
manipulatorname = 'gripper'
d = 1.072
HOME = np.zeros(6)

# Initialize environment
env = orpy.Environment()
env.SetViewer(viewername)
vw = env.GetViewer()
vw.Show(showviewer)
floor = env.ReadKinBodyXMLFile('../xml/floor.kinbody.xml')
lrobot = env.ReadRobotXMLFile('../xml/denso_sensor_gripper_base.xml')
rrobot = env.ReadRobotXMLFile('../xml/denso_sensor_gripper_base.xml')
labtable = env.ReadKinBodyXMLFile('../xml/labtable.kinbody.xml')

floor.SetTransform(Utils.ComputeTTrans(pZ, -0.001))

XMLData = GenerateRectangularCage(0.3, 0.4, 0.5)
cage = env.ReadKinBodyXMLData(XMLData)

collisionchecker = orpy.RaveCreateCollisionChecker(env, collisioncheckername)
env.SetCollisionChecker(collisionchecker)

# Add objects to env
env.Add(floor)
env.Add(labtable)
env.Add(cage)
env.Add(lrobot, True)
env.Add(rrobot, True)

Utils.DisableGripper(lrobot)
Utils.DisableGripper(rrobot)

# Load IK models
iktype = orpy.IkParameterization.Type.Transform6D
lmanip = lrobot.SetActiveManipulator(manipulatorname)
likmodel = orpy.databases.inversekinematics.InverseKinematicsModel(lrobot, iktype=iktype)
if not likmodel.load():
    likmodel.autogenerate()

rmanip = rrobot.SetActiveManipulator(manipulatorname)
rikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(rrobot, iktype=iktype)
if not rikmodel.load():
    rikmodel.autogenerate()

# Set up the scene
robot_zoffset = Utils.ComputeTTrans(pZ, 0.590) # robot base is 0.59m. above the floor
ltheta = 0
lTrobot = reduce(np.dot, [robot_zoffset, Utils.ComputeTTrans(pY, -d/2), 
                          Utils.ComputeTRot(pZ, ltheta)])
lrobot.SetTransform(lTrobot)

# The transformation of the right robot is from stefan_chair_env.yaml
rquat = np.array([0.001, -0.006, -0.001, 1.000])
rtrans = np.array([-0.005, 1.071, 0.000])
rTrobot_offset = np.eye(4)
rTrobot_offset[0:3, 0:3] = -orpy.rotationMatrixFromQuat(rquat)
rTrobot_offset[0:3, 3] = rtrans
rTrobot = np.dot(lTrobot, rTrobot_offset)
rrobot.SetTransform(rTrobot)

Ttable_offset = np.eye(4)
Ttable_offset[0:3, 0:3] = Utils.ComputeTRot(pZ, np.pi)[0:3, 0:3]
Ttable_offset[0:3, 3] = np.array( [ 0.6385794, 0.470379, 0.1492327 - 0.013])
Ttable = np.dot(lTrobot, Ttable_offset)
labtable.SetTransform(Ttable)

################################################################################
# AN EXAMPLE SOLUTION SEQUENCE OF VERTICES
################################################################################
# CONFIG
class NewConfig(object):

    def __init__(self, qrobots, tobj, qgrasps, isurface):

        self.qrobots = qrobots
        self.tobj = tobj
        self.qgrasps = qgrasps
        self.isurface = isurface

        self.approachingdirs = []
        for robotindex in xrange(len(qrobots)):
            if qgrasps[robotindex] is None:
                self.approachingdirs.append(None)
            else:
                self.approachingdirs.append(qgrasps[robotindex][1])

# VERTEX
class NewVertex(object):
    def __init__(self, config, trajtype=None, level=0):
        self.config = config
        self.parentindex = None
        self.trajectories = [None, None]
        self.objecttrajectory = None
        self.level = level
        self.trajtype = trajtype
        self.index = 0 ## to be assigned when added to the tree
        self.id = (self.level, self.config.isurface, 
                   self.config.approachingdirs[0], self.config.approachingdirs[1])

with open('vertices.pkl', 'rb') as f:
    vertices = pickle.load(f)


################################################################################
# TRYING OUT THE NEW PLANNER
################################################################################
import ClosedChainPlanner as CCP
from ClosedChainPlanner import (FW, BW)

ivstart = 3
ivgoal = 4

vstart = vertices[ivstart]
Tobjstart = vstart.config.tobj
posestart = orpy.poseFromMatrix(Tobjstart)
se3configstart = CCP.SE3Config(posestart[:4], posestart[4:])
cccstart = CCP.CCConfig(se3configstart, vstart.config.qrobots, vstart.config.qgrasps, FW)
ccvstart = CCP.CCVertex(cccstart, FW)

vgoal = vertices[ivgoal]
Tobjgoal = vgoal.config.tobj
posegoal = orpy.poseFromMatrix(Tobjgoal)
se3configgoal = CCP.SE3Config(posegoal[:4], posegoal[4:])
cccgoal = CCP.CCConfig(se3configgoal, vgoal.config.qrobots, vgoal.config.qgrasps, BW)
ccvgoal = CCP.CCVertex(cccgoal, BW)

query = CCP.CCQuery(ccvstart, ccvgoal)
upperlimits = [0.6, 0.25, 1.5]
lowerlimits = [0.28, -0.35, 0.726]
query.SetTranslationalLimits(upperlimits, lowerlimits)



planner = CCP.CCPlanner(cage, [lrobot, rrobot], manipulatorname)

res = planner.Solve(query, 600)

Q = planner._query
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

cctraj = CCTrajectory(lietraj, transtraj, lwp, rwp, timestamps, vstart.config.qgrasps)

"""
maxiter = 50
res = planner.CCShortcut(cctraj, maxiter)


timemult = 3.0
planner.VisualizeCCMotion(lietraj, transtraj, lwp, rwp, timestamps, timemult=timemult)

planner.VisualizeCCMotion(res.lietraj, res.transtraj, res.waypointsdict[0], res.waypointsdict[1], res.timestamps, timemult=2.0)


"""
