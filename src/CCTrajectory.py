import numpy as np
import TOPP

# SE3 planning for objects
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/rrtse3/')
import SE3Utils
import lie as Lie

eps = 1e-6

class CCTrajectory(object):
    
    def __init__(self, lietraj, transtraj, lwaypoints, rwaypoints, timestamps, qgrasps):
        self.lietraj = lietraj
        self.transtraj = transtraj
        self.waypointsdict = dict()
        self.waypointsdict[0] = lwaypoints[:]
        self.waypointsdict[1] = rwaypoints[:]
        self.timestamps = timestamps[:]
        self.qgrasps = qgrasps[:]
        
    
    @staticmethod
    def Concatenate(cctrajectorylist):
        lietrajslist = []
        transtrajslist = []
        lwaypointslist = []
        rwaypointslist = []
        timestampslist = []
        qgrasps = cctrajectory[0].qgrasps

        for cctrajectory in cctrajectorylist:
            lietrajslist.append(cctrajectory.lietraj)
            transtrajslist.append(cctrajectory.transtraj)
            lwaypointslist.append(cctrakectpru.waypointsdict[0])
            rwaypointslist.append(cctrajectory.waypointsdict[1])
            timestampslist.append(cctrajectory.timestamps)

        newlietraj = ConcatenateLieTrajectories(lietrajslist)
        newtranstraj = ConcatenateTOPPTrajectories(transtrajslist)
        newlwaypoints = MergeWaypointsList(lwaypointslist)
        newrwaypoints = MergeWaypointsList(rwaypointslist)
        newtimestamps = MergeTimeStampsList(timestampslist)

        return CCTrajectory(newlietraj, newtranstraj, 
                            newlwaypoints, newrwaypoints, newtimestamps, qgrasps)


################################################################################
#                                 UTILITIES
################################################################################

def ConcatenateLieTrajectories(lietrajslist):
    newRlist = None
    newtrajlist = None
    
    for lietraj in lietrajslist:
        # List of rotation matrices
        if newRlist is None:
            newRlist = []
        else:
            # Check soundness
            Rdist = SE3Utils.SO3Distance(lietraj.Rlist[0], newRlist[-1])
            assert(Rdist <= eps)
            Rlist = lietraj.Rlist[:]
            Rlist.pop(0)
            
        newRlist = newRlist + Rlist

        # List of rotation trajectories
        if newtrajlist is None:
            newtrajlist = []
        else:
            trajlist = lietraj.trajlist[:]

        newtrajlist = newtrajlist + trajlist

    return Lie.LieTraj(newRlist, newtrajlist)


def ConcatenateTOPPTrajectories(topptrajslist):
    newtopptrajstring = None

    for traj in topptrajslist:
        if newtopptrajstring is None:
            newtopptrajstring = ''
            separator = ''
        else:
            separator = '\n'
            
        newtopptrajstring = newtopptrajstring + separator + str(traj)
        
    return TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString(newtopptrajstring)


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
            assert(RobotConfigDistance(W[0][0:6], newwaypoints[-1][0:6]) < eps)
            W.pop(0)

        newwaypoints = newwaypoints + W
        
    return newwaypoints


        
            
