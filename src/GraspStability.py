import openravepy as orpy
import numpy as np
from scipy.spatial import ConvexHull
from cdd import RepType


def GetLocalContactPoints(robot, manip_object):
    """
    GetLocalCOntactPoints returns contact points described in the
    object coordinate.

    This function is tailored for parallel jaw grippers. The returned
    set of contact points consists of two subsets; each subset for
    each contact surface.
    """
    
    # Disable OpenRAVE grasping for collicion checking purpose
    isgrabbing = robot.IsGrabbing(manip_object) is not None
    if isgrabbing:
        robot.Release(manip_object)
    
    collisionreport = orpy.CollisionReport()

    # In order to get contact points, we need to use PQP collision checker
    env = robot.GetEnv()
    collisionchecker = env.GetCollisionChecker()
    if not (collisionchecker.GetXMLId() == 'pqp'):
        env.CheckCollision(robot, manip_object, report=collisionreport)
        points = [c.pos for c in collisionreport.contacts]
        if len(points) < 1:
            raise Exception('No contact detected')
    else:
        # The previously set collisionchecker is not PQP
        pqpcollisionchecker = orpy.RaveCreateCollisionChecker(env, 'pqp')
        env.SetCollisionChecker(pqpcollisionchecker)
        
        env.CheckCollision(robot, manip_object, report=collisionreport)
        points = [c.pos for c in collisionreport.contacts]
        if len(points) < 1:
            raise Exception('No contact detected')

        # Change the previous checker back
        env.SetCollisionChecker(collisionchecker)

    if isgrabbing:
        robot.Grab(manip_object)
        
    # Create a convex hull with specified precision. This is for
    # removing redundant contact points.
    hull = ConvexHull(points, qhull_options='E0.001')

    # Obtain filtered contact points
    contactpoints = [points[index] for index in hull.vertices]

    Tcom = manip_object.GetTransform()
    Tcom[0:3, 3] = manip_object.GetCenterOfMass()
    Tcominv = InverseTransformation(Tcom)

    # Local descriptions of contact points
    localcontactpoints = [np.dot(Tcominv, np.append(p, 1))[0:3] for p in contactpoints]

    manip = robot.GetActiveManipulator()
    Tmanip_local = np.dot(Tcominv, manip.GetTransform())
    pmanip_local = Tmanip_local[0:3, 3]
    lateral = Tmanip_local[0:3, 1]
    
    surf1 = []
    surf2 = []
    for lcp in localcontactpoints:
        d = lcp - pmanip_local
        if np.dot(d, lateral) > 0:
            surf1.append(lcp)
        else:
            surf2.append(lcp)

    return [surf1, surf2]


def InverseTransformation(T):
    R = np.array(T[0:3, 0:3]).T
    p = -np.dot(R, T[0:3, 3])
    return np.vstack([np.hstack([R, np.reshape(p, (3, 1))]), 
                      np.array([0, 0, 0, 1])])
    

def SkewFromVect(w):
    return np.array([[    0, -w[2],  w[1]],
                     [ w[2],     0, -w[0]],
                     [-w[1],  w[0],     0]])


def TransformPoint(T, p):
    """
    TransformPoint transforms a 3D point according to the
    transformation T.
    """
    return np.dot(T, np.append(p, 1))[0:3]
