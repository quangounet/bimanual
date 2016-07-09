import numpy as np
import cdd

NUMBER_TYPE = 'float'  # 'float' or 'fraction'


class ConeException(Exception):

    def __init__(self, M):
        self.M = M


class NotConeFace(ConeException):

    def __str__(self):
        return "Matrix is not a cone face"


class NotConeSpan(ConeException):

    def __str__(self):
        return "Matrix is not a cone span"


def SpanToFace(R, data_type = 0, numtype = NUMBER_TYPE):
    """
    SpanToFace returns the face matrix A of the span matrix R, that
    is, a matrix such that

        {x = R z, z >= 0} if and only if {A x <= 0}.

    V-representation:
    first column elements -- 0 for ray
                             1 for vertex
    """
    V = np.hstack((np.tile(data_type, (R.shape[1], 1)), R.T))
    
    V_cdd = cdd.Matrix(V, number_type = numtype)
    V_cdd.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(V_cdd)
    H = np.array(P.get_inequalities())
    return H


def FaceToSpan(A, b = None, numtype = NUMBER_TYPE):
    """
    Compute the span matrix F^S of the face matrix F,
    that is, a matrix such that

        {A x <= b} if and only if {x = R z, z >= 0}.

    H-representation:
    matrix: [b, A]
    """
    
    if b is None:
        b = np.zeros((A.shape[0], 1))
        
    try:
        H = np.hstack((b, A))
    except ValueError:
        H = np.hstack((np.reshape(b, (A.shape[0], 1)), A))
        
    H_cdd = cdd.Matrix(H, number_type = numtype)
    H_cdd.rep_type = cdd.RepType.INEQUALITY
    P = cdd.Polyhedron(H_cdd)
    V = np.array(P.get_generators())
    return V


def ExtractMatrix(M, rep_type):
    if (rep_type == cdd.RepType.GENERATOR):
        return M[:, 1:].T
    else:
        return M[:, 0], M[:, 1:]


FractionToFloatArray = np.vectorize(float)
