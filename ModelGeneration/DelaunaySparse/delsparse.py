import os 
import sys
HOME = "/rds/general/user/dl2119/home/ICLR_Interp"
# HOME = os.environ["PATH_INTP_FOLDER"]
os.environ["PATH_INTP_FOLDER"] = HOME
sys.path.append(HOME + "/")
sys.path.append(HOME + "/ModelGeneration/DelaunaySparse")

# Import the Delaunay Fortran code.
import _delsparse
import numpy as np
# Return the source point indices and weights associated with a set of
# interpolation points. Takes points in row-major (C style) format.
# 
# INPUTS:
#   pts -- 2D Numpy array of float64 points, where each row is one point.
#   q -- 2D numpy array of float64 points where Delaunay predictions
#        are to be made, where each row is one point.
# 
# OUTPUT:
#   (indices, weights) -- Where "indices" is a 2D NumPy array of integers
#      and each row, i, enumerates the indices of rows in "pts" that are
#      the vertices of the simplex containing q[i], and each corresponding
#      row of weights (a 2D NumPy array of float64) provides the convex
#      weights such that q[i] = np.dot(pts[indices[i]], weights[i]).
# 
def _delaunay_simplex(pts, q, allow_extrapolation=True, print_errors=True,
                     parallel=True, pmode=None, chain=None,
                     ibudget=10000, epsilon=2**(-23), check_spacing=False):
    # Enable parallelism.
    if parallel:
        import os
        os.environ["OMP_NESTED"] = "TRUE"
    # Import NumPy.
    import numpy as np
    # Get the predictions from VTdelaunay
    pts_in = np.asarray(pts.T, dtype=np.float64, order="F")
    p_in = np.asarray(q.T, dtype=np.float64, order="F")
    simp_out = np.ones(shape=(p_in.shape[0]+1, p_in.shape[1]), 
                       dtype=np.int32, order="F")
    weights_out = np.ones(shape=(p_in.shape[0]+1, p_in.shape[1]), 
                          dtype=np.float64, order="F")
    error_out = np.ones(shape=(p_in.shape[1],), 
                        dtype=np.int32, order="F")
    if parallel:
        _delsparse.delaunaysparsep(pts_in.shape[0], pts_in.shape[1],
                                  pts_in, p_in.shape[1], p_in, simp_out,
                                  weights_out, error_out, extrap=100.0,
                                  pmode=pmode, ibudget=ibudget,
                                  eps=epsilon, chain=chain,
                                  exact=check_spacing)
    else:
        _delsparse.delaunaysparses(pts_in.shape[0], pts_in.shape[1],
                                  pts_in, p_in.shape[1], p_in, simp_out,
                                  weights_out, error_out, extrap=100.0, 
                                  ibudget=ibudget, eps=epsilon,
                                  chain=chain, exact=check_spacing)
    # Remove "extrapolation" errors if the user doesn't care.
    if allow_extrapolation: error_out = np.where(error_out == 1, 0, error_out)
    else:
        if 1 in error_out:
            class Extrapolation(Exception): pass
            raise(Extrapolation("Encountered extrapolation point when making Delaunay prediction."))
    # Handle any errors that may have occurred.
    if (sum(error_out) != 0):
        if print_errors:
            unique_errors = sorted(np.unique(error_out))
            print(" [Delaunay errors:",end="")
            for e in unique_errors:
                if (e == 0): continue
                indices = tuple(str(i) for i in range(len(error_out))
                                if (error_out[i] == e))
                if (len(indices) > 5): indices = indices[:2] + ('...',) + indices[-2:]
                print(" %3i"%e,"at","{"+",".join(indices)+"}", end=";")
            print("] ")
        # Reset the errors to simplex of 1s (to be 0) and weights of 0s.
        bad_indices = (error_out > (1 if allow_extrapolation else 0))
        simp_out[:,bad_indices] = 1
        weights_out[:,bad_indices] = 0
    # Adjust the output simplices and weights to be expected shape.
    indices  = simp_out.T - 1
    weights = weights_out.T
    # Return the appropriate shaped pair of points and weights
    return (indices, weights)

# Construct a function that converts indices and weights into a real number prediction.
def delaunay_approx(q, points, values):
    """q: prediction locations q.shape = (N, dim). 
        points: underlying points points.shape = (N, dim)
        values: f(points)"""
    q = np.array(q, dtype=float)
    if len(q.shape) == 1:
        inds, wts = _delaunay_simplex(points.copy(), np.reshape(q,(1,len(q))))
        return np.dot(values[inds[0]], wts[0])
    else:
        inds, wts = _delaunay_simplex(points.copy(), q)
        vals = values[inds.flatten()].reshape(wts.shape)
        return np.sum(vals * wts, axis=1)




