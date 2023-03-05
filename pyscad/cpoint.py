"""@file Control Point convenience functions
"""

# Most operations will act on intervals for normal floats
from typing import Sequence, Optional

# Used for vectorizing functions that operate on CPoints
import functools

# We will use the numpy implementation of IEEE 754 64 bit floats
import numpy as np
import numpy.typing as npt

from .flint import flint, FloatLike

# A control point is a 2-D or 3-D vector of numbers
CPoint = Sequence[FloatLike]

def to_cpts(x: npt.NDArray[FloatLike],
            y: npt.NDArray[FloatLike],
            z: Optional[npt.NDArray[FloatLike]] = None) -> npt.NDArray[CPoint]:
    """Turn arrays of x,y,z components into an array of points
    @param x The x input array
    @param y The y input array
    @param z The z input array
    @return The array of points
    """
    x = np.array(x)
    y = np.array(y)
    if z is None:
        return np.array([x.T, y.T]).T
    else:
        z = np.array(z)
        return np.array([x.Y, y.T, z.T]).T

def to_comp(pts: npt.NDArray[CPoint]) -> Sequence[npt.NDArray[FloatLike]]:
    """Turn arrays of points into arrays of x,y,z components
    @param pts The input array of points
    @return A sequence of arrays each component
    """
    t_pts = pts.T
    if len(t_pts) == 2:
        return t_pts[0].T, t_pts[1].T
    elif len(t_pts) == 3:
        return t_pts[0].T, t_pts[1].T, t_pts[2].T
    else:
        raise ValueError('The input array should contain 2-D or 3-D points')

def cp_mag(x: CPoint) -> FloatLike:
    """Get the magnitude of a control point vector
    @param x The 2 or 3 dimensional control point vector
    @return A scalar corresponding to the magnitude x vector
    """
    sqr_sum = np.sum(x*x, axis=-1)
    # Some work is done by hand to have this smoothly handle array inputs
    # If a single CPoint is fed in, we return a flint or float magnitude
    if isinstance(sqr_sum, flint):
        return sqr_sum.sqrt()
    elif isinstance(sqr_sum, (float, int)):
        return np.sqrt(sqr_sum)
    # If a numpy array of CPoints is fed in we have to apply to sqrt to all of them
    else:
        shape = sqr_sum.shape
        n = 1
        for dim in shape:
            n *= dim
        mag = sqr_sum.reshape((n,))
        for i in range(n):
            if isinstance(mag[i], flint):
                mag[i] = mag[i].sqrt()
            else:
                mag[i] = np.sqrt(mag[i])
        return mag.reshape(shape)

def cp_unit(x: CPoint) -> CPoint:
    """Get a unit vector of a control point vector
    @param x The 2 or 3 dimensional control point vector
    @return A unit vector in the same direction as the x vector
    """
    xmag = cp_mag(x)
    # This gives a little help to the broadcasting so we can do element wise
    # multiplication/division with CPoint vectors and scalars
    if len(x.shape) > 1:
        sh = list(xmag.shape)+[1]
        return x/xmag.reshape(sh)
    else:
        return x/xmag

def cp_vectorize(_func = None, *, ignore = ()):
    """Vectorize functions that have float arguments and return CPoints
    @param ignore, a tuple of positional arguments to not vectorize
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args):
            # First cast all non-ignored arguments to numpy arrays
            ign = tuple([0] + [i+1 for i in list(ignore)])
            vec_args = [arg if i in ign else np.array(arg) for i, arg in enumerate(args)]
            # Confirm vector arguments are all the same shape
            sh = None
            for i, varg in enumerate(vec_args):
                if i not in ign:
                    if sh is None:
                        sh = varg.shape
                    else:
                        if sh != varg.shape:
                            raise ValueError("All vectorized arguments must have the same shape")
            if sh == ():
                # For scalar just call the function
                return func(*vec_args)
            else:
                # For numpy arrays, reshape into a 1-D array
                num = 1
                for dim in sh:
                    num *= dim
                for i, varg in enumerate(vec_args):
                    if i not in ign:
                        vec_args[i] = vec_args[i].reshape((num,))
                # Get the value of the first value in the numpy array
                vargs = [varg if i in ign else varg[0] for i, varg in enumerate(vec_args)]
                first = func(*vargs)
                if isinstance(first, np.ndarray):
                    cl = len(first)
                    res = np.empty((num,cl), dtype=first.dtype)
                else:
                    dtype = flint if isinstance(first, flint) else float
                    res = np.empty((num,), dtype=dtype)
                res[0] = first
                # Loop over all other values in the numpy array
                for j in range(1,num):
                    vargs = [varg if i in ign else varg[j] for i, varg in enumerate(vec_args)]
                    res[j] = func(*vargs)
                # Then reshape the output to look like the input
                if isinstance(first, np.ndarray):
                    return res.reshape(list(sh)+[cl])
                else:
                    return res.reshape(sh)
        return wrapper
    if _func is None:
        return decorator
    else:
        return decorator(_func)

