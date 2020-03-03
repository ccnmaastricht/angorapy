from scipy import linalg
import numpy as np


def pearson_correlation(x, y):

    dtype = type(1.0 + x[0] + y[0])

    xmean = x.mean(dtype=dtype)
    ymean = y.mean(dtype=dtype)

    # By using `astype(dtype)`, we ensure that the intermediate calculations
    # use at least 64 bit floating point.
    xm = x.astype(dtype) - xmean
    ym = y.astype(dtype) - ymean

    # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
    # scipy.linalg.norm(xm) does not overflow if xm is, for example,
    # [-5e210, 5e210, 3e200, -3e200]
    normxm = linalg.norm(xm)
    normym = linalg.norm(ym)

    r = np.dot(xm/normxm, ym/normym)

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = max(min(r, 1.0), -1.0)

    return r
