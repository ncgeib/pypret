""" Miscellaneous helper functions

These functions fulfill small numerical tasks used in several places in the
package.
"""
import numpy as np
# make numba jit an optional dependence
# see https://github.com/numba/numba/issues/3735
try:
    from numba import jit
except ImportError:
    def jit(pyfunc=None, **kwargs):
        def wrap(func):
            return func
        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap

# Constants for convenience (not more accurate)
# two pi
twopi = 6.2831853071795862
# sqrt(2)
sqrt2 = 1.4142135623730951
# speed of light
sol = 299792458.0

# Constants that give slightly more accuracy (~1 ulp)
# sqrt(2 * pi)
sqrt2pi = 2.5066282746310007


def as_list(x):
    """ Try to convert argument to list and return it.

    Useful to implement function arguments that could be scalar values
    or lists.
    """
    try:
        return list(x)
    except TypeError:
        return list([x])

@jit(nopython=True, cache=True)
def abs2(x):
    """ Calculates the squared magnitude of a complex array.
    """
    return x.real * x.real + x.imag * x.imag


def rms(x, y):
    """ Calculates the root mean square (rms) error between ``x`` and ``y``.
    """
    return np.sqrt(abs2(x - y).mean())


@jit(nopython=True, cache=True)
def norm2(x):
    """ Calculates the squared L2 or Euclidian norm of array ``x``.
    """
    return abs2(x).sum()


@jit(nopython=True, cache=True)
def norm(x):
    """ Calculates the L2 or Euclidian norm of array ``x``.
    """
    return np.sqrt(abs2(x).sum())


def phase(x):
    """ The phase of a complex array."""
    return np.unwrap(np.angle(x))


def nrms(x, y):
    """ Calculates the normalized rms error between ``x`` and ``y``.

    The convention for normalization varies. Here we use::

        max |y|

    as normalization.
    """
    n = np.abs(y).max()
    if n == 0.0:
        raise ValueError("Second array cannot be zero.")
    return rms(x, y) / n


def mean(x, y):
    """ Calculates the mean of the distribution described by (x, y).
    """
    return np.sum(x * y) / np.sum(y)


def variance(x, y):
    """ Calculates the variance of the distribution described by (x, y).
    """
    dx = x - mean(x, y)
    return np.sum(dx * dx * y) / np.sum(y)


def standard_deviation(x, y):
    """ Calculates the standard deviation of the distribution described by
        (x, y).
    """
    return np.sqrt(variance(x, y))


def gaussian(x, x0=0.0, sigma=1.0):
    """ Calculates a Gaussian function with center ``x0`` and standard
        deviation ``sigma``.
    """
    d = (x - x0) / sigma
    return np.exp(-0.5 * d * d)


def rescale(x, window=[0.0, 1.0]):
    """ Rescales a numpy array to the range specified by ``window``.

    Default is [0, 1].
    """
    maxx = np.max(x)
    minx = np.min(x)
    return (x - minx) / (maxx - minx) * (window[1] - window[0]) + window[0]


def marginals(data, normalize=False, axes=None):
    """ Calculates the marginals of the data array.

    axes specifies the axes of the marginals, e.g., the axes on which the
    sum is projected.

    If axis is None a list of all marginals is returned.
    """
    if axes is None:
        axes = range(data.ndim)
    axes = as_list(axes)
    full_axes = list(range(data.ndim))
    m = []
    for i in axes:
        # for the marginal sum over all axes except the specified one
        margin_axes = tuple(j for j in full_axes if j != i)
        m.append(np.sum(data, axis=margin_axes))
    if normalize:
        m = [rescale(mx) for mx in m]
    return tuple(m) if len(m) != 1 else m[0]


def find(x, condition, n=1):
    """ Return the index of the nth element that fulfills the condition.
    """
    search_n = 1
    for i in range(len(x)):
        if condition(x[i]):
            if search_n == n:
                return i
            search_n += 1
    return -1


def best_scale(E, E0):
    """ Scales rho so that::

        sum (rho * |E| - |E0|)^2

    is minimal.
    """
    Eabs, E0abs = np.abs(E), np.abs(E0)
    return np.sum(Eabs * E0abs) / np.sum(Eabs * Eabs)


def arglimit(y, threshold=1e-3, padding=0.0, normalize=True):
    """ Returns the first and last index where `y >= threshold * max(abs(y))`.
    """
    t = np.abs(y)
    if normalize:
        t /= np.max(t)

    idx1 = find(t, lambda x: x >= threshold)
    if idx1 == -1:
        idx1 = 0
    idx2 = find(t[::-1], lambda x: x >= threshold)
    if idx2 == -1:
        idx2 = t.shape[0] - 1
    else:
        idx2 = t.shape[0] - 1 - idx2

    return (idx1, idx2)


def limit(x, y=None, threshold=1e-3, padding=0.25, extend=True):
    """ Returns the maximum x-range where the y-values are sufficiently large.

    Parameters
    ----------
    x : array_like
        The x values of the graph.
    y : array_like, optional
        The y values of the graph. If `None` the maximum range of `x` is
        used. That is only useful if `padding > 0`.
    threshold : float
        The threshold relative to the maximum of `y` of values that should be
        included in the bracket.
    padding : float
        The relative padding on each side in fractions of the bracket size.
    extend : bool, optional
        Signals if the returned range can be larger than the values in ``x``.
        Default is `True`.

    Returns
    -------
    xl, xr : float
        Lowest and biggest value of the range.

    """
    if y is None:
        x1, x2 = np.min(x), np.max(x)
        if not extend:
            return (x1, x2)
    else:
        idx1, idx2 = arglimit(y, threshold=threshold)
        x1, x2 = sorted([x[idx1], x[idx2]])

    # calculate the padding
    if padding != 0.0:
        pad = (x2 - x1) * padding
        x1 -= pad
        x2 += pad

    if not extend:
        x1 = max(x1, np.min(x))
        x2 = min(x2, np.max(x))

    return (x1, x2)


def fwhm(x, y):
    """ Calculates the full width at half maximum of the distribution described
        by (x, y).
    """
    xl, xr = limit(x, y, threshold=0.5, padding=0.0)
    return np.abs(xr - xl)


def edges(x):
    """ Calculates the edges of the array elements.

    Assuming that the input array contains the midpoints of a supposed data
    set, the function returns the (N+1) edges of the data set points.
    """
    diff = np.diff(x)
    reverse = False
    if np.any(np.sign(diff) != np.sign(diff[0])):
        raise ValueError("Input array must be sorted")
    elif diff[0] < 0.0:
        x = x[::-1]
        reverse = True

    result = np.concatenate((
        [1.5 * x[0] - 0.5 * x[1]],
        0.5 * (x[1:] + x[:-1]),
        [1.5 * x[-1] - 0.5 * x[-2]]
    ))
    if reverse:
        result = result[::-1]

    return result


def build_coords(*axes):
    """ Builds a coordinate array from the axes.
    """
    AXES = np.meshgrid(*axes, indexing='ij')
    return np.stack(AXES, axis=-1)


def mask_phase(x, amp, phase, threshold=1e-3):
    mask = (amp / np.max(amp) < threshold)
    blank_phase = np.ma.masked_array(phase, mask=mask)
    blank_x = np.ma.masked_array(x, mask=mask)
    return blank_x, blank_phase

def retrieval_report(res):
    """ Simple helper that prints out important information from the
    retrieval result object.
    """
    print("Retrieval report")
    print("trace error".ljust(15) + "R = %.17e".rjust(25) % res.trace_error)
    if hasattr(res, "trace_error_optimal"):
        print("min. trace error".ljust(15) + "R0 = %.17e".rjust(25) % res.trace_error_optimal)
        print("".ljust(15) + "R - R0 = %.17e".rjust(25) % (res.trace_error - res.trace_error_optimal))
        print()
        print("pulse error".ljust(15) + "ε = %.17e".rjust(25) % res.pulse_error)
