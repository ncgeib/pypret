""" Provides a function to generate random pulses with specified TBP.
"""
import numpy as np
import scipy.optimize as opt
from . import lib


def random_pulse(pulse, tbp, edge_value=None, check=True):
    """ Creates a random pulse with a specified time-bandwidth product.

    Parameters
    ----------
    pulse : Pulse instance
    tbp : float
        The specified time-bandwidth product.
    edge_value : float, optional
        The maximal value for the pulse amplitude at the edges of the grid.
        It defaults to the double value epsilon ~2e-16.

    Returns
    -------
    bool : True on success, False if an error occured. The resulting pulse
        is stored in the Pulse instance passed to the function.

    Notes
    -----
    The function creates random pulses by iteratively restricting the bandwidth
    in time and frequency domain. It starts from random complex values in
    frequency domain, multiplies a Gaussian function, transforms in the
    time domain and multiplies a Gaussian function again. The filter functions
    are Gaussians with the specified time-bandwidth product.
    The TBP of the Gaussian filters, however, does not directly correspond
    to the TBP of the resulting pulse. To use this algorithm to generate a
    pulse with exactly the specified TBP, it is run in the range 0.5 * TBP to
    1.5 * TBP using a scalar root search (brentq). Usually this guarantees
    convergence within a few tries.
    The larger the TBP the larger the number of points has to be. So the
    algorithm may fail to find a solution if pulse.N is too small.
    """
    if edge_value is None:
        # this is roughly the roundoff error induced by an FFT
        edge_value = pulse.N * np.finfo(np.double).eps
    # access/calculate some fundamental grid parameters
    t, w = pulse.t, pulse.w
    t1, t2 = t[0], t[-1]
    w1, w2 = w[0], w[-1]
    t0, w0 = 0.5 * (t1 + t2), 0.5 * (w1 + w2)
    log_edge = np.log(edge_value)

    """ Calculate the width of a Gaussian function that drops exactly to
        edge_value at the edges of the grid.
    """
    spectral_width = np.sqrt(-0.125 * (w1 - w2)**2 / log_edge)
    # Now the same in the temporal domain
    max_temporal_width = np.sqrt(-0.125 * (t1 - t2)**2 / log_edge)
    # The actual temporal width is obtained by the uncertainty relation
    # from the specified TBP
    temporal_width = 2.0 * tbp / spectral_width

    if temporal_width > max_temporal_width:
        print("The required time-bandwidth product cannot be reached! "
              "Decrease edge_value or increase pulse.N!")
        return False

    # special case for TBP = 0.5 (transform-limited case)
    if tbp == 0.5:
        phase = np.exp(1.0j * lib.twopi * np.random.rand(pulse.N))
        pulse.spectrum = lib.gaussian(w, w0, spectral_width) * phase
        return True

    # create the filter functions, the scaling by the number of rounds is
    # purely a heuristic
    spectral_filter = lib.gaussian(w, w0, spectral_width)

    """ The algorithm works by iteratively filtering in the frequency and time
        domain. However, the chosen filter functions only roughly give
        the correct TBP. To obtain the exact result we scale the temporal
        filter bandwidth by a factor and perform a scalar minimization on
        that value.
    """
    spectrum = (np.random.rand(pulse.N) *
                np.exp(1j * lib.twopi * np.random.rand(pulse.N)))
    # rough guess for the relative range in which our optimal value lies
    factor_min, factor_max = 0.5, 1.5

    def create_pulse(factor):
        """ This performs the filtering. """
        temporal_filter = lib.gaussian(t, t0, temporal_width * factor)

        pulse.spectrum = spectrum * spectral_filter
        pulse.field = pulse.field * temporal_filter

    def objective(factor):
        """ This function should be zero """
        create_pulse(factor)
        return tbp - pulse.time_bandwidth_product

    # The objective function has to change sign in the bounds we chose
    i = 0
    while np.sign(objective(factor_min)) == np.sign(objective(factor_max)):
        # for some random arrays this condition is not always fulfilled.
        # just try again
        spectrum = (np.random.rand(pulse.N) *
                    np.exp(1j * lib.twopi * np.random.rand(pulse.N)))
        if i == 10:
            # I have never observed this case.
            raise ValueError('Could not create a pulse for these parameters!')

    # actually perform the optimization
    factor = opt.brentq(objective, factor_min, factor_max)
    # and finally create the pulse
    create_pulse(factor)

    # The random pulse is stored in pulse
    return True


def random_gaussian(pulse, fwhm, phase_max=0.1 * np.pi):
    """ Generates a Gaussian pulse with random phase.

    Its pulse of duration is given by ``fwhm``.
    """
    # convert intensity fwhm to field std-dev.
    sigma = 0.5 * fwhm / np.sqrt(np.log(2.0))
    phase = np.exp(1.0j * np.random.uniform(-phase_max, phase_max, pulse.N))
    pulse.field = lib.gaussian(pulse.t, sigma=sigma) * phase
    return True
