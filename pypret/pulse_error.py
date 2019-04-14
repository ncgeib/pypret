""" This module implements testing procedures for retrieval algorithms.
"""
import numpy as np
from scipy import optimize
from . import lib


def pulse_error(E, E0, ft, dot_ambiguity=False,
                spectral_shift_ambiguity=False):
    ''' Calculates the normalized rms error between two pulse spectra while
    taking into account the retrieval ambiguities.

    One step in `optimal_rms_error` (the determination of the initial bracket)
    could probably be more efficient, see [Dorrer2002]_). We use the less
    elegant but maybe more straightforward way of simply sampling the
    range for a bracket that encloses a minimum.

    Parameters
    ----------
    E, E0: 1d-array
        Complex-valued arrays that contain the spectra of the pulses.
        ``E`` will be matched against ``E0``.
    ft : FourierTransform instance
        Performs Fourier transforms on the pulse grid.
    dot_ambiguity : bool, optional
        Takes the direction of time ambiguity into account. Default is
        ``False``.
    spectral_shift_ambiguity : bool, optional
        Takes the spectral shift ambiguity into account. Default is ``False``.
    '''
    test_fields = [[ft.w, E, E0]]
    if spectral_shift_ambiguity:
        # spectrally shift by exactly half the grid size
        Et = ft.backward(E)
        Et *= np.exp(0.5j * ft.N * ft.dw * ft.t)
        test_fields.append([ft.w, ft.forward(Et), E0])
    if dot_ambiguity:
        max_iter = len(test_fields)
        for i in range(max_iter):
            tf = test_fields[i]
            test_fields.append([tf[0], tf[1].conj(), tf[2]])

    best_error = np.inf
    for w, spec1, spec2 in test_fields:
        error, matched = optimal_rms_error(w, spec1, spec2)
        if error < best_error:
            best_error = error
            best_match = matched
    return best_error, best_match


def best_constant_phase(E, E0):
    """ Finds ``c`` with ``|c| = 1`` so that ``sum(abs2(c * y1 - y2))`` is
    minimal.

    Uses an analytic solution.
    """
    A = np.sum(E.conj() * E0)
    c = A / np.abs(A)
    err1 = np.sum(lib.abs2(c * E - E0))
    err2 = np.sum(lib.abs2(-c * E - E0))
    if err2 < err1:
        c = -c
    return c


def optimal_rms_error(w, E, E0):
    """ Calculates the RMS error of two arrays, ignoring scaling, constant
    and linear phase of one of them.

    Formally it calculates the minimal error::

        R = sqrt(|rho * exp(i*(x*a + b)) * y1 - y2|^2 / |y2|^2)

    with respect to rho, a and b. If additionally ``conjugation = True`` then
    the error for conjugate(y1) is calculated and the best transformation of y1
    is also returned.
    """
    # E is rescaled so that the amplitudes match in the least-squares sense
    E = E * lib.best_scale(E, E0)

    # find optimal linear and constant phase
    # determine the frequency spacing
    dw = np.max(np.abs(np.diff(w)))
    # rescale the objective function to make it easier for the optimizer
    scale = 1.0 / np.sqrt(np.sum(lib.abs2(E0)) * E.shape[0])

    def objective(alpha):
        linear = np.exp(1.0j * alpha / dw * w)
        phase0 = best_constant_phase(linear * E, E0)
        cresiduals = (phase0 * linear * E - E0) * scale
        return lib.norm2(cresiduals)

    # find an initial bracket
    alphas = np.linspace(-np.pi, np.pi, 2 * E.shape[0])
    err = np.array([objective(a) for a in alphas])
    idx = np.argmin(err)
    bracket = [
        alphas[max(0, idx - 1)],
        alphas[min(alphas.shape[0] - 1, idx + 1)]
    ]
    # run a bounded optimization on that bracket to obtain high precision
    res = optimize.minimize_scalar(
        objective,
        bounds=bracket,
        method='bounded',
        options=dict(maxiter=100, xatol=1e-10)
    )
    linear = np.exp(1.0j * res.x / dw * w)
    phase0 = best_constant_phase(linear * E, E0)
    E = phase0 * linear * E

    return lib.nrms(E, E0), E
