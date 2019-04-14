import numpy as np
from . import lib


def autocorrelation(pulse, tau=None, background=False, fast=False):
    ''' Calculates the collinear second order autocorrelation G2(tau).

    Parameters
    ----------
    pulse : Pulse instance
        The pulse of which the autocorrelation is calculated
    tau : 1d-array, optional
        The delays at which the autocorrelation is evaluated. If `None`
        the temporal grid of the pulse object is used.
    background : bool, optional
        Includes the background term of the autocorrelation (see Notes).
        Default is `False`.
    fast : bool, optional
        Calculates the whole autocorrelation, with background terms and higher
        frequency terms (see Notes). Corresponds to measuring with a "fast"
        detector. It implies `background=True`. Default is `False`.

    Returns
    -------
    tau : 1d-array
        The delay axis at which the autocorrelation was evaluated.
    ac : 1d-array
        The autocorrelation

    Notes
    -----
    Calculates the following expression::

        G2(tau) = int | [E(t-tau) + E(t)]^2 |^2 dt

    This expression is expanded in terms of the pulse envelope and then
    evaluated with help of the convolution theorem and the Fourier transform.
    Specifically, it implements Eq. (9.7) on page 460 from [Diels]_.
    '''
    ft = pulse.ft.forward
    ift = pulse.ft.backward
    field = pulse.field

    env2 = lib.abs2(field)
    ac = 2.0 * ift(lib.abs2(ft(env2)))
    if fast:
        # fast implies background
        background = True
        f = np.conj(field)
        a1 = 4.0 * ift(np.real(ft(f * env2).conj() * ft(f)))
        a2 = ift(lib.abs2(ft(f * f)))
        exp = np.exp(-1.0j * pulse.w0 * tau)
        ac += np.real(exp * (a1 + exp * a2))
    if background:
        ac += pulse.ft.dt * np.sum(env2 * env2) / lib.tpi

    if tau is not None:
        ac = pulse.ft.backward_at(ft(ac), tau)
    else:
        tau = pulse.t
    ac = np.real(ac)
    # scale for classic 3:1 and 8:1 ratios
    if fast:
        ac *= 8.0 / np.max(ac)
    elif background:
        ac *= 3.0 / np.max(ac)
    else:
        ac /= np.max(ac)

    return tau, ac
