import numpy as np
from . import lib


def autocorrelation(pulse, tau=None, collinear=False):
    ''' Calculates the intensity or second-order autocorrelation G2(tau).

    Parameters
    ----------
    pulse : Pulse instance
        The pulse of which the autocorrelation is calculated
    tau : 1d-array, optional
        The delays at which the autocorrelation is evaluated. If `None`
        the temporal grid `pulse.t` of the pulse object is used.
    collinear : bool, optional
        Calculates the collinear autocorrelation, with background and higher
        frequency terms. Otherwise only the non-collinear intensity
        autocorrelation without background is calculated. Default is `False`.

    Returns
    -------
    tau : 1d-array
        The delay axis at which the autocorrelation was evaluated.
    ac : 1d-array
        The autocorrelation signal.

    Notes
    -----
    Calculates the following expression::

        G2(tau) = int | [E(t-tau) + E(t)]^2 |^2 dt

    where E(t) is the real-valued electric field.
    This expression is expanded in terms of the complex-valued pulse envelope
    and then evaluated with help of the convolution theorem and the Fourier
    transform. Terms containing oscillations in t, e.g., exp(2j w0 t), are
    neglected.
    Specifically, it implements Eq. (9.7) on page 460 from [Diels2006]_.
    '''
    ft, ift = pulse.ft.forward, pulse.ft.backward
    ift_at = pulse.ft.backward_at

    # if the delays are not provided, use the pulse grid
    fft = False
    if tau is None:
        fft = True  # in this case we can use the FFT directly
        tau = pulse.t

    if collinear:
        f = pulse.field
        i = lib.abs2(f)  # intensity
        if fft:
            # uses the n log(n) FFT
            a0 = ift(lib.abs2(ft(pulse.intensity)))
            a1 = ift(np.real(ft(f).conj() * ft(f * i)))
            a2 = ift(lib.abs2(ft(f * f)))
        else:
            # uses the n^2 DFT
            a0 = ift_at(lib.abs2(ft(pulse.intensity)), tau)
            a1 = ift_at(np.real(ft(f).conj() * ft(f * i)), tau)
            a2 = ift_at(lib.abs2(ft(f * f)), tau)

        ac = (  4.0 * np.real(a0)  # A0
              + pulse.dt * (i * i).sum() / np.pi  # background
              + 8.0 * np.real(a1 * np.exp(1.0j * pulse.w0 * tau))  # A1
              + 2.0 * np.real(a2 * np.exp(2.0j * pulse.w0 * tau)) )  # A2
        # scale for classic 8:1 ratio
        ac *= 8.0 / np.max(ac)
    else:
        i = pulse.intensity
        ac_omega = 4.0 * lib.abs2(ft(i))
        if fft:
            ac = ift(ac_omega).real
        else:
            ac = ift_at(ac_omega, tau).real
        # normalize
        ac /= np.max(ac)

    return tau, ac
