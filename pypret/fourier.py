""" This module implements the Fourier transforms on linear grids.

The following code approximates the continuous Fourier transform (FT) on
equidistantly spaced grids. While this is usually associated with
'just doing a fast Fourier transform (FFT)', surprisingly, much can be done
wrong.

The reason is that the correct expressions depend on the grid location. In
fact, the FT can be calculated with one DFT but in general it requires a prior
and posterior multiplication with phase factors.

The FT convention we are going to use here is the following::

    Ẽ(w) = 1/2pi ∫ E(t) exp(+i w t) dt
    E(t) =       ∫ Ẽ(w) exp(-i t w) dw

where w is the angular frequency. We can approximate these integrals by their
Riemann sums on the following equidistantly spaced grids::

    t_k = t_0 + k Δt, k=0, ..., N-1
    w_n = w_0 + n Δw, n=0, ..., N-1

and define E_k = E(t_k) and Ẽ_n = Ẽ(w_n) to obtain::

    Ẽ_n = Δt/2pi ∑_k E_k exp(+i w_n t_k)
    E_k = Δw     ∑_n Ẽ_n exp(-i t_k w_n).

To evaluate the sum using the FFT we can expand the exponential to obtain::

    Ẽ_n = Δt/2pi exp(+i n t_0 Δw) ∑_k [E_k exp(+i t_k w_0) ] exp(+i n k Δt Δw)
    E_k = Δw     exp(-i t_k w_0)  ∑_n [Ẽ_n exp(-i n t_0 Δw)] exp(-i k n Δt Δw)

Additionally, we have to require the so-called reciprocity relation for
the grid spacings::

          !
    Δt Δw = 2pi / N = ζ     (reciprocity relation)

This is what enables us to use the DFT/FFT! Now we look at the definition of
the FFT in NumPy::

     fft[x_m] -> X_k =     ∑_k exp(-2pi i m k / N)
    ifft[X_k] -> x_m = 1/N ∑_k exp(+2pi i k m / N)

which gives the final expressions::

    Ẽ_n = Δt N/2pi r_n   ifft[E_k s_k  ]
    E_k = Δw       s_k^*  fft[Ẽ_n r_n^*]

    with r_n = exp(+i n t_0 Δw)
         s_k = exp(+i t_k w_0)

where ^* means complex conjugation. We see that the array to be transformed
has to be multiplied with an appropriate phase factor before and after
performing the DFT. And those phase factors mainly depend on the starting
points of the grids: w_0 and t_0. Note also that due to our sign convention
for the FT we have to use ifft for the forward transform and vice versa.

Trivially, we can see that for w_0 = t_0 = 0 the phase factors vanish and
the FT is approximated well by just the DFT. However, in optics these
grids are unusual.
For w_0 = l Δw and t_0 = m Δt, where l, m are integers (i.e., w_0 and t_0
are multiples of the grid spacing), the phase factors can be incorperated
into the DFT. Then the phase factors can be replaced by circular shifts of the
input and output arrays.

This is exactly what the functions (i)fftshift are doing for one specific
choice of l and m, namely for::

    t_0 = -floor(N/2) Δt
    w_0 = -floor(N/2) Δw.

In this specific case only we can approximate the FT by::

    Ẽ_n = Δt N/2pi fftshift(ifft(ifftshift(E_k)))
    E_k = Δw       fftshift( fft(ifftshift(Ẽ_n))) (no mistake!)

We see that the ifftshift _always_ has to appear on the inside. Failure to do
so will still be correct for even N (here fftshift is the same as ifftshift)
but will produce wrong results for odd N.

Additionally you have to watch out not to violate the assumptions for the
grid positions. Using a symmetrical grid, e.g.,::
    x = linspace(-1, 1, 128)
will also produce wrong results, as the elements of x are not multiples of the
grid spacing (but shifted by half a grid point).

The main drawback of this approach is that circular shifts are usually far more
time- and memory-consuming than an elementwise multiplication, especially for
higher dimensions. In fact I see no advantage in using the shift approach at
all. But for some reason it got stuck in the minds of people and you find the
notion of having to re-order the output of the DFT everywhere.

Long story short: here we are going to stick with multiplying the correct
phase factors. The code tries to follow the notation used above.

Literature
----------

.. [1] W. L. Briggs and v. E. Henson, "The DFT: an owners' manual for the
   discrete Fourier transform," (SIAM, 1995).

.. [2] E. W. Hansen, "Fourier transforms: principles and applications," (John
   Wiley & Sons, 2014).

.. [3] L. N. Trefethen and J. A. C. Weideman, "The exponentially convergent
   trapezoidal rule," SIAM Review 56, 385-458 (2014).


Disclaimer
----------

THIS CODE IS FOR EDUCATIONAL PURPOSES ONLY! The code in this package was not
optimized for accuracy or performance. Rather it aims to provide a simple
implementation of the basic algorithms.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import numpy as np
# scipy.fftpack is still faster than numpy.fft (should change in numpy 1.17)
import scipy.fftpack as fft
from . import io
from .lib import twopi, sqrt2pi


class FourierTransform(io.IO):
    """ This class implements the Fourier transform on linear grids.

    Please note again that there are obvious accuracy and performance issues
    with this implementation. For starters, it instantiates at least one new
    array whenever the FT is computed. Additionally, the way the phase factors
    are calculated can lead to rounding errors. Also one should use a faster
    FFT like FFTW for more performance.
    This simple implementation is mainly for educational use.
    """
    _io_store = ['N', 'dt', 'dw', 't0', 'w0']

    def __init__(self, N, dt=None, dw=None, t0=None, w0=None):
        """ Creates conjugate grids and calculates the Fourier transform.

        Parameter
        ---------
        N : int
            Array size
        dt : float, optional
            The temporal grid spacing. If ``None`` will be calculated by the
            reciprocity relation ``dt = 2 * pi / (N * dw)``. Exactly one of
            ``dt`` or ``dw`` has be provided.
        dw : float, optional
            The spectral grid spacing. If ``None`` will be calculated by the
            reciprocity relation ``dw = 2 * pi / (N * dt)``. Exactly one of
            ``dt`` or ``dw`` has be provided.
        t0 : float, optional
            The first element of the temporal grid. If ``None`` will be
            ``t0 = -floor(N/2) * dt``.
        w0 : float, optional
            The first element of the spectral grid. If ``None`` will be
            ``w0 = -floor(N/2) * dw``.
        """
        if dw is None and dt is not None:
            dw = np.pi / (0.5 * N * dt)
        elif dt is None and dw is not None:
            dt = np.pi / (0.5 * N * dw)
        else:
            raise ValueError("Exactly one of the grid spacings has to be "
                             "provided!")

        if t0 is None:
            t0 = -np.floor(0.5 * N) * dt
        if w0 is None:
            w0 = -np.floor(0.5 * N) * dw
        self.N = N
        self.dt = dt
        self.dw = dw
        self.t0 = t0
        self.w0 = w0
        self._post_init()

    def _post_init(self):
        """ Hook to initialize an object from storage.
        """
        # calculate the grids
        n = k = np.arange(self.N)
        self.t = self.t0 + k * self.dt
        self.w = self.w0 + n * self.dw
        # pre-calculate the phase factors
        self._r = np.exp(1.0j * n * self.t0 * self.dw)
        self._s = np.exp(1.0j * self.t * self.w0)

    def forward(self, x):
        """ Calculates the (forward) Fourier transform of ``x``.
        """
        return self.dt * self.N * self._r * fft.ifft(self._s * x) / twopi

    def backward(self, x):
        """ Calculates the backward (inverse) Fourier transform of ``x``.
        """
        return self.dw * self._s.conj() * fft.fft(self._r.conj() * x)


class Gaussian:
    """ This class can be used for testing the Fourier transform.
    """

    def __init__(self, dt, t0=0.0, phase=0.0):
        """ Instantiates a shifted Gaussian function.

        The Gaussian is calculated by::

            f(t) = exp(-0.5 (t - t0)**2 / dt**2) * exp(1.0j * phase)

        Its Fourier transform is

            F(w) = 1/2pi exp(-0.5 ())

        Parameters
        ----------
        dt : float
            The standard deviation of the temporal amplitude distribution.
        t0 : float
            The center of the temporal amplitude distribution.
        phase : float
            The linear phase coefficient of the temporal distribution.
        """
        self.dt = dt
        self.t0 = t0
        self.phase = phase

    def temporal(self, t):
        """ Returns the temporal distribution.
        """
        arg = (t - self.t0) / self.dt
        return np.exp(-0.5 * arg**2) * np.exp(1.0j * self.phase * t)

    def spectral(self, w):
        """ Returns the spectral distribution.
        """
        w = w + self.phase
        arg = w * self.dt
        return (self.dt * np.exp(-0.5 * arg**2) * np.exp(1.0j * self.t0 * w) /
                sqrt2pi)
