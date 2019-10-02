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

     fft[x_m] -> X_k =     ∑_m exp(-2pi i m k / N)
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

Trivially, we can see that for ``w_0 = t_0 = 0`` the phase factors vanish and
the FT is approximated well by just the DFT. However, in optics these
grids are unusual.
For ``w_0 = l Δw`` and ``t_0 = m Δt``, where l, m are integers (i.e., w_0 and
t_0 are multiples of the grid spacing), the phase factors can be
incorperated into the DFT. Then the phase factors can be replaced by circular
shifts of the input and output arrays.

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

Good, more comprehensive expositions of the issues above can be found in
[Briggs1995]_ and [Hansen2014]_. For the reason why the first-order
approximation to the Riemann integral suffices, see [Trefethen2014]_.
"""
import numpy as np
# scipy.fftpack is still faster than numpy.fft (should change in numpy 1.17)
import scipy.fftpack as fft
from . import io
from .lib import twopi, sqrt2pi
_fft_backend = 'scipy'
try:
    import pyfftw
    _fft_backend = 'pyfftw'
except ImportError:
    pass


class FourierTransformBase(io.IO):
    """ This class implements the Fourier transform on linear grids.

    This simple implementation is mainly for educational use.

    Attributes
    ----------
    N : int
        Size of the grid
    dt : float
        Temporal spacing
    dw : float
        Frequency spacing (angular frequency)
    t0 : float
        The first element of the temporal grid
    w0 : float
        The first element of the frequency grid
    t : 1d-array
        The temporal grid
    w : 1d-array
        The frequency grid (angular frequency)
    """
    _io_store = ['N', 'dt', 'dw', 't0', 'w0']

    def __init__(self, N, dt=None, dw=None, t0=None, w0=None):
        """ Creates conjugate grids and calculates the Fourier transform.

        Parameters
        ----------
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
        # TODO: possibly inaccurate for large t0, w0
        self._fr = self.dt * self.N / twopi * np.exp(1.0j * n * self.t0 *
                                                     self.dw)
        self._fs = np.exp(1.0j * self.t * self.w0)
        # complex conjugate of the above
        self._br = np.exp(-1.0j * n * self.t0 * self.dw)
        self._bs = self.dw * np.exp(-1.0j * self.t * self.w0)

    def forward_at(self, x, w):
        """ Calculates the forward Fourier transform of `x` at the
        frequencies `w`.

        This function calculates the Riemann sum directly and has quadratic
        runtime. However, it can evaluate the integral at arbitrary
        frequencies, even if they are non-equidistantly spaced. Effectively,
        it performs a trigonometric interpolation.
        """
        Dnk = self.dt / twopi * np.exp(1.0j * w[:, None] * self.t[None, :])
        return Dnk @ x

    def backward_at(self, x, t):
        """ Calculates the backward Fourier transform of `x` at the
        times `t`.

        This function calculates the Riemann sum directly and has quadratic
        runtime. However, it can evaluate the integral at arbitrary
        times, even if they are non-equidistantly spaced. Effectively,
        it performs a trigonometric interpolation.
        """
        Dkn = self.dw * np.exp(-1.0j * t[:, None] * self.w[None, :])
        return Dkn @ x


# =============================================================================
# Fourier backend selection
# =============================================================================
if _fft_backend == "scipy":
    class FourierTransform(FourierTransformBase):

        def forward(self, x, out=None):
            """ Calculates the (forward) Fourier transform of ``x``.

            For n-dimensional arrays it operates on the last axis, which has
            to match the size of `x`.

            Parameters
            ----------
            x : ndarray
                The array of which the Fourier transform will be calculated.
            out : ndarray or None, optional
                A location into which the result is stored. If not provided or
                None, a freshly-allocated array is returned.
            """
            if out is None:
                out = np.empty(x.shape, dtype=np.complex128)
            out[:] = self._fr * fft.ifft(self._fs * x)
            return out

        def backward(self, x, out=None):
            """ Calculates the backward (inverse) Fourier transform of ``x``.

            For n-dimensional arrays it operates on the last axis, which has
            to match the size of `x`.

            Parameters
            ----------
            x : ndarray
                The array of which the Fourier transform will be calculated.
            out : ndarray or None, optional
                A location into which the result is stored. If not provided or
                None, a freshly-allocated array is returned.
            """
            if out is None:
                out = np.empty(x.shape, dtype=np.complex128)
            out[:] = self._bs * fft.fft(self._br * x)
            return out

elif _fft_backend == "pyfftw":
    class FourierTransform(FourierTransformBase):

        def _post_init(self):
            super()._post_init()
            # do not need the additional N factor
            n = np.arange(self.N)
            self._fr = self.dt / twopi * np.exp(1.0j * n * self.t0 * self.dw)
            # create the aligned arrays
            a = self._field = pyfftw.empty_aligned(self.N, dtype="complex128")
            b = self._spectrum = pyfftw.empty_aligned(self.N,
                                                      dtype="complex128")
            # instantiate the FFTW objects
            self._fft = pyfftw.FFTW(b, a, direction="FFTW_FORWARD")
            self._ifft = pyfftw.FFTW(a, b, direction="FFTW_BACKWARD")

        def forward(self, x, out=None):
            """ Calculates the (forward) Fourier transform of ``x``.

            For n-dimensional arrays it operates on the last axis, which has
            to match the size of `x`.

            Parameters
            ----------
            x : ndarray
                The array of which the Fourier transform will be calculated.
            out : ndarray or None, optional
                A location into which the result is stored. If not provided or
                None, a freshly-allocated array is returned.
            """
            if out is None:
                out = np.empty(x.shape, dtype=np.complex128)
            f, s = self._field, self._spectrum
            if x.ndim == 1:
                # fast code path for single dimension
                f[:] = x
                f *= self._fs
                self._ifft.execute()
                s *= self._fr
                out[:] = s
            else:
                # implicitly work along last axis and return copy
                for idx in np.ndindex(x.shape[:-1]):
                    f[:] = x[idx]
                    f *= self._fs
                    self._ifft.execute()
                    s *= self._fr
                    out[idx] = s
            return out

        def backward(self, x, out=None):
            """ Calculates the backward (inverse) Fourier transform of ``x``.

            For n-dimensional arrays it operates on the last axis, which has
            to match the size of `x`.

            Parameters
            ----------
            x : ndarray
                The array of which the Fourier transform will be calculated.
            out : ndarray or None, optional
                A location into which the result is stored. If not provided or
                None, a freshly-allocated array is returned.
            """
            if out is None:
                out = np.empty(x.shape, dtype=np.complex128)
            f, s = self._field, self._spectrum
            if x.ndim == 1:
                # fast code path for single dimension
                s[:] = x
                s *= self._br
                self._fft.execute()
                f *= self._bs
                out[:] = f
            else:
                # implicitly work along last axis and return copy
                for idx in np.ndindex(x.shape[:-1]):
                    s[:] = x[idx]
                    s *= self._br
                    self._fft.execute()
                    f *= self._bs
                    out[idx] = f
            return out


class Gaussian:
    """ This class can be used for testing the Fourier transform.
    """

    def __init__(self, dt, t0=0.0, phase=0.0):
        """ Instantiates a shifted Gaussian function.

        The Gaussian is calculated by::

            f(t) = exp(-0.5 (t - t0)^2 / dt^2) * exp(1.0j * phase)

        Its Fourier transform is::

            F(w) = dt/sqrt(2pi) exp(-0.5 * (w + phase)^2 * dt^2 +
                                     1j * t0 * w)

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
