""" This module tests the Fourier implementation.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import numpy as np
from pypret.fourier import FourierTransform, Gaussian
from pypret import lib


def test_gaussian_transformation():
    """ This test compares the numerical approximation of the Fourier transform
    to the analytic solution for a Gaussian function. It uses non-centered
    grids and a non-centered Gaussian on purpose.
    """
    # define the grid parameters
    # choose some arbitrary values to break symmetries
    dt = 0.32
    N = 205
    dw = np.pi / (0.5 * N * dt)
    t0 = -(N//2 + 2.1323) * dt
    w0 = -(N//2 - 1.23) * dw
    # and actually create it
    ft = FourierTransform(N, dt, t0=t0, w0=w0)
    # create and calculate a non-centered Gaussian distribution
    gaussian = Gaussian(10 * dt, 0.1 * t0, 0.12 * w0)
    temporal0 = gaussian.temporal(ft.t)
    spectral0 = gaussian.spectral(ft.w)

    # calculate the numerical approximations
    spectral1 = ft.forward(temporal0)
    temporal1 = ft.backward(spectral0)

    temporal_error = lib.nrms(temporal1, temporal0)
    spectral_error = lib.nrms(spectral1, spectral0)

    # calculate the error (actual error depends on the FFT implementation)
    assert temporal_error < 1e-14
    assert spectral_error < 1e-14


if __name__ == "__main__":
    test_gaussian_transformation()
