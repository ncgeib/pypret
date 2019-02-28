""" This module tests the MeshData implementation.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import numpy as np
import matplotlib.pyplot as plt
from pypret.material import BK7


def test_sellmeier():
    assert abs(BK7.n(500e-9) - 1.5214) < 1e-4
    assert abs(BK7.n(800e-9) - 1.5108) < 1e-4
    assert abs(BK7.n(1200e-9) - 1.5049) < 1e-4


if __name__ == "__main__":
    test_sellmeier()
    wl = np.linspace(300e-9, 1200e-9, 1000)

    fig, ax = plt.subplots()
    ax.plot(wl * 1e9, BK7.n(wl))
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("refractive index")
    ax.set_title("BK7")
    ax.grid()
    fig.tight_layout()
    plt.show()
