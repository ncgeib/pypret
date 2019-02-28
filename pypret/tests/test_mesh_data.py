""" This module tests the MeshData implementation.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import numpy as np
from pypret.mesh_data import MeshData


def test_mesh_data():
    x = np.linspace(-1.0, 2.0, 100)
    y = np.linspace(2.0, -1.0, 110)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**2 + 0.4 * Y**2 - 1.0

    md = MeshData(Z, x, y, labels=['delay', 'wavelength'],
                  units=['s', 'm'])

    md2 = md.copy()
    md.marginals()
    md.normalize()
    md.autolimit()
    md.limit((0.0, 0.5), (-0.5, 1.0))

    x2 = np.linspace(-1.0, 2.0, 50)
    y2 = np.linspace(2.0, -1.0, 60)
    md.interpolate(y2, x2)

    md.flip()


if __name__ == "__main__":
    test_mesh_data()
