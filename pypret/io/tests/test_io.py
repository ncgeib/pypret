""" This module tests the io subpackage implementation.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import numpy as np
from pypret import io
from pprint import pformat
from os import remove


class IO1(io.IO):
    x = 1

    def squared(self):
        return self.x * self.x

    def __repr__(self):
        return "IO1(x={0})".format(self.x)


class Grid(io.IO):
    _io_store = ['N', 'dx', 'x0']

    def __init__(self, N, dx, x0=0.0):
        # This is _not_ called upon loading from storage
        self.N = N
        self.dx = dx
        self.x0 = x0
        self._post_init()

    def _post_init(self):
        # this is called upon loading from storage
        # calculate the grids
        n = np.arange(self.N)
        self.x = self.x0 + n * self.dx

    def __repr__(self):
        return "TestIO1(N={0}, dx={1}, x0={2})".format(
                    self.N, self.dx, self.x0)


def test_io():
    # test flat arrays
    _assert_io(np.arange(5))
    _assert_io(np.arange(5, dtype=np.complex128))
    # test nested structures of various types
    _assert_io([{'a': 1.0, 'b': np.uint16(1)}, np.random.rand(10),
                True, None, "hello", 1231241512354134123412353124, b"bytes"])
    _assert_io([[[1]], [[[[1], 2], 3], 4], 5])
    # Test custom objects
    _assert_io(IO1())
    _assert_io(Grid(128, 0.23, x0=-2.3))


def _assert_io(x):
    """ This is slightly hacky: we use pprint to recursively print the objects
    and compare the resulting strings to make sure they are the same. This
    only works as pprint sorts the dictionary entries by their keys before
    printing.

    This requires custom objects to implement __repr__.
    """
    io.save(x, "test.hdf5")
    x2 = io.load("test.hdf5")
    remove("test.hdf5")
    s1 = pformat(x)
    s2 = pformat(x2)
    if s1 != s2:
        print(s1)
        print(s2)
        assert False


if __name__ == "__main__":
    test_io()
