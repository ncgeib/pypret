""" Sets the default options for HDF5 data.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import numpy as np
from types import SimpleNamespace as SN


class HDF5Options:
    """ A class that handles the correct HDF5 options for different data sets.

    The reason is simply that native HDF5 compression will actually increase
    the file size for small arrays (< 300 bytes). This class selects different
    HDF5 options based on the dataset over the method ``__call__``. It can
    be subclassed to support more sophisticated selection strategies.
    """
    def __init__(self):
        self.compression_threshold = 300  # bytes
        self.libver = 'latest'
        self.driver = None
        self.kwds = dict()
        self.encoding = 'utf-8'
        self.compressed_dataset = SN(
                compression='gzip', chunks=True, fletcher32=True, shuffle=True,
                compression_opts=9)
        self.dataset = SN(
                compression=None, chunks=None, fletcher32=False, shuffle=False)

    def copy(self):
        ret = HDF5Options()
        ret.compression_threshold = self.compression_threshold
        ret.libver = self.libver
        ret.driver = self.driver
        ret.encoding = self.encoding
        ret.kwds = dict(**self.kwds)
        ret.compressed_dataset = SN(**self.compressed_dataset.__dict__)
        ret.dataset = SN(**self.dataset.__dict__)
        return ret

    def __call__(self, arr):
        ''' Returns the correct dataset creation options for an array.
        '''
        arr = np.asanyarray(arr)
        if arr.nbytes > self.compression_threshold:
            kwargs = self.compressed_dataset.__dict__
        else:
            kwargs = self.dataset.__dict__
        return kwargs


DEFAULT_OPTIONS = HDF5Options()
