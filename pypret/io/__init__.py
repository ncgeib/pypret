""" A subpackage that provides Python object persistence in HDF5 files.

It was written to make the storage of arbitrary nested Python structures
in the exchangable HDF5 format easy. Its main purpose is to easily add
persistence to existing numerical or data analysis codes.

While the files itself are plain HDF5 and can be read in any language
supporting HDF5, the format is not compatible to Matlab's own file format.
If you are searching for such a solution look at the hdf5storage package.

Usage
-----
The module exports a ``save()`` function that stores arbitrary structures
of Python and NumPy data types. For example

>>> x = {'data': [1, 2, 3], 'xrange': np.arange(5, dtype=np.uint8)}
>>> io.save(x, "test.hdf5")

This function should suffice for most needs as long as only standard types
are used. The ``load()`` function loads these files and restores the structure
and the types of the data:

>>> io.load("test.hdf5")
{'data': [1, 2, 3], 'xrange': array([0, 1, 2, 3, 4], dtype=uint8)}

Custom Objects
---------------
If you are using objects as simple containers without functionality you may
consider using the SimpleNamespace class from the ``types`` module of the
standard library. The advantage is that io knows how to handle it.::

    from types import SimpleNamespace
    a = SimpleNamespace(name="my object", data=np.arange(5))
    a.data2 = np.arange(10)
    copra.save(a)

If your objects are containers with methods but without a custom ``__init__()``
the simplest way is to inherit or mix-in the ``IO`` class::

    class Data(io.IO):
        x = 1

        def squared(self):
            return self.x * self.x

When using the ``IO`` class by default all instance attributes are stored
and loaded. More flexibility can be achieved by specifying ``_io``-attributes
of your custom class.

    _io_store : list of str or None, optional
        Specify the the instance attributes that are stored exclusively. Acts
        as a whitelist. If ``None`` all instance attributes are stored. Default
        is ``None``.
    _io_store_not : list of str or None, optional
        Specify which instance attributes are not stored. Acts as a blacklist.
        If ``None`` no blacklisting is done.

If you want to add attributes to storage you can call the
``_io_add_to_storage(key)`` method on your instance.
The IO class initalizes the instance without calling ``__init__()``. Instead
``__new__()`` is called on the class and afterwards the ``_post_init()``
method which subclasses can implement. A fully working example of a class
is the following (reduced from copra.FourierTransform)::

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

In this example the object can be exactly reproduced upon loading but only
a minimal amount of storage is required.

If you want to implement your own storage interface for a custom object
you should inherit from ``IO`` and implement your own ``to_dict()`` and
``from_dict()`` methods. Look at the implementation of the default in ``IO``
to understand their behavior.


File Format
-----------
The file format this module uses is a straightforward mapping of Python
types to the HDF5 data structure. Dictionaries and objects are mapped to
HDF5 groups, numpy arrays use h5py's type translation.
Iterables are converted to groups by introducing artificial keys of the
type ``idx_%d``. This is rather inefficient which explains why the
module should not be used to store large numerical arrays as a Python list.
To store the type information it uses an HDF5 attribute ``__class__``.
Furthermore, for scalars the attribute ``__dtype__`` and for strings the
attribute ``__encoding__`` are additionally used.

In conclusion, nested structures of Python types stored with this package are
not suitable for exchanging. Dictionaries of numerical data stored with this
package can be easily opened with any program that supports HDF5.
"""
from .handlers import (save_to_level, load_from_level, TypeHandler,
                       InstanceHandler)
from .io import save, load, IO, MetaIO
