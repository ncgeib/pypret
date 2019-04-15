Installation
============

Installation with ``pip`` or ``conda`` is currently neither supported nor
necessary. Just clone the code repository from git::

    git clone https://github.com/ncgeib/pypret.git

and the directory ``pypret`` within contains all the required code of the
package. Either add its location to your PYTHONPATH or copy it in your
working directory.

As the package matures I may add an installer.

Requirements
------------

It requires Python >=3.6 and recent versions of NumPy and SciPy. Furthermore,
it requires ``h5py`` for storage and loading.
Optional dependencies are

    - pyfftw (for faster FFTs)
    - numba (for optimization of some low-level routines)
    - python-magic (to recognize zipped HDF5 files)
