"""
A module that provides mixin classes to enable persistence with HDF5 files.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import h5py
import lzma
import tempfile
import types
import os
from . import handlers
from . import options
try:
    import magic
except ModuleNotFoundError:
    magic = None

VERSION = 1.0


def save(val, path, archive=False, options=options.DEFAULT_OPTIONS):
    ''' Saves an object in an HDF5 file.

    Parameters
    ----------
    val : object
        Any Python value that is made up of storeable instances. Those are
        built-in types, numpy datatypes and types with custom handlers.
    path : str or Path instance
        Save path of the HDF5 file. Existing files will be overwritten!
    archive : bool, optional
        If ``True`` will compress the whole hdf5 file. This is useful when
        dealing with (many) small HDF5 files as those contain significant
        overhead.
    options : HDF5Options instance, optional
        The HDF5 options that will be used for saving. Defaults to the
        global options instance `DEFAULT_OPTIONS`.
    '''
    if archive:
        _savez(val, path, options=options)
    else:
        _save(val, path, options=options)


def _savez(val, path, options=options.DEFAULT_OPTIONS):
    ''' Saves an object in a compressed HDF5 file.
    '''
    # This is not a secure way to write and read from a tempfile.
    # But due to h5py's limitations I did not find a better way.
    tmppath = tempfile.mktemp()
    _save(val, tmppath, options)
    with open(tmppath, 'rb') as f:
        with lzma.LZMAFile(path, 'w') as zf:
            zf.write(f.read())
    os.remove(tmppath)


def _save(val, path, options=options.DEFAULT_OPTIONS):
    ''' Saves an object in an HDF5 file.

    Parameters
    ----------
    val : object
        Any Python value that is made up of storeable instances. Those are
        built-in types, numpy datatypes and types with custom handlers.
    path : str or Path instance
        Save path of the HDF5 file. Existing files will be overwritten!
    options : HDF5Options instance
        The HDF5 options that will be used for saving. Defaults to the
        global options instance `DEFAULT_OPTIONS`.
    '''
    with h5py.File(path, 'w', libver=options.libver, driver=options.driver,
                   **options.kwds) as f:
        f.attrs['__version__'] = VERSION
        handlers.save_to_level(val, f, options, name=None)


def _load(path, obj=None):
    ret = None
    with h5py.File(path, 'r') as f:
        if '__version__' not in f.attrs:
            raise ValueError('File `%s` is not a valid serialized file.' %
                             path)
        if f.attrs['__version__'] != VERSION:
            raise ValueError('File `%s` was created with a different version.'
                             % path)
        # special case: A single dataset stored at the top-level file group.
        # In this case there is no '__class__'-attribute and we call the
        # load function directly on this single dataset.
        keys = list(f.keys())
        if len(keys) == 1:
            ds = f[keys[0]]
            if isinstance(ds, h5py.Dataset):
                ret = handlers.load_from_level(ds, obj=obj)
        # General case: recursively load the object structure from the file
        # starting with the root group.
        if ret is None:
            ret = handlers.load_from_level(f, obj=obj)
    return ret


def _loadz(path, obj=None):
    tmppath = tempfile.mktemp()
    with open(tmppath, 'wb') as f:
        with lzma.LZMAFile(path, 'r') as zf:
            f.write(zf.read())
    ret = load(tmppath, obj=obj)
    os.remove(tmppath)
    return ret


def load(path, obj=None, archive=None):
    ''' Reads a possibly compressed HDF5 file.

    If archive is ``None`` it is retrieved with python-magic.
    '''
    if archive is None:
        if magic is None:
            archive = False  # simply assume non-compressed file - may fail
        else:
            mime = magic.from_file(str(path))
            archive = (mime == 'XZ compressed data')
    if archive:
        return _loadz(path, obj=obj)
    return _load(path, obj=obj)


class IOHandler(handlers.InstanceHandler):
    level_type = 'group'
    instances = []  # instances are filled by MetaIO

    def save(cls, val, level, options, name):
        attrs = val.to_dict()
        for key, value in attrs.items():
            handlers.save_to_level(value, level, options, key)

    def load_from_level(cls, level, obj=None):
        attrs = handlers.DictHandler.load_from_level(level)
        if obj is None:
            # call from_dict on class
            t = cls.get_type(level)
            return t.from_dict(attrs)
        # else call update_from_dict on object
        obj.update_from_dict(attrs)


class MetaIO(type):
    ''' All sub-classes of IO are automatically registered with the default
        IOHandler. This allows easy storage by simply subclassing IO.
    '''
    def __new__(cls, clsname, bases, attrs):
        newclass = super().__new__(cls, clsname, bases, attrs)
        IOHandler.register(newclass)
        return newclass


class IO(metaclass=MetaIO):
    """ Provides an interface for saving to and loading from a HDF5 file.

    This class can be mixed-in to easily add persistence to your existing
    Python classes. By default all attributes of an object will be stored.
    Upon loading these attributes will be loaded and `__init__` will *not*
    be called.

    Often a better way is to store only the necessary attributes by
    giving a list of attribute names in the private attribute `_io_store`.
    Then you have to overwrite the `_post_init()` method that initializes
    your object from these stored attributes. It is usually also be called at
    the end of the original `__init__` and should not mean extra effort.

    Lastly, you can simply overwrite `load_from_dict` to implement a
    completely custom loader.
    """
    # A list of attribute names that are not stored
    _io_store_not = []
    # None or a list of attribute names that are stored exclusively.
    _io_store = None

    def __new__(cls, *args, **kwargs):
        self = super(IO, cls).__new__(cls)
        return self

    def save(self, path, archive=False, options=options.DEFAULT_OPTIONS):
        save(self, path, archive=archive, options=options)

    def save_to_group(self, g, name):
        handlers.save_to_level(self, g, self._io_options, name)

    def to_dict(self):
        # local rename
        io_store = self._io_store
        io_store_not = self._io_store_not
        cls = self.__class__
        # make sure that _io_store and _io_store are stored if they differ from
        # the class defaults
        if io_store is not None:
            if (io_store != cls._io_store and
                    '_io_store' not in io_store):
                io_store = io_store + ['_io_store']  # make copy
            if (io_store_not != cls._io_store_not and
                    '_io_store_not' not in io_store):
                io_store = io_store + ['_io_store_not']  # make copy
        # actually store the attributes
        attrs = dict()
        for key, value in self.__dict__.items():
            if io_store is not None:
                if key not in io_store:
                    continue
            elif (key in io_store_not or key[:2] == '__' or
                    isinstance(value, types.MethodType)):
                continue
            attrs[key] = value
        return attrs

    @classmethod
    def load(cls, path):
        ret = load(path)
        if not isinstance(ret, cls):
            raise ValueError('File does not store instance of class '
                             '`%s` but of `%s`.' % (str(type(ret)), str(cls)))
        return ret

    @classmethod
    def load_from_group(cls, group):
        ret = handlers.load_from_level(group)
        if not isinstance(ret, cls):
            raise ValueError('Group does not store instance of class '
                             '`%s` but of `%s`.' % (str(type(ret)), str(cls)))
        return ret

    @classmethod
    def from_dict(cls, attrs):
        obj = cls.__new__(cls)
        obj.update_from_dict(attrs)
        return obj

    def update_from_dict(self, attrs):
        for key, value in attrs.items():
            setattr(self, key, value)
        self._post_init()

    def _post_init(self):
        """ Hook to initialize an object from storage.
        """
        pass

    def _add_to_storage(self, key):
        if self._io_store is None:
            self._io_store = []
        self._io_store.append(key)

    def update_from_group(self, group):
        handlers.load_from_level(group, obj=self)

    def update(self, path):
        load(path, obj=self)
