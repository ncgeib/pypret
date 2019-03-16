""" Implements functions that handle the serialization of types and classes.

Type handlers store and load objects of exactly that type. Instance handlers
work also work for subclasses of that type.

The instance handlers are processed in the order they are stored. This means
that if an object is an instance of several handled classes it will not raise
an error and will be handled by the first matching handler in the OrderedDict.
"""
import numpy as np
import types
import inspect
from collections import OrderedDict

""" The handler dictionaries are automatically filled when Handler class
    definitions are parsed via the metaclass __new__ function.
"""
# Saver dictionaries: classes as keys, handler classes as values
type_saver_handlers = dict()
instance_saver_handlers = OrderedDict()
# Loader dictionaries: class names as keys, handler classes as values
loader_handlers = dict()


def classname(val):
    """ Returns a qualified class name as string.

    The qualified class name consists of the module and the class name,
    separated by a dot. If an instance is passed to this function, the name
    of its class is returned.

    Parameters
    ----------
    val : instance or class
        The instance or a class of which the qualified class name is returned.

    Returns
    -------
    str : The qualified class name.
    """
    if inspect.isclass(val):
        return ".".join([val.__module__,
                         val.__name__])
    return ".".join([val.__class__.__module__,
                     val.__class__.__name__])


def set_attribute(level, key, value):
    level.attrs[key] = np.string_(value)


def get_attribute(level, key):
    return level.attrs[key].decode('ascii')


def set_classname(level, clsname):
    set_attribute(level, '__class__', clsname)


def get_classname(level):
    return get_attribute(level, '__class__')


def save_to_level(val, level, options, name=None):
    """ A generic save function that dispatches the correct handler.
    """
    t = type(val)
    if t in type_saver_handlers:
        return type_saver_handlers[t].save_to_level(val, level, options, name)
    for i in instance_saver_handlers:
        if isinstance(val, i):
            return instance_saver_handlers[i].save_to_level(val, level,
                                                            options, name)
    raise ValueError("%s of type %s is not supported by any handler!" %
                     (str(val), str(t)))


def load_from_level(level, obj=None):
    """ Loads an object from an HDF5 group or dataset.

    Parameters
    ----------
    level : h5py.Dataset or h5py.Group
        An HDF5 node that stores an object in a valid format.
    obj : instance or None
        If provided this instance will be updated from the HDF5 node instead
        of creating a new instance of the stored object.

    Returns
    -------
    instance of the stored object
    """
    clsname = get_classname(level)
    if clsname not in loader_handlers:
        raise ValueError('Class `%s` has no registered handler.' % clsname)
    handler = loader_handlers[clsname]
    return handler.load_from_level(level, obj=obj)


class TypeRegister(type):
    """ Metaclass that registers a type handler in a global dictionary.
    """
    def __new__(cls, clsname, bases, attrs):
        # convert all methods to classmethods
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):
                attrs[attr_name] = classmethod(attr_value)
        newclass = super().__new__(cls, clsname, bases, attrs)
        # register the class as a handler for all specified types
        for t in newclass.types:
            newclass.register(t)
        return newclass


class InstanceRegister(type):
    """Metaclass that registers an instance handler in global dictionary.
    """
    def __new__(cls, clsname, bases, attrs):
        # convert all methods to classmethods
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):
                attrs[attr_name] = classmethod(attr_value)
        newclass = super().__new__(cls, clsname, bases, attrs)
        # register the class as a handler for all specified instances
        for t in newclass.instances:
            newclass.register(t)
        return newclass


class Handler:
    # a default for subclasses (only `group` subclasses have to overwrite)
    level_type = 'dataset'

    @classmethod
    def save_to_level(cls, val, level, options, name):
        """ A generic wrapper around the custom save method that each
            handler implements. It creates a dataset or a group depending
            on the `level_type` class attribute and sets the `__class__`
            attribute correctly.
            For more flexibility subclasses can overwrite this method.
        """
        # get the qualified class name of the object to be saved
        clsname = classname(val)
        # create the dataset or the group and call the save method on it
        if cls.is_dataset() and name is None:
            # if we want to save a dataset in the root group (name = None)
            # we have to give it a name
            name = "default"
        if cls.is_group():
            if name is not None:
                level = cls.create_group(level, name, options)
            set_classname(level, clsname)
        ret = cls.save(val, level, options, name)
        if cls.is_dataset():
            set_classname(ret, clsname)
        return ret

    @classmethod
    def load_from_level(cls, level, obj=None):
        """ The loader that has to be implemented by subclasses.
        """
        raise NotImplementedError()

    @classmethod
    def create_group(cls, level, name, options):
        return level.create_group(name)

    @classmethod
    def create_dataset(cls, data, level, name, **kwargs):
        ds = level.create_dataset(name, data=data, **kwargs)
        return ds

    @classmethod
    def is_group(cls):
        return cls.level_type == 'group'

    @classmethod
    def is_dataset(cls):
        return cls.level_type == 'dataset'

    @classmethod
    def get_type(cls, level):
        return cls.casting[get_classname(level)]


class TypeHandler(Handler, metaclass=TypeRegister):
    """ Handles data of a specific type or class.
    """
    types = []
    casting = {}

    @classmethod
    def register(cls, t):
        global type_saver_handlers, loader_handlers
        if t in type_saver_handlers:
            raise ValueError('Type `%s` is already handled by `%s`.' %
                             (str(t), str(type_saver_handlers[t])))
        typename = classname(t)
        type_saver_handlers[t] = cls
        loader_handlers[typename] = cls
        cls.casting[typename] = t


class InstanceHandler(Handler, metaclass=InstanceRegister):
    """ Handles all instances of a specific (parent) class.

    If an instance is subclass to several classes for which a handler exists,
    no error will be raised (in contrast to TypeHandler). Rather, the first
    match in the global instance_saver_handlers OrderedDict will be used.
    """
    instances = []
    casting = {}

    @classmethod
    def register(cls, t):
        global instance_saver_handlers, loader_handlers
        if t in instance_saver_handlers:
            raise ValueError('Instance `%s` is already handled by `%s`.' %
                             (str(t), str(instance_saver_handlers[t])))
        typename = classname(t)
        instance_saver_handlers[t] = cls
        loader_handlers[typename] = cls
        cls.casting[typename] = t


# Specific handlers
class NoneHandler(TypeHandler):
    types = [type(None)]

    def save(cls, val, level, options, name):
        ds = cls.create_dataset(0, level, name, **options(0))
        return ds

    def load_from_level(cls, level, obj=None):
        return None


class ScalarHandler(TypeHandler):
    types = [float, bool, complex, np.int8, np.int16, np.int32,
             np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
             np.float16, np.float32, np.float64, np.bool_, np.complex64,
             np.complex128]

    def save(cls, val, level, options, name):
        ds = cls.create_dataset(val, level, name, **options(val))
        return ds

    def load_from_level(cls, level, obj=None):
        # cast to the correct type
        type_ = cls.get_type(level)
        # retrieve scalar dataset
        return type_(level[()])


class IntHandler(TypeHandler):
    """ Special int handler to deal with Python's variable size ints.

    They are stored as byte arrays. Probably not the most efficient solution...
    """
    types = [int]

    def save(cls, val, level, options, name):
        val = val.to_bytes((val.bit_length() + 7) // 8, byteorder='little')
        data = np.frombuffer(val, dtype=np.uint8)
        ds = cls.create_dataset(data, level, name, **options(data))
        return ds

    def load_from_level(cls, level, obj=None):
        return int.from_bytes(level[:].tobytes(), byteorder='little')


class TimeHandler(TypeHandler):
    types = [np.datetime64, np.timedelta64]

    def save(cls, val, level, options, name):
        val2 = val.view('<i8')
        ds = cls.create_dataset(val2, level, name, **options(val2))
        set_attribute(ds, '__dtype__', val.dtype)
        return ds

    def load_from_level(cls, level, obj=None):
        val = level[()]
        dtype = get_attribute(level, '__dtype__')
        return val.view(dtype)


class StringHandler(TypeHandler):
    types = [str]

    def save(cls, val, level, options, name):
        b = val.encode(encoding=options.encoding)
        ds = BytesHandler.save(b, level, options, name)
        set_attribute(ds, '__encoding__', options.encoding)
        return ds

    def load_from_level(cls, level, obj=None):
        bstring = level[:].tobytes()
        return bstring.decode(get_attribute(level, '__encoding__'))


class BytesHandler(TypeHandler):
    types = [bytes]

    def save(cls, val, level, options, name):
        data = np.frombuffer(val, dtype=np.uint8)
        ds = cls.create_dataset(data, level, name, **options(data))
        return ds

    def load_from_level(cls, level, obj=None):
        return level[:].tobytes()


class DictHandler(TypeHandler):
    level_type = 'group'
    types = [dict]

    def save(cls, val, level, options, name):
        for key, value in val.items():
            save_to_level(value, level, options, key)

    def load_from_level(cls, level, obj=None):
        obj = dict()
        for key, value in level.items():
            obj[key] = load_from_level(value)
        return obj


class SimpleNamespaceHandler(TypeHandler):
    level_type = 'group'
    types = [types.SimpleNamespace]

    def save(cls, val, level, options, name):
        for key, value in val.__dict__.items():
            save_to_level(value, level, options, key)

    def load_from_level(cls, level, obj=None):
        obj = types.SimpleNamespace()
        for key, value in level.items():
            setattr(obj, key, load_from_level(value))
        return obj


class ListHandler(TypeHandler):
    """ Despite its name it also handles tuples.
    """
    level_type = 'group'
    types = [list, tuple]

    def save(cls, val, level, options, name):
        for idx, element in enumerate(val):
            save_to_level(element, level, options, 'idx_%d' % idx)

    def load_from_level(cls, level, obj=None):
        obj = []
        length = len(list(level.keys()))
        for idx in range(length):
            obj.append(load_from_level(level['idx_%d' % idx]))
        # cast to tuple if necessary
        type_ = cls.get_type(level)
        return type_(obj)


class NDArrayHandler(TypeHandler):
    types = [np.ndarray]

    def save(cls, val, level, options, name):
        ds = cls.create_dataset(val, level, name, **options(val))
        return ds

    def load_from_level(cls, level, obj=None):
        return level[:]
