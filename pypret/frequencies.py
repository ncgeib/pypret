""" This module handles conversion between frequency units.

The supported units and their shorthands are:

- wl : wavelength in meter
- om: angular frequency in rad/s
- f: frequency in 1/s
- k: angular wavenumber in rad/m

The conversion functions have the form `shorthand2shorthand` which is not
pythonic but very short. A more pythonic conversion can be achieved by using
the `convert` function

>>> convert(x, 'wl', 'om')

The shorthands will be used throughout the package to identify frequency units.

The functions in this module should be used wherever a frequency convention
is necessary to avoid mistakes and make the code more expressive.
"""
from copy import copy
from .lib import sol, twopi


frequency_labels = {
    'wl': 'wavelength',
    'om': 'angular frequency',
    'f': 'frequency',
    'k': 'angular wavenumber'
}

frequency_units = {
    'wl': 'm',
    'om': 'Hz rad',
    'f': 'Hz',
    'k': 'rad/m'
}


def om2wl(om):
    return twopi/om*sol


def k2wl(k):
    return twopi/k


def f2wl(f):
    return sol/f


def wl2f(wl):
    return sol/wl


def om2f(om):
    return om/twopi


def k2f(k):
    return k*sol/twopi


def wl2om(wl):
    return twopi*sol/wl


def f2om(f):
    return twopi*f


def k2om(k):
    return k*sol


def wl2k(wl):
    return twopi/wl


def om2k(om):
    return om/sol


def f2k(f):
    return twopi*f/sol


# this dictionary can be used for programmatic conversions
conversions = {
    'wl': {
        'wl': lambda x: copy(x),
        'om': wl2om,
        'f': wl2f,
        'k': wl2k
    },
    'om': {
        'wl': om2wl,
        'om': lambda x: copy(x),
        'f': om2f,
        'k': om2k
    },
    'f': {
        'wl': f2wl,
        'om': f2om,
        'f': lambda x: copy(x),
        'k': f2k
    },
    'k': {
        'wl': k2wl,
        'om': k2om,
        'f': k2f,
        'k': lambda x: copy(x)
    }
}


def convert(x, unit1, unit2):
    """ Convert between two frequency units.

    Parameters
    ----------
    x : float or array_like
        Numerical value or array that should be converted.
    unit1, unit2 : str
        Shorthands for the original unit (`unit1`) and the destination unit
        (`unit2`).

    Returns
    -------
    float or array_like
        The converted numerical value or array. It will always be a copy, even
        if `unit1 == unit2`.

    Notes
    -----
    Unit shorthands can be any of
    `wl` : wavelength in meter
    `om` : angular frequency in rad/s
    `f` : frequency in 1/s
    `k` : angular wavenumber in rad/m
    """
    return conversions[unit1][unit2](x)
