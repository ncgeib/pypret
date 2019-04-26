""" This module provides classes to calculate the refractive index
based on Sellmeier equations.

This is required to correctly model d-scan measurements.

Currently only very few materials are implemented. But more should be easy
to add. If the refractive index is described by formula 1 or 2 from
refractiveindex.info you can simply instantiate `SellmeierF1` or `SellmeierF2`.
If not, inherit from BaseMaterial and implement the `self._func` method.
"""
import numpy as np
from .frequencies import convert
from . import lib
from . import io


class BaseMaterial(io.IO):
    """ Abstract base class for dispersive materials.
    
    """
    
    def __init__(self, coefficients, freq_range, scaling=1.0e6,
                 check_bounds=True, name="", long_name=""):
        ''' Creates a dispersive material.
        
        Parameters
        ----------
        coefficients: ndarray
            The Sellmeier coefficients.
        freq_range : iterable
            The wavelength range in which the Sellmeier equation is valid
            (given in m).
        check_bounds : bool, optional
            Specifies if the frequency argument should be checked on every
            evaluation to match the allowed range.
        scaling : float, optional
            Specifies the scaling of the Sellmeier formula. E.g., most
            Sellmeier formulas are defined in terms of µm (micrometer), 
            whereas our function interface works in meter. In that case the
            scaling would be `1e6`. Default is `1.0e6`.
        '''
        if len(freq_range) != 2:
            raise ValueError("Frequency range must specified with two elements.")
        self._coefficients = np.array(coefficients)
        self._range = np.array(freq_range)
        self._scaling = scaling
        self.check = check_bounds
        self.name = name
        self.long_name = long_name
        
    def _check(self, x):
        if not self.check:
            return
        minx, maxx = np.min(x), np.max(x)
        if (minx < self._range[0]) or (maxx > self._range[1]):
            raise ValueError('Wavelength array [%e, %e] outside of valid range '
                             'of the Sellmeier equation [%e, %e].' % 
                             (minx, maxx, self._range[0], self._range[1]))          
    
    def _convert(self, x, unit):
        '''This is intended for conversion to be used in `self._func`.'''
        if unit != 'wl':
            x = convert(x, unit, 'wl')
        self._check(x)
        if self._scaling != 1.0:
            x = x * self._scaling
        return x    
    
    def n(self, x, unit='wl'):
        '''The refractive index at frequency `x` specified in units `unit`. '''
        return self._func(self._convert(x, unit))    
    
    def k(self, x, unit='wl'):
        '''The wavenumber in the material in rad / m.'''
        wl = convert(x, unit, "wl")
        return self.n(wl, unit="wl") * lib.twopi / wl


class SellmeierF1(BaseMaterial):
    ''' Defines a dispersive material via a specific Sellmeier equation.

        This subclass supports materials with a Sellmeier equation of the
        form::

            n^2(l) - 1 = c1 + c2 * l^2 / (l2 - c3^2) + ...

        This is formula 1 from refractiveindex.info [DispersionFormulas]_.
    '''
    def _func(self, x):
        c = self._coefficients
        x2 = x * x
        n2 = np.full_like(x, 1.0 + c[0])
        for i in range(1, len(c)-1, 2):
            n2 += c[i] * x2 / (x2 - c[i+1] * c[i+1])
        return np.sqrt(n2)


class SellmeierF2(BaseMaterial):
    ''' Defines a dispersive material via a specific Sellmeier equation.

        This subclass supports materials with a Sellmeier equation of the
        form::

            n^2(l) - 1 = c1 + c2 * l^2 / (l2 - c3) + ...

        This is formula 2 from refractiveindex.info [DispersionFormulas]_.
    '''
    def _func(self, x):
        c = self._coefficients
        x2 = x * x
        n2 = np.full_like(x, 1.0 + c[0])
        for i in range(1, c.size - 1, 2):
            n2 += c[i] * x2 / (x2 - c[i+1])
        return np.sqrt(n2)

FS = SellmeierF1(coefficients=[0.0000000, 0.6961663,
                               0.0684043, 0.4079426,
                               0.1162414, 0.8974794,
                               9.8961610],
                 freq_range=[0.21e-6, 6.7e-6],
                 name="FS",
                 long_name="Fused silica (fused quartz)")
"""Material instance describing fused silica (fused quartz).

The data was taken from refractiveindex.info
"""

BK7 = SellmeierF2(coefficients=[0.00000000000, 1.039612120,
                                0.00600069867, 0.231792344,
                                0.02001791440, 1.010469450,
                                103.560653],
                  freq_range=[0.3e-6, 2.5e-6],
                  name="BK7", long_name="N-BK7 (SCHOTT)")
"""Material instance describing N-BK7 (SCHOTT).

The data was taken from refractiveindex.info
"""