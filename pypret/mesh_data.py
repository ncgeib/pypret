""" This module implements an object for dealing with two-dimensional data.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from . import lib
from . import io


class MeshData(io.IO):
    _io_store = ["data", "axes", "labels", "units", "uncertainty"]

    def __init__(self, data,  *axes, uncertainty=None, labels=None,
                 units=None):
        """ Creates a MeshData instance.

        Parameters
        ----------
        data : ndarray
            A at least two-dimensional array containing the data.
        *axes : ndarray
            Arrays specifying the coordinates of the data axes. Must be given
            in indexing order.
        uncertainty : ndarray
            An ndarray of the same size as `data` that contains some measure
            of the uncertainty of the meshdata. E.g., it could be the standard
            deviation of the data.
        labels : list of str, optional
            A list of strings labeling the axes. The last element labels the
            data itself, e.g. ``labels`` must have one more element than the
            number of axes.
        units : list of str, optional
            A list of unit strings.
        """
        self.data = data.copy()
        self.axes = [np.array(a).copy() for a in axes]
        if uncertainty is not None:
            self.uncertainty = uncertainty.copy()
        else:
            self.uncertainty = None
        if self.ndim != len(axes):
            raise ValueError("Number of supplied axes is wrong!")
        if self.shape != tuple(ax.size for ax in self.axes):
            raise ValueError("Shape of supplied axes is wrong!")
        self.labels = labels
        if self.labels is None:
            self.labels = ["" for ax in self.axes]
        self.units = units
        if self.units is None:
            self.units = ["" for ax in self.axes]

    @property
    def shape(self):
        """ Returns the shape of the data as a tuple.
        """
        return self.data.shape

    @property
    def ndim(self):
        """ Returns the dimension of the data as integer.
        """
        return self.data.ndim

    def copy(self):
        """ Creates a copy of the MeshData instance. """
        return MeshData(self.data, *self.axes, uncertainty=self.uncertainty,
                        labels=self.labels, units=self.units)

    def marginals(self, normalize=False, axes=None):
        """ Calculates the marginals of the data.

        axes specifies the axes of the marginals, e.g., the axes on which the
        sum is projected.
        """
        return lib.marginals(self.data, normalize=normalize, axes=axes)

    def normalize(self):
        """ Normalizes the maximum of the data to 1.
        """
        self.scale(1.0 / self.data.max())

    def scale(self, scale):
        if self.uncertainty is not None:
            self.uncertainty *= scale
        self.data *= scale

    def autolimit(self, *axes, threshold=1e-2, padding=0.25):
        """ Limits the data based on the marginals.
        """
        if len(axes) == 0:
            # default: operate on all axes
            axes = list(range(self.ndim))
        marginals = lib.marginals(self.data)
        limits = []
        for i, j in enumerate(axes):
            limit = lib.limit(self.axes[j], marginals[j],
                              threshold=threshold, padding=padding)
            limits.append(limit)
        self.limit(*limits, axes=axes)

    def limit(self, *limits, axes=None):
        """ Limits the data range of this instance.

        Parameters
        ----------
        *limits : tuples
            The data limits in the axes as tuples. Has to match the dimension
            of the data or the number of axes specified in the `axes`
            parameter.
        axes : tuple or None
            The axes in which the limit is applied. Default is `None` in which
            case all axes are selected.
        """
        if axes is None:
            # default: operate on all axes
            axes = list(range(self.ndim))
        axes = lib.as_list(axes)
        if len(axes) != len(limits):
            raise ValueError("Number of limits must match the specified axes!")
        slices = []
        for j in range(self.ndim):
            if j in axes:
                i = axes.index(j)
                ax = self.axes[j]
                x1, x2 = limits[i]
                # do it this way as we cannot assume them to be sorted...
                idx1 = np.argmin(np.abs(ax - x1))
                idx2 = np.argmin(np.abs(ax - x2))
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                elif idx1 == idx2:
                    raise ValueError('Selected empty slice along axis %d!' % i)
                slices.append(slice(idx1, idx2 + 1))
            else:
                # empty slice
                slices.append(slice(None))
            self.axes[j] = self.axes[j][slices[-1]]
        self.data = self.data[(*slices,)]
        if self.uncertainty is not None:
            self.uncertainty = self.uncertainty[(*slices,)]

    def interpolate(self, axis1=None, axis2=None, degree=2, sorted=False):
        """ Interpolates the data on a new two-dimensional, equidistantly
        spaced grid.
        """
        axes = [axis1, axis2]
        for i in range(self.ndim):
            if axes[i] is None:
                axes[i] = self.axes[i]
        # FITPACK can only deal with strictly increasing axes
        # so sort them beforehand if necessary...
        orig_axes = self.axes
        data = self.data.copy()
        if self.uncertainty is not None:
            uncertainty = self.uncertainty.copy()
        if not sorted:
            for i in range(len(orig_axes)):
                idx = np.argsort(orig_axes[i])
                orig_axes[i] = orig_axes[i][idx]
                data = np.take(data, idx, axis=i)
                if self.uncertainty is not None:
                    uncertainty = np.take(uncertainty, idx, axis=i)
        dataf = RegularGridInterpolator(tuple(orig_axes), data,
                                        bounds_error=False, fill_value=0.0)
        grid = lib.build_coords(*axes)
        self.data = dataf(grid)
        self.axes = axes
        if self.uncertainty is not None:
            dataf = RegularGridInterpolator(tuple(orig_axes), uncertainty,
                                            bounds_error=False, fill_value=0.0)
            self.uncertainty = dataf(grid)

    def flip(self, *axes):
        """ Flips the data on the specified axes.
        """
        if len(axes) == 0:
            return
        axes = lib.as_list(axes)
        slices = [slice(None) for ax in self.axes]
        for ax in axes:
            self.axes[ax] = self.axes[ax][::-1]
            slices[ax] = slice(None, None, -1)
        self.data = self.data[slices]
        if self.uncertainty is not None:
            self.uncertainty = self.uncertainty[slices]
