""" This module provides the basic classes for the pulse retrieval algorithms.
"""
import numpy as np
from types import SimpleNamespace
from .. import io
from ..mesh_data import MeshData
from ..pulse_error import pulse_error
from .. import lib
from ..pnps import BasePNPS

# global dictionary that contains all PNPS classes
_RETRIEVER_CLASSES = {}


# =============================================================================
# Metaclass and factory
# =============================================================================
class MetaRetriever(type):
    """ Metaclass that registers Retriever classes in a global dictionary.
    """
    def __new__(cls, clsmethod, bases, attrs):
        global _RETRIEVER_CLASSES
        newclass = super().__new__(cls, clsmethod, bases, attrs)
        method = newclass.method
        if method is None:
            return newclass
        # register the Retriever method, e.g. 'copra'
        if method in _RETRIEVER_CLASSES:
            raise ValueError("Two retriever classes implement retriever '%s'."
                             % method)
        _RETRIEVER_CLASSES[method] = newclass
        return newclass


class MetaIORetriever(io.MetaIO, MetaRetriever):
    # to fix metaclass conflicts
    pass


# =============================================================================
# Retriever Base class
# =============================================================================
class BaseRetriever(io.IO, metaclass=MetaIORetriever):
    """ The abstract base class for pulse retrieval.

    This class implements common functionality for different retrieval
    algorithms.
    """
    method = None
    supported_schemes = None
    _io_store = ['pnps', 'options', 'logging', 'log',
                 '_retrieval_state', '_result']

    def __init__(self, pnps, logging=False, verbose=False, **kwargs):
        self.pnps = pnps
        self.ft = self.pnps.ft
        self.options = SimpleNamespace(**kwargs)
        self._result = None
        self.logging = logging
        self.verbose = verbose
        self.log = None
        rs = self._retrieval_state = SimpleNamespace()
        rs.running = False
        if (self.supported_schemes is not None and
                pnps.scheme not in self.supported_schemes):
            raise ValueError("Retriever '%s' does not support scheme '%s'. "
                             "It only supports %s." %
                             (self.method, pnps.scheme, self.supported_schemes)
                             )

    def retrieve(self, measurement, initial_guess, weights=None,
                 **kwargs):
        """ Retrieve pulse from ``measurement`` starting at ``initial_guess``.

        Parameters
        ----------
        measurement : MeshData
            A MeshData instance that contains the PNPS measurement. The first
            axis has to correspond to the PNPS parameter, the second to the
            frequency. The data has to be the measured _intensity_ over the
            frequency (not wavelength!). The second axis has to match exactly
            the frequency axis of the underlying PNPS instance. No
            interpolation is done.
        initial_guess : 1d-array
            The spectrum of the pulse that is used as initial guess in the
            iterative retrieval.
        weights : 1d-array
            Weights that are attributed to the measurement for retrieval.
            In the case of (assumed) Gaussian uncertainties with standard
            deviation sigma they should correspond to 1/sigma.
            Not all algorithms support using the weights.
        kwargs : dict
            Can override retrieval options specified in :func:`__init__`.

        Notes
        -----
        This function provides no interpolation or data processing. You have
        to write a retriever wrapper for that purpose.
        """
        self.options.__dict__.update(**kwargs)
        if not isinstance(measurement, MeshData):
            raise ValueError("measurement has to be a MeshData instance!")
        self._retrieve_begin(measurement, initial_guess, weights)
        self._retrieve()
        self._retrieve_end()

    def _retrieve_begin(self, measurement, initial_guess, weights):
        pnps = self.pnps
        if not np.all(pnps.process_w == measurement.axes[1]):
            raise ValueError("Measurement has to lie on simulation grid!")
        # Store measurement
        self.measurement = measurement
        self.parameter = measurement.axes[0]
        self.Tmn_meas = measurement.data

        self.initial_guess = initial_guess
        # set the size
        self.M, self.N = self.Tmn_meas.shape
        # Setup the weights
        if weights is None:
            self._weights = np.ones((self.M, self.N))
        else:
            self._weights = weights.copy()
        # Retrieval state
        rs = self._retrieval_state
        rs.approximate_error = False
        rs.running = True
        rs.steps_since_improvement = 0
        # Initialize result
        res = self._result = SimpleNamespace()
        res.trace_error = self.trace_error(self.initial_guess)
        res.approximate_error = False
        res.spectrum = self.initial_guess.copy()
        # Setup the logger
        if self.logging:
            log = self.log = SimpleNamespace()
            log.trace_error = []
            log.initial_guess = self.initial_guess.copy()
        else:
            self.log = None
        if self.verbose:
            print("Started retriever '%s'" % self.method)
            print("Options:")
            print(self.options)
            print("Initial trace error R = {:.10e}".format(res.trace_error))
            print("Starting retrieval...")
            print()

    def _retrieve_end(self):
        rs = self._retrieval_state
        rs.running = False
        res = self._result
        if res.approximate_error:
            res.trace_error = self.trace_error(res.spectrum)
            res.approximate_error = False

    def _project(self, measured, Smk):
        """ Performs the projection on the measured intensity.
        """
        # in frequency domain
        Smn = self.ft.forward(Smk)
        # project and specially handle values with zero amplitude
        absSmn = np.abs(Smn)
        f = (absSmn > 0.0)
        Smn[~f] = np.sqrt(measured[~f] + 0.0j)
        Smn[f] = Smn[f] / absSmn[f] * np.sqrt(measured[f] + 0.0j)
        # back in time domain
        Smk2 = self.ft.backward(Smn)
        return Smk2

    def _objective_function(self, spectrum):
        """ Calculates the minimization objective from the pulse spectrum.

        This is Eq. 11 in the paper:

            r = sum (Tmn^meas - mu * Tmn)
        """
        # calculate the PNPS trace
        Tmn = self.pnps.calculate(spectrum, self.parameter)
        return self._r(Tmn)

    def trace_error(self, spectrum, store=True):
        """ Calculates the trace error from the pulse spectrum.
        """
        Tmn = self.pnps.calculate(spectrum, self.parameter)
        return self._R(Tmn, store=store)

    def _r(self, Tmn, store=True):
        """ Calculates the minimization objective r from a simulated trace Tmn.
        """
        diff = self._error_vector(Tmn, store=store)
        return np.sum(diff * diff)

    def _error_vector(self, Tmn, store=True):
        """ Calculates the residual vector from measured to simulated
        intensity.
        """
        # rename
        rs = self._retrieval_state
        Tmn_meas = self.Tmn_meas
        # scaling factor
        w2 = self._weights * self._weights
        mu = np.sum(Tmn_meas * Tmn * w2) / np.sum(Tmn * Tmn * w2)
        # store intermediate results in current retrieval state
        if store:
            rs.mu = mu
            rs.Tmn = Tmn
            rs.Smk = self.pnps.Smk
        return np.ravel((Tmn_meas - mu * Tmn) * self._weights)

    def _R(self, Tmn, store=True):
        """ Calculates the trace error from a simulated trace Tmn.
        """
        r = self._r(Tmn, store=store)
        return self._Rr(r)

    def _Rr(self, r):
        """ Calculates the trace error from the minimization objective r.
        """
        return np.sqrt(r / (self.M * self.N *
                            (self.Tmn_meas * self._weights).max()**2))

    def result(self, pulse_original=None, full=True):
        """ Analyzes the retrieval results in one retrieval instance
            and processes it for plotting or storage.
        """
        rs = self._retrieval_state
        if self._result is None or self._retrieval_state.running:
            return None
        res = SimpleNamespace()
        # the meta data
        res.parameter = self.parameter
        res.options = self.options
        res.logging = self.logging
        res.measurement = self.measurement
        # store the retriever itself
        if full:
            res.pnps = self.pnps
        else:
            res.pnps = None

        # the pulse spectra
        # 1 - the retrieved pulse
        res.pulse_retrieved = self._result.spectrum
        # 2 - the original test pulse, optional
        res.pulse_original = pulse_original
        # 3 - the initial guess
        res.pulse_initial = self.initial_guess

        # the measurement traces
        # 1 - the original data used for retrieval
        res.trace_input = self.Tmn_meas
        # 2 - the trace error and the trace calculated from the retrieved pulse
        res.trace_error = self.trace_error(res.pulse_retrieved)
        res.trace_retrieved = rs.mu * rs.Tmn
        res.response_function = rs.mu
        # the weights
        res.weights = self._weights

        # this is set if the original spectrum is provided
        if res.pulse_original is not None:
            # the trace error of the test pulse (non-zero for noisy input)
            res.trace_error_optimal = self.trace_error(res.pulse_original)
            # 3 - the optimal trace calculated from the test pulse
            res.trace_original = rs.mu * rs.Tmn
            dot_ambiguity = False
            if self.pnps.method == "ifrog" or self.pnps.scheme == "shg-frog":
                dot_ambiguity = True
            # the pulse error to the test pulse
            res.pulse_error, res.pulse_retrieved = pulse_error(
                    res.pulse_retrieved, res.pulse_original, self.ft,
                    dot_ambiguity=dot_ambiguity)

        if res.logging:
            # the logged trace errors
            res.trace_errors = np.array(self.log.trace_error)
            # the running minimum of the trace errors (for plotting)
            res.rm_trace_errors = np.minimum.accumulate(res.trace_errors,
                                                        axis=-1)
        if self.verbose:
            lib.retrieval_report(res)
        return res


def Retriever(pnps: BasePNPS, method: str = "copra", maxiter=300, maxfev=None,
              logging=False, verbose=False, **kwargs) -> BaseRetriever:
    """ Creates a retriever instance.

    Parameters
    ----------
    pnps : PNPS
        A PNPS instance that is used to simulate a PNPS measurement.
    method : str, optional
        Type of solver.  Should be one of
            - 'copra'       :class:`(see here) <COPRARetriever>`
            - 'gpa'         :class:`(see here) <GPARetriever>`
            - 'gp-dscan'     :class:`(see here) <GPDSCANRetriever>`
            - 'pcgpa'       :class:`(see here) <PCGPARetriever>`
            - 'pie'         :class:`(see here) <PIERetriever>`
            - 'lm'          :class:`(see here) <LMRetriever>`
            - 'bfgs'        :class:`(see here) <BFGSRetriever>`
            - 'de'          :class:`(see here) <DERetriever>`
            - 'nelder-mead' :class:`(see here) <NMRetriever>`

        'copra' is the default choice.
    maxiter : int, optional
        The maximum number of algorithm iterations. The default is 300.
    maxfev : int, optional
        The maximum number of function evaluations. If given, the algorithms
        stop before this number is reached. Not all algorithms support this
        feature. Default is ``None``, in which case it is ignored.
    logging : bool, optional
        Stores trace errors and pulses over the iterations if supported
        by the retriever class. Default is `False`.
    verbose : bool, optional
        Prints out trace errors during the iteration if supported by the
        retriever class. Default is `False`.
    """
    method = method.lower()
    try:
        cls = _RETRIEVER_CLASSES[method]
    except KeyError:
        raise ValueError("Retriever '%s' is unknown!" % (method))
    return cls(pnps, maxiter=maxiter, maxfev=maxfev,
               logging=logging, verbose=verbose, **kwargs)
