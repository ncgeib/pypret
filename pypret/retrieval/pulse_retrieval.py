""" This module provides the basic classes for the pulse retrieval algorithms.

Disclaimer
----------

THIS CODE IS FOR EDUCATIONAL PURPOSES ONLY! The code in this package was not
optimized for accuracy or performance. Rather it aims to provide a simple
implementation of the basic algorithms.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import numpy as np
from types import SimpleNamespace
from .. import io
from ..mesh_data import MeshData
from ..pulse_error import pulse_error
from .. import lib

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


def Retriever(pnps, method, **kwargs):
    """ Factory function to create PNPS instances.
    """
    method = method.lower()
    try:
        cls = _RETRIEVER_CLASSES[method]
    except KeyError:
        raise ValueError("Retriever '%s' is unknown!" % (method))
    return cls(pnps, **kwargs)


class MetaIORetriever(io.MetaIO, MetaRetriever):
    # to fix metaclass conflicts
    pass


# =============================================================================
# Retriever Base class
# =============================================================================
class RetrieverBase(io.IO, metaclass=MetaIORetriever):
    method = None
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

    def retrieve(self, measurement, initial_guess):
        """ Retrieve pulse from ``measurement`` starting at ``initial_guess``.

        Parameter
        ---------
        measurement : MeshData
            A MeshData instance that contains the PNPS measurement. The first
            axis has to correspond to the PNPS parameter, the second to the
            frequency. The data has to be the measured _intensity_ over the
            frequency (not wavelength!). The second axis has to match exactly
            the frequency axis of the underlying PNPS instance. No
            interpolation is done.
        initial_guess : 1d-array
            The spectrum of the pulse that is used in the iterative
            retrieval.

        Notes
        -----
        This function provides no interpolation or data processing. You have
        to write a retriever wrapper for that purpose.
        """
        if not isinstance(measurement, MeshData):
            raise ValueError("measurement has to be a MeshData instance!")
        self._retrieve_begin(measurement, initial_guess)
        self._retrieve()
        self._retrieve_end()

    def _retrieve_begin(self, measurement, initial_guess):
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
        rs = self._retrieval_state
        Tmn_meas = self.Tmn_meas
        mu = np.sum(Tmn_meas * Tmn) / np.sum(Tmn * Tmn)
        # store intermediate results in current retrieval state
        if store:
            rs.mu = mu
            rs.Tmn = Tmn
            rs.Smk = self.pnps.Smk
        return np.sum((Tmn_meas - mu * Tmn)**2)

    def _R(self, Tmn, store=True):
        """ Calculates the trace error from a simulated trace Tmn.
        """
        r = self._r(Tmn, store=store)
        return self._Rr(r)

    def _Rr(self, r):
        """ Calculates the trace error from the minimization objective r.
        """
        return np.sqrt(r / (self.M * self.N)) / self.Tmn_meas.max()

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
        # 2 - the original test pulse
        res.pulse_original = pulse_original
        # 3 - the initial guess
        res.pulse_initial = self.initial_guess

        # the measurement traces
        # 1 - the original data used for retrieval
        res.trace_input = self.Tmn_meas
        # 2 - the trace error and the trace calculated from the retrieved pulse
        res.trace_error = self.trace_error(res.pulse_retrieved)
        res.trace_retrieved = rs.mu * rs.Tmn

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




class StepRetriever(RetrieverBase):

    def _retrieve(self):
        # local rename
        o = self.options
        log = self.log
        res = self._result
        rs = self._retrieval_state
        # store current guess in attribute
        spectrum = self.initial_guess.copy()
        # initialize R
        R = res.trace_error
        for i in range(o.max_iterations):
            # store trace error and spectrum for later analysis
            if self.logging:
                # if the trace error was only approximated, calculate it here
                if rs.approximate_error:
                    R = self.trace_error(spectrum, store=False)
                log.trace_error.append(R)
            # Perform a single retriever step in one of the algorithms
            R, new_spectrum = self._retrieve_step(i, spectrum.copy())
            # update the solution if the result is better
            if R < res.trace_error:
                # R is calculated for the input, i.e., the old spectrum.
                res.trace_error = R
                res.approximate_error = rs.approximate_error
                res.spectrum[:] = spectrum  # store the old spectrum
                rs.steps_since_improvement = 0
            else:
                rs.steps_since_improvement += 1
            # accept the new spectrum
            spectrum[:] = new_spectrum
            if self.verbose:
                if i == 0:
                    print("iteration".ljust(10) + "trace error".ljust(20))
                s = "{:d}".format(i + 1).ljust(10)
                if rs.approximate_error:
                    s += "~"
                else:
                    s += " "
                s += "{:.10e}".format(R)
                if R == res.trace_error:
                    s += "*"
                print(s)
            if not rs.running:
                break
        if self.verbose:
            print()
            print("~ approximate trace error")
            print("* accepted as best trace error")
            print()

        # return the retrieved spectrum
        # for a more detailed analysis call self.result()
        return res.spectrum

















