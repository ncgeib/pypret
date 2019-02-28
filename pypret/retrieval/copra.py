""" This module implements the common pulse retrieval algorithm.

Disclaimer
----------

THIS CODE IS FOR EDUCATIONAL PURPOSES ONLY! The code in this package was not
optimized for accuracy or performance. Rather it aims to provide a simple
implementation of the basic algorithms.

Author: Nils Geib, nils.geib@uni-jena.de
"""
import numpy as np
from .pulse_retrieval import StepRetriever
from .. import lib


class COPRARetriever(StepRetriever):
    method = "copra"

    def __init__(self, pnps, logging=False, verbose=False, max_iterations=300,
                 alpha=0.25, **kwargs):
        super().__init__(pnps, logging=logging, verbose=verbose,
                         max_iterations=max_iterations, alpha=alpha, **kwargs)

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

    def _retrieve_begin(self, measurement, initial_guess):
        super()._retrieve_begin(measurement, initial_guess)
        pnps = self.pnps
        rs = self._retrieval_state
        rs.mode = "local"  # COPRA starts with local mode
        # calculate the maximum gradient norm
        # self.trace_error() was called beforehand -> rs.Tmn and rs.Smk exist!
        Smk2 = self._project(self.Tmn_meas / rs.mu, rs.Smk)
        nablaZnm = pnps.gradient(Smk2, self.parameter)
        rs.current_max_gradient = np.max(np.sum(lib.abs2(nablaZnm), axis=1))

    def _retrieve_step(self, iteration, En):
        """ Perform a single COPRA step.
        
        Parameter
        ---------
        iteration : int
            The current iteration number - mainly for logging.
        En : 1d-array
            The current pulse spectrum.        
        """
        # local rename
        ft = self.ft
        options = self.options
        pnps = self.pnps
        rs = self._retrieval_state
        Tmn_meas = self.Tmn_meas
        # current gradient -> last gradient
        rs.previous_max_gradient = rs.current_max_gradient
        rs.current_max_gradient = 0.0
        # switch iteration
        if rs.steps_since_improvement == 5:
            rs.mode = "global"
        # local iteration
        if rs.mode == "local":
            # running estimate for the trace
            Tmn = np.zeros((self.M, self.N))
            for m in np.random.permutation(np.arange(self.M)):
                p = self.parameter[m]
                Tmn[m, :] = pnps.calculate(En, p)
                Smk2 = self._project(Tmn_meas[m, :] / rs.mu, pnps.Smk)
                nablaZnm = pnps.gradient(Smk2, p)
                # calculate the step size
                Zm = lib.norm2(Smk2 - pnps.Smk)
                gradient_norm = lib.norm2(nablaZnm)
                if gradient_norm > rs.current_max_gradient:
                    rs.current_max_gradient = gradient_norm
                gamma = Zm / max(rs.current_max_gradient,
                                 rs.previous_max_gradient)
                # update the spectrum
                En -= gamma * nablaZnm
            # Tmn is only an approximation as En changed in the iteration!
            rs.approximate_error = True
            R = self._R(Tmn)  # updates rs.mu!!!
        # global iteration
        elif rs.mode == "global":
            Tmn = pnps.calculate(En, self.parameter)
            r = self._r(Tmn)
            R = self._Rr(r)  # updates rs.mu!!!
            rs.approximate_error = False
            # gradient descent w.r.t. Smk
            gradrmk = (-4 * rs.mu * ft.dt / (ft.dw * lib.twopi) *
                       ft.backward(ft.forward(pnps.Smk) *
                                   (Tmn_meas - rs.mu * Tmn)))
            etar = options.alpha * r / lib.norm2(gradrmk)
            Smk2 = pnps.Smk - etar * gradrmk
            # gradient descent w.r.t. En
            nablaZn = pnps.gradient(Smk2, self.parameter).sum(axis=0)
            # calculate the step size
            Z = lib.norm2(Smk2 - pnps.Smk)
            etaz = options.alpha * Z / lib.norm2(nablaZn)
            # update the spectrum
            En -= etaz * nablaZn        
        return R, En
