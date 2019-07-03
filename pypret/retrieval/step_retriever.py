""" This module implements specific pulse retrieval algorithms, e.g.,
COPRA, GPA, PCGPA, etc.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from .retriever import BaseRetriever
from .. import lib


class StepRetriever(BaseRetriever):

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
        for i in range(o.maxiter):
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


class COPRARetriever(StepRetriever):
    """ This module implements the common pulse retrieval algorithm
    [Geib2019]_.
    """
    method = "copra"

    def __init__(self, pnps, alpha=0.25, **kwargs):
        """ For a full documentation of the arguments see :class:`Retriever`.

        Parameters
        ----------
        alpha : float, optional
            Scales the step size in the global stage of COPRA. Higher values
            mean potentially faster convergence but less accuracy. Lower
            values provide higher accuracy for the cost of speed. Default is
            0.25.
        """
        super().__init__(pnps, alpha=alpha, **kwargs)

    def _retrieve_begin(self, measurement, initial_guess, weights):
        super()._retrieve_begin(measurement, initial_guess, weights)
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

        Parameters
        ----------
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
            w2 = self._weights * self._weights
            gradrmk = (-4 * ft.dt / (ft.dw * lib.twopi) *
                       ft.backward(rs.mu * ft.forward(pnps.Smk) *
                                   (Tmn_meas - rs.mu * Tmn) * w2))
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


class PCGPARetriever(StepRetriever):
    """ This class implements the principal components generalized projections
    algorithm (PCGPA) for SHG-FROG.

    We follow the algorithm as described in [Kane1999]_ but use the PNPS
    formalism from [Geib2019]_ and some minor modifications:

    - it supports both the singular value decomposition and the power
      method to find/approximate the largest eigenvector.
    - the projection includes the scaling factor µ. This makes the method
      robust against initial guesses with the wrong magnitude. It should
      have no adverse effect.

    """
    method = "pcgpa"
    supported_schemes = ["shg-frog"]

    def __init__(self, pnps, decomposition="power", **kwargs):
        """ For a full documentation of the arguments see :class:`Retriever`.

        Parameters
        ----------
        decomposition : str, optional
            It specifies how the FROG signal is decomposed. If `power` (the
            default) the power method is used to find the largest eigenvalue.
            If `svd` a full singular value decomposition is performed. This
            is potentially much slower but more accurate.
        """
        super().__init__(pnps, decomposition=decomposition, **kwargs)

    def _retrieve_begin(self, measurement, initial_guess, weights):
        super()._retrieve_begin(measurement, initial_guess, weights)
        if np.any(self.parameter != measurement.axes[0]):
            raise ValueError("The delay has to be sampled exactly at the "
                             "temporal simulation grid.")

    def _retrieve_step(self, iteration, En):
        # local rename
        ft = self.ft
        options = self.options
        pnps = self.pnps
        Tmn_meas = self.Tmn_meas
        rs = self._retrieval_state

        R = self.trace_error(En)  # updates rs.mu!!!
        # project on measured intensity
        Smk2 = self._project(Tmn_meas / rs.mu, pnps.Smk)
        # to outer product form
        for n in range(ft.N):
            Smk2[:, n] = np.roll(Smk2[::-1, n], n)
        if options.decomposition == "power":
            # apply power method iteration once
            Ek = ft.backward(En)
            Ek[:] = Smk2.conj() @ Ek
            Ek[:] = Ek.conj() / lib.norm(Ek)
        elif options.decomposition == "svd":
            # use full svd (slow!)
            U, s, V = np.linalg.svd(Smk2)
            Ek = U[:, 0] * np.sqrt(s[0])  # select U
        ft.forward(Ek, out=En)
        return R, En


class GPARetriever(StepRetriever):
    """ Implements the classical generalized projections algorithm for
    SHG-FROG as described in [DeLong1994]_ and [Trebino2000].

    As far as I know the determination of the step size in GPA is not
    made explicit in the publications. It is usually done in a line search.
    In this implementation we offer three different options:

    - an exact line search using a Brent style minimizer
    - a backtracking (inexact) line search using the Armijo-Goldstein
      condition with c=0.5 and tau=0.5.
    - the same heuristic choice for the step size used in copra.

    The last method is the fastest, but as the first is the classic choice
    for GPA, it is the default.
    """
    method = "gpa"
    supported_schemes = ["shg-frog"]

    def __init__(self, pnps, step_size="exact", **kwargs):
        """ For a full documentation of the arguments see :class:`Retriever`.

        Parameters
        ----------
        step_size : str, optional
            Specifies how the step size of the gradient step in GPA is
            determined. Default is `exact` which performs an exact line search.
            `inexact` performs a backtracking line search and `copra` uses the
            ad-hoc estimates for the step size used in COPRA.
        """
        super().__init__(pnps, step_size=step_size, **kwargs)

    def _retrieve_begin(self, measurement, initial_guess):
        super()._retrieve_begin(measurement, initial_guess)
        if np.any(self.parameter != measurement.axes[0]):
            raise ValueError("The delay has to be sampled exactly at the "
                             "temporal simulation grid.")

    def _retrieve_step(self, iteration, En):
        # local rename
        ft = self.ft
        options = self.options
        pnps = self.pnps
        Tmn_meas = self.Tmn_meas
        rs = self._retrieval_state

        R = self.trace_error(En)  # updates rs.mu!!!
        # obtain intermediate results
        delay, Amk, Ek, Smk, Tmn = pnps.intermediate(self.parameter)
        Ek = Ek[0, :]  # the same for every parameter
        # project on measured intensity
        Smk2 = self._project(Tmn_meas / rs.mu, Smk)
        # calculate the gradient of Z w.r.t. to the temporal pulse envelope
        # by directly implementing (S58) of [Geib2019]_
        dS = Smk2 - Smk
        indices = np.array(np.rint(self.parameter / ft.dt), dtype=np.int32)
        gradient = np.zeros((self.M, self.N), dtype=np.complex128)
        for m in range(self.M):
            gradient[m, :] = np.roll(dS[m, :] * Ek.conj(), -indices[m])
        gradient = -2 * np.sum(gradient + dS * Amk.conj(), axis=0)

        # approximate the step size with the "copra" step size
        gamma0 = lib.norm2(dS) / lib.norm2(gradient)
        if options.step_size == "copra":
            # directly choose the copra step size
            gamma = gamma0
        else:
            # do a line search
            def objective(gamma):
                self.trace_error(ft.forward(Ek - gamma * gradient))
                return lib.norm2(Smk2 - pnps.Smk)
            if options.step_size == "exact":
                # perform an exact line search
                bracket = [0.9 * gamma0, gamma0]
                ret = minimize_scalar(objective,
                                      bracket=bracket,
                                      method="brent")
                gamma = ret.x
            elif options.step_size == "inexact":
                # perform a back-tracking line search until the Armijo
                # condition is fulfilled.
                t = 0.5 * lib.norm2(gradient)
                tau = 0.5
                if iteration == 0:
                    rs.old_gamma = 5.0 * gamma0
                gamma = 2 * rs.old_gamma
                objective0 = objective(0.0)
                while objective(gamma) - objective0 > -gamma * t:
                    gamma = gamma * tau
                rs.old_gamma = gamma
        Ek = Ek - gamma * gradient

        ft.forward(Ek, out=En)
        return R, En


class GPDSCANRetriever(StepRetriever):
    """ This class implements a pulse retrieval algorithm for SHG and THG
    d-scan based on the paper [Miranda2017]_.

    In our tests we found that it does not converge in the noiseless case.
    In other words the global solution to the least-squares problem is not a
    fixed point of the iteration.
    """
    supported_schemes = ["shg-dscan", "thg-dscan"]
    method = "gp-dscan"

    def _retrieve_begin(self, measurement, initial_guess):
        super()._retrieve_begin(measurement, initial_guess)
        pnps = self.pnps
        rs = self._retrieval_state
        # calculate phase mask once
        rs.Hmn = np.zeros((self.M, self.N), dtype=np.complex128)
        for i, p in enumerate(self.parameter):
            rs.Hmn[i, :] = pnps.mask(p)

    def _retrieve_step(self, iteration, En):
        # local rename
        ft = self.ft
        pnps = self.pnps
        Tmn_meas = self.Tmn_meas
        rs = self._retrieval_state
        Hmn = rs.Hmn

        R = self.trace_error(En)  # updates rs.mu!!!
        # project on measured intensity
        Smk2 = self._project(Tmn_meas / rs.mu, pnps.Smk)
        # modify En
        if pnps.process == "shg":
            Smk2 *= ft.backward(Hmn * En).conj()
            Smk2 /= np.abs(Smk2)**(2/3)
            En = np.sum(Hmn.conj() * ft.forward(Smk2), axis=0)
        elif pnps.process == "thg":
            Smk2 *= ft.backward(Hmn * En).conj()**2
            Smk2 /= np.abs(Smk2)**(4/5)
            En = np.sum(Hmn.conj() * ft.forward(Smk2), axis=0)
        return R, En


class PIERetriever(StepRetriever):
    """ This class implements a pulse retrieval algorithm for SHG-FROG based on
    the ptychographical iterative engine (PIE). It is based on the paper
    [Sidorenko2016]_ and its erratum [Sidorenko2017]_.

    We modified the algorithm to include the scaling factor µ in the
    projection. This makes the method robust against initial guesses with the
    wrong magnitude. It should have no adverse effect.
    """
    method = "pie"
    supported_schemes = ["shg-frog"]

    def _retrieve_step(self, iteration, En):
        # local rename
        ft = self.ft
        pnps = self.pnps
        rs = self._retrieval_state
        Tmn_meas = self.Tmn_meas

        # running estimate for the trace
        Tmn = np.zeros((self.M, self.N))
        # random choice for the step size scaling
        beta = np.random.uniform(0.1, 0.5)
        for m in np.random.permutation(np.arange(self.M)):
            p = self.parameter[m]
            Tmn[m, :] = pnps.calculate(En, p)
            # get intermediate results from private attribute
            delay, Amk, Ek, Smk, _ = pnps._tmp[p]
            # project
            Smk2 = self._project(Tmn_meas[m, :] / rs.mu, Smk)
            # perform update
            Ek += beta * Amk.conj() * (Smk2 - Smk) / lib.abs2(Ek).max()
            # update the spectrum
            ft.forward(Ek, out=En)

        # Tmn is only an approximation as En changed in the iteration!
        rs.approximate_error = True
        R = self._R(Tmn)  # updates rs.mu!!!

        return R, En
