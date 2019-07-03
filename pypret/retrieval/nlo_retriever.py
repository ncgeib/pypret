""" This module implements retrieval algorithms based on general
nonlinear optimization algorithms such as Levenberg-Marquadt,
differential evolution, or Nelder-Mead.
"""
import numpy as np
from scipy import optimize
from .retriever import BaseRetriever


class NLORetriever(BaseRetriever):

    def _scalar_objective(self, x):
        # rename
        rs = self._retrieval_state
        log = self.log
        # calculate trace error
        En = x.view(np.complex128)
        r = self._objective_function(En)
        R = self._Rr(r)
        # printing and logging
        if rs.nfev % 100 == 0 and self.verbose:
            print(rs.nfev, R)
        if self.logging:
            log.trace_error.append(R)
        rs.nfev += 1
        # normalize the error to avoid ill-scaling
        return r / self.Tmn_meas.max()**2

    def _vector_objective(self, x):
        # rename
        rs = self._retrieval_state
        log = self.log
        # calculate the error vector
        En = x.view(np.complex128)
        Tmn = self.pnps.calculate(En, self.parameter)
        diff = self._error_vector(Tmn, store=False)
        R = self._Rr(np.sum(diff * diff))
        # printing and logging
        if rs.nfev % 100 == 0 and self.verbose:
            print(rs.nfev, R)
        if self.logging:
            log.trace_error.append(R)
        rs.nfev += 1
        # normalize the error vector to avoid ill-scaling
        return diff / np.sqrt(self.M * self.N) / self.Tmn_meas.max()

    def _retrieve_begin(self, measurement, initial_guess, weights):
        super()._retrieve_begin(measurement, initial_guess, weights)
        rs = self._retrieval_state
        rs.nfev = 0

    def _retrieve_end(self):
        super()._retrieve_end()
        self._result.nfev = self._retrieval_state.nfev

    def result(self, pulse_original=None, full=True):
        res = super().result(pulse_original=pulse_original, full=full)
        res.nfev = self._retrieval_state.nfev
        return res


class LMRetriever(NLORetriever):
    """ Implements pulse retrieval based on the Levenberg-Marquadt algorithm.

    This is an efficient nonlinear least-squares solver, however, it will still
    be *very* slow for large pulses (N > 256). The reason is that the
    (MN x N) Jacobian is evaluated using numerical differentiation.

    The recommendation is to use this method either on small problems or to
    refine or verify solutions provided by a different algorithm.
    """
    method = "lm"

    def __init__(self, pnps, ftol=1e-08, xtol=1e-08, gtol=1e-08, lm_verbose=0,
                 **kwargs):
        """ For a full documentation of the arguments see :class:`Retriever`.

        For the documentation of `ftol`, `xtol`, `gtol` see the documentation
        of :func:`scipy.optimize.least_squares`. They are passed directly
        to the optimizer. If you want to run the optimizer for a fixed
        number of iterations, set all values to 1e-14 to effectively
        disable the stopping criteria.
        """
        super().__init__(pnps, ftol=ftol, xtol=xtol, gtol=gtol,
                         lm_verbose=lm_verbose, **kwargs)

    def _retrieve(self):
        # local rename
        o = self.options
        res = self._result
        # store current guess in attribute
        spectrum = self.initial_guess.copy()
        # This algorithm is not robust against the scaling of the input vector!
        spectrum /= np.abs(spectrum).max()
        x0 = spectrum.view(np.float64).copy()
        # calculate the maximum number of function evaluations
        max_nfev = None
        if o.maxfev is not None:
            max_nfev = o.maxfev // x0.shape[0]
        optres = optimize.least_squares(
                        self._vector_objective,
                        x0,
                        method='trf',
                        jac='2-point',
                        max_nfev=max_nfev,
                        tr_solver='exact',
                        ftol=o.ftol,
                        gtol=o.gtol,
                        xtol=o.xtol,
                        verbose=o.lm_verbose
                    )
        res.approximate_error = False
        res.spectrum = optres.x.view(dtype=np.complex128)
        res.trace_error = self.trace_error(res.spectrum)
        res.approximate_error = False
        return res.spectrum


class DERetriever(NLORetriever):
    """ This retriever uses the gradient-free differential evolution algorithm.

    It tries to match the parameters described in [Escoto2018]_ as far as
    they are mentioned. No further effort was made to optimize them. If you
    are interested in using DE as a pulse retrieval algorithm you are
    advised to study the documentation at
    :func:`scipy.optimize.differential_evolution`.

    The initial population in our implementation is based on the provided guess
    with added complex, Gaussian noise of 5% of the maximum amplitude.
    In our tests we saw no convergence when starting from completely
    random initial guesses.
    """
    method = "de"

    def _retrieve(self):
        # local rename
        o = self.options
        res = self._result
        # calculate the maximum number of function evaluations
        max_nfev = None
        if o.maxfev is not None:
            max_nfev = int(round(o.maxfev / 10 - 1))
        # generate initial population
        init = [self.initial_guess.view(np.float64).copy()]
        amp = np.abs(self.initial_guess).max()
        for i in range(9):
            sol = (self.initial_guess +
                   0.05 * amp * np.random.normal(size=self.N) +
                   0.05j * amp * np.random.normal(size=self.N))
            init.append(sol.view(np.float64))
        optres = optimize.differential_evolution(
                    self._scalar_objective,
                    [(-1.0, 1.0) for i in range(2 * self.N)],
                    strategy='rand1bin',
                    maxiter=max_nfev,
                    recombination=0.5,
                    popsize=10,  # is overwritten by init
                    tol=1e-7,
                    polish=False,
                    init=np.array(init)
                )
        res.approximate_error = False
        res.spectrum = optres.x.view(dtype=np.complex128)
        res.trace_error = self.trace_error(res.spectrum)
        res.approximate_error = False
        return res.spectrum


class NMRetriever(NLORetriever):
    """ This retriever uses the gradient-free Nelder-Mead algorithm.
    """
    method = "nm"

    def _retrieve(self):
        # local rename
        o = self.options
        res = self._result
        # store current guess in attribute
        spectrum = self.initial_guess.copy()
        # This algorithm is not robust against the scaling of the input vector!
        spectrum /= np.abs(spectrum).max()
        x0 = spectrum.view(np.float64).copy()
        # calculate the maximum number of function evaluations
        max_nfev = None
        if o.maxfev is not None:
            max_nfev = o.maxfev
        optres = optimize.minimize(
                    self._scalar_objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxfev': max_nfev},
                )
        res.approximate_error = False
        res.spectrum = optres.x.view(dtype=np.complex128)
        res.trace_error = self.trace_error(res.spectrum)
        res.approximate_error = False
        return res.spectrum


class BFGSRetriever(NLORetriever):
    """ This retriever uses the BFGS algorithm with numerical differentiation.
    """
    method = "bfgs"

    def _retrieve(self):
        # local rename
        o = self.options
        res = self._result
        # store current guess in attribute
        spectrum = self.initial_guess.copy()
        # This algorithm is not robust against the scaling of the input vector!
        spectrum /= np.abs(spectrum).max()
        x0 = spectrum.view(np.float64).copy()
        # calculate the maximum number of function evaluations
        max_nfev = None
        if o.maxfev is not None:
            max_nfev = o.maxfev // x0.shape[0]
        optres = optimize.minimize(
                    self._scalar_objective,
                    x0,
                    method='BFGS',
                    options={'maxiter': max_nfev},
                )
        res.approximate_error = False
        res.spectrum = optres.x.view(dtype=np.complex128)
        res.trace_error = self.trace_error(res.spectrum)
        res.approximate_error = False
        return res.spectrum
