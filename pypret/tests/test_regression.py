""" This test spots regressions in the trace calculation and the
retrieval algorithm by comparing their outputs against ones obtained by a
previous version of the code.

It does not necessarily test the correctness of the calculations - simply
that they did not change.
"""
from pathlib import Path
import numpy as np
import pypret
from pypret import PNPS, material


def get_pnps(pulse, method, process):
    if method == "miips":
        parameter = np.linspace(0.0, 2.0*np.pi, pulse.N//2)  # delta in rad
        pnps = PNPS(pulse, method, process, gamma=22.5e-15, alpha=1.5 * np.pi)
    elif method == "dscan":
        parameter = np.linspace(-0.025, 0.025, pulse.N//2)  # insertion in m
        pnps = PNPS(pulse, method, process, material=material.BK7)
    elif method == "ifrog":
        parameter = pulse.t
        pnps = PNPS(pulse, method, process)
    elif method == "frog":
        parameter = pulse.t
        pnps = PNPS(pulse, method, process)
    elif method == "tdp":
        parameter = np.linspace(pulse.t[0], pulse.t[-1], pulse.N//2)
        pnps = PNPS(pulse, method, process, center=790e-9, width=10.6e-9)
    else:
        raise ValueError("Method not supported!")
    return pnps, parameter


def test_regression():
    # test if a test pulse already exists
    dirname = Path(__file__).parent / Path("data")
    pulse_path = dirname / Path("pulse.hdf5")
    if not pulse_path.exists():
        # create simulation grid
        ft = pypret.FourierTransform(64, dt=5.0e-15)
        # instantiate a pulse object, central wavelength 800 nm
        pulse = pypret.Pulse(ft, 800e-9)
        # create a random pulse with time-bandwidth product of 2.
        pypret.random_pulse(pulse, 1.0, edge_value=1e-8)
        # store the pulse
        pulse.save(pulse_path, archive=True)
    # initial pulse
    initial_path = dirname / Path("initial.hdf5")
    if not initial_path.exists():
        pulse = pypret.load(pulse_path, archive=True)
        pypret.random_gaussian(pulse, 50e-15)
        pulse.save(initial_path, archive=True)
    initial = pypret.load(initial_path, archive=True).spectrum

    # =========================================================================
    # Test for regressions in the trace calculation
    # =========================================================================
    pulse = pypret.load(pulse_path, archive=True)
    for method, dct in pypret.pnps._PNPS_CLASSES.items():
        for process, cls in dct.items():
            scheme = process + "-" + method
            pnps, parameter = get_pnps(pulse, method, process)
            pnps.calculate(pulse.spectrum, parameter)
            trace = pnps.trace
            trace_path = (dirname / 
                          Path("%s-%s-trace.hdf5" % (method, process)))
            # store if not available
            if not trace_path.exists():
                pypret.save(trace.data, trace_path, archive=True)
            else:
                trace2 = pypret.load(trace_path, archive=True)
                assert pypret.lib.nrms(trace.data, trace2) < 1e-15
                # use the stored values for the next test
                trace.data = trace2

            # test for regressions in the retrieval algorithms
            retrieval_algorithms = ["copra"]
            if scheme in ["shg-dscan", "thg-dscan"]:
                retrieval_algorithms += ["gp-dscan"]
            if scheme == "shg-frog":
                retrieval_algorithms += ["gpa", "pcgpa", "pie"]

            for algorithm in retrieval_algorithms:
                fname = (dirname /
                         Path("%s-%s-%s-retrieved.hdf5" %
                              (method, process, algorithm)))
                ret = pypret.Retriever(pnps, algorithm, maxiter=10, maxfev=256)
                np.random.seed(1234)
                ret.retrieve(trace, initial)
                result = ret._result.spectrum
                if not fname.exists():
                    pypret.save(result, fname)
                else:
                    result2 = pypret.load(fname)
                    assert pypret.lib.nrms(result, result2) < 1e-8


if __name__ == "__main__":
    test_regression()
