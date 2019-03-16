""" This script compares pulse retrieval using different general-purpose,
nonlinear optimization algorithms.

It can be used to reproduce the results shown in Fig. 2 in our paper
[Geib2019]_.
"""
import path_helper
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from pypret import (FourierTransform, Pulse, random_gaussian, random_pulse,
                    PNPS, Retriever, load, save)
from pypret.graphics import plot_meshdata

# CONFIG
repeat = 10  # how often to repeat the retrieval
algorithms = ['lm', 'bfgs', 'de', 'nm']  # algorithms to test
verbose = False  # print out the error during iteration

# %%
# Create a simulation grid
ft = FourierTransform(64, dt=20.0e-15)
# instantiate a pulse with central frequency 800 nm
pulse = Pulse(ft, 800e-9)
# create a random, localized pulse with time-bandwidth product 1
random_pulse(pulse, 1.0, edge_value=1e-7)

# instantiate an SHG-FROG measurement
parameter = pulse.t  # delay in s
pnps = PNPS(pulse, 'frog', 'shg')
# simulate the noiseless measurement
original_spectrum = pulse.spectrum
pnps.calculate(pulse.spectrum, parameter)
measurement = pnps.trace

data = SimpleNamespace(measurement=measurement, pulse=pulse,
                       results={})
for algorithm in algorithms:
    ret = Retriever(pnps, algorithm, verbose=verbose, logging=True,
                    maxfev=20000)
    for i in range(repeat):
        print("Running algorithm %s run %d/%d" % (algorithm.upper(), i+1,
                                                  repeat))
        # create initial pulse, Gaussian in time domain
        random_gaussian(pulse, 50e-15)
        ret.retrieve(measurement, pulse.spectrum)
        res = ret.result(original_spectrum)
        print("Finished after %d function evaluations." % res.nfev)
        print("final trace error R=%.15e" % res.trace_error)
        # store the result with the best trace error
        if i == 0 or res.trace_error < data.results[algorithm].trace_error:
            data.results[algorithm] = res

# save simulation data for further plotting
save(data, "nlo_retriever_data.hdf5")

# %%
# Plot: this part can be run separately
data = load("nlo_retriever_data.hdf5")
fig, (ax1, ax2) = plt.subplots(1, 2)

# mesh plot of the SHG-FROG trace
im = plot_meshdata(ax1, data.measurement, cmap="nipy_spectral")
ax1.set_title("SHG-FROG")

for algorithm in algorithms:
    res = data.results[algorithm]
    iterations = np.arange(res.trace_errors.size)
    ax2.plot(iterations, np.minimum.accumulate(res.trace_errors),
             label=algorithm.upper())
ax2.set_yscale('log')
ax2.set_xlabel("function evaluations")
ax2.set_ylabel("trace error R")
ax2.legend(loc="best")
fig.tight_layout()
plt.show()
