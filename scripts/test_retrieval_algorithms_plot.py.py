""" This script plots the results obtained by `test_retrieval_algorithms.py`.
Therefore, that script has to be run first.

It reproduces Fig. 4, 5 and 7 from [Geib2019]_.
"""
import path_helper
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import pypret

# the configs to plot: (scheme, algorithm) separated by subplot
configs = [
    ("shg-frog", "copra"),
    ("shg-frog", "pcgpa"),
    ("shg-frog", "pie"),
    ("pg-frog", "copra"),
    ("shg-tdp", "copra"),
    ("shg-dscan", "copra"),
    ("thg-dscan", "copra"),
    ("sd-dscan", "copra"),
    ("shg-ifrog", "copra"),
    ("thg-ifrog", "copra"),
    ("sd-ifrog", "copra"),
    ("shg-miips", "copra"),
    ("thg-miips", "copra"),
    ("sd-miips", "copra")
]
noise_levels = [0.0, 1e-2, 3e-2]

def get_fname(scheme, algorithm, noise):
    return "%s_%s_noise_%.1e.hdf5.7z" % (scheme.upper(),
                                         algorithm.upper(), noise)

# %%
path = Path("results")
results = {}
for scheme, algorithm in configs:
    for noise in noise_levels:
        # iterate over configs
        fname = get_fname(scheme, algorithm, noise)
        ares = SimpleNamespace(pulse_errors=[], trace_errors=[],
                               relative_trace_errors=[],
                               relative_rm_trace_errors=[])
        # iterate over retrieved pulses
        for res in pypret.load(path / fname, archive=True):
            # for every pulse several retrievals were made
            # select solution with lowest trace error
            pres = res.retrievals[np.argmin([r.trace_error
                                             for r in res.retrievals])]
            # now store the final trace error of the solution
            ares.trace_errors.append(pres.trace_error)
            # relative trace error (R - R0 in the paper)
            ares.relative_trace_errors.append(pres.trace_error -
                                              pres.trace_error_optimal)
            # the running minimum of the trace error minus the optimal
            # trace error (R - R0 in the paper) - for plotting
            ares.relative_rm_trace_errors.append(pres.rm_trace_errors -
                                                 pres.trace_error_optimal)
            # the pulse error of the solution
            ares.pulse_errors.append(pres.pulse_error)
        # convert to numpy arrays and calculate the median
        ares.trace_errors = np.array(ares.trace_errors)
        ares.relative_trace_errors = np.array(ares.relative_trace_errors)
        ares.relative_rm_trace_errors = np.array(ares.relative_rm_trace_errors)
        ares.pulse_errors = np.array(ares.pulse_errors)
        # store in dictionary
        results[fname] = ares

pypret.save(results, path / "plot_results.hdf5")

# %%
# print out the results (equivalent to Fig. 7)
results = pypret.load(path / "plot_results.hdf5")
# print numerical results
s = "filename".ljust(40)
s += "trace error".ljust(15)
s += "R - R0".ljust(15)
s += "pulse_error".ljust(15)
print(s)
for scheme, algorithm in configs:
    for noise in noise_levels:
        fname = "%s_%s_noise_%.1e.hdf5.7z" % (scheme.upper(),
                                              algorithm.upper(), noise)
        ares = results[fname]
        s = fname.ljust(40)
        s += ("%.3e" % np.median(ares.trace_errors)).ljust(15)
        s += ("%.3e" % np.median(ares.relative_trace_errors)).ljust(15)
        s += ("%.3e" % np.median(ares.pulse_errors)).ljust(15)
        print(s)

# %%
# do the plots (equivalent to Fig. 4 and 5)
plot_configs = [
    [("shg-frog", "copra"),
     ("shg-frog", "pcgpa"),
     ("shg-frog", "pie")],
    [("shg-ifrog", "copra"),
     ("thg-ifrog", "copra"),
     ("sd-ifrog", "copra")],
    [("shg-dscan", "copra"),
     ("thg-dscan", "copra"),
     ("sd-dscan", "copra")],
    [("shg-miips", "copra"),
     ("thg-miips", "copra"),
     ("sd-miips", "copra")]
]

# first plot no noise case
noise = 0.0
fig, axs = plt.subplots(2, 2, figsize=(20.0/2.54, 20.0/2.54))
for config, ax in zip(plot_configs, axs.flat):
    for scheme, algorithm in config:
        fname = get_fname(scheme, algorithm, noise)
        ares = results[fname]
        errors = np.median(ares.relative_rm_trace_errors, axis=0)
        iterations = np.arange(errors.size)
        ax.plot(iterations, errors, label=scheme + " " + algorithm)
    ax.set_xlabel("iterations")
    ax.set_ylabel("R (running minimum)")
    ax.legend(loc="best")
    ax.set_yscale("log")
fig.tight_layout()
plt.show()

# second plot 1% noise case
noise = 0.01
fig, axs = plt.subplots(2, 2, figsize=(20.0/2.54, 20.0/2.54))
for config, ax in zip(plot_configs, axs.flat):
    for scheme, algorithm in config:
        fname = get_fname(scheme, algorithm, noise)
        ares = results[fname]
        errors = np.median(ares.relative_rm_trace_errors, axis=0)
        iterations = np.arange(errors.size)
        ax.plot(iterations, errors, label=scheme + " " + algorithm)
    ax.set_xlabel("iterations")
    ax.set_ylabel("R - R0 (running minimum)")
    ax.legend(loc="best")
    ax.set_yscale("symlog", linthreshy=1e-5)
fig.tight_layout()
plt.show()
