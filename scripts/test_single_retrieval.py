""" This module shows how to use the benchmark script. It triggers
a single retrieval simulation and plots the results. It can be used
to quickly assess the performance of a single algorithm/measurement scheme
combination.
"""
import path_helper
import pypret
from benchmarking import benchmark_retrieval, RetrievalResultPlot

scheme = ( # can be one of the following
        "shg-frog"
#        "pg-frog"
#        "tg-frog"
#        "shg-tdp"
#        "shg-dscan"
#        "thg-dscan"
#        "sd-dscan"
#        "shg-ifrog"
#        "thg-ifrog"
#        "sd-ifrog"
#        "shg-miips"
#        "thg-miips"
#        "sd-miips"        
        )

pulses = pypret.load("pulse_bank.hdf5")
res = benchmark_retrieval(pulses[2], scheme, "copra", repeat=1,
                          verbose=True, maxiter=300,
                          additive_noise=0.01)
rrp = RetrievalResultPlot(res.retrievals[0])
rrp.fig.savefig("result.png")
