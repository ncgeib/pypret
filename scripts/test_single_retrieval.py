import path_helper
import pypret
from benchmarking import benchmark_retrieval, RetrievalResultPlot

scheme = ( # can be one of the following
#        "shg-frog"
#        "pg-frog"
#        "shg-tdp"
        "shg-dscan"
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
res = benchmark_retrieval(pulses[1], scheme, "copra", repeat=1,
                          verbose=True, max_iterations=300,
                          additive_noise=0.01)
rrp = RetrievalResultPlot(res.retrievals[0])
#rrp.fig.savefig("result.png")
