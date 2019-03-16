""" This script tests COPRA against a lot of different PNPS schemes
and compares it against PCGPA and ptychographic retrieval for SHG-FROG.
It reproduces the data of Fig. 4, 5 and 7 from [Geib2019]_.

Notes
-----
As we are using multiprocessing to speed up the parameter scan you may not be 
able to run this script inside of an IDE such as spyder. In that case
please run the script from the commandline using standard Python.

For plotting the results see `test_retrieval_algorithms_plot.py`.
"""
import path_helper
import pypret
from benchmarking import benchmark_retrieval, RetrievalResultPlot
from pathlib import Path
from concurrent import futures

# the configs to test: (scheme, algorithm)
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
maxworkers = 10  # number of processes used
maxiter = 100  # 300 in the paper
npulses = 10  # 100 in the paper
repeat = 3  # 10 in the paper
# [0.0, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2] in the paper
noise_levels = [1e-2, 3e-2]

# block the main routine as we are using multiprocessing
if __name__ == "__main__":
    pulses = pypret.load("pulse_bank.hdf5")

    path = Path("results")
    if not path.exists():
        path.mkdir()

    for scheme, algorithm in configs:
        for noise in noise_levels:
            print("Testing %s with %s and noise level %.1f%%" %
                  (scheme.upper(), algorithm.upper(), noise * 100))
            results = []
            # run the different pulses in different processes,
            # not optimal but better than no parallelism
            fs = {}
            with futures.ProcessPoolExecutor(max_workers=maxworkers) as executor:
                for i, pulse in enumerate(pulses[:npulses]):
                    future = executor.submit(benchmark_retrieval,
                                    pulses, scheme, algorithm, repeat=repeat,
                                    verbose=False, maxiter=maxiter,
                                    additive_noise=noise)
                    fs[future] = i
                for future in futures.as_completed(fs):
                    i = fs[future]
                    try:
                        res = future.result()
                    except Exception as exc:
                        print('Retrieval generated an exception: %s' % exc)
                    results.append(res)
                    print("Finished pulse %d/%d" % (i+1, npulses) )
            fname = "%s_%s_noise_%.1e.hdf5.7z" % (scheme.upper(),
                                                  algorithm.upper(), noise)
            pypret.save(results, path / fname, archive=True)
            print("Stored in %s" % fname)
