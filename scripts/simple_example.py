""" This shows a simple application of pypret.

It calculates a PNPS trace from a pulse and displays it.
"""
import path_helper
import numpy as np
import pypret
from pypret import (FourierTransform, Pulse, random_pulse, PNPS, MeshDataPlot)

ft = FourierTransform(256, dt=2.5e-15)
pulse = Pulse(ft, 800e-9)
random_pulse(pulse, 2.0)

method = "ifrog"
process = "shg"

if method == "miips":
    # MIIPS
    parameter = np.linspace(0.0, 2.0*np.pi, 128)  # delta in rad
    pnps = PNPS(pulse, method, process, gamma=22.5e-15, alpha=1.5 * np.pi)
elif method == "dscan":
    # d-scan
    parameter = np.linspace(-0.025, 0.025, 128)  # insertion in m
    pnps = PNPS(pulse, method, process, material=pypret.material.BK7)
elif method == "ifrog":
    # ifrog
    if process == "sd":
        parameter = np.linspace(pulse.t[0], pulse.t[-1], pulse.N * 4)
    else:
        parameter = pulse.t  # delay in s
    pnps = PNPS(pulse, method, process)
elif method == "frog":
    # frog
    parameter = pulse.t  # delay in s
    pnps = PNPS(pulse, method, process)
elif method == "tdp":
    # d-scan
    parameter = np.linspace(pulse.t[0], pulse.t[-1], 128)  # delay in s
    pnps = PNPS(pulse, method, process, center=790e-9, width=10.6e-9)
else:
    raise ValueError("Method not supported!")
pnps.calculate(pulse.spectrum, parameter)

# Example how to save the calculation
#pnps.save("test_pnps.hdf5")
#pnps2 = pypret.load("test_pnps.hdf5")

md = pnps.trace
md.autolimit(1)
mdp = MeshDataPlot(md, show=False)
mdp.ax.set_title("SHG-iFROG")

mdp.show()