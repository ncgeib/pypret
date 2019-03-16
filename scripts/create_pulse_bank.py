""" This script generates 100 pulses with time-bandwidth product 2 and
stores them in an HDF5 file. Those pulses are used by other scripts
in this folder.
"""
import path_helper
import pypret

ft = pypret.FourierTransform(256, dt=5.0e-15)
pulse = pypret.Pulse(ft, 800e-9)

pulses = []
for i in range(100):
    pypret.random_pulse(pulse, 2.0)
    pulses.append(pulse.copy())

pypret.save(pulses, "pulse_bank.hdf5")
