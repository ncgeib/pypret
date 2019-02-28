import path_helper
import pypret

ft = pypret.FourierTransform(256, dt=5.0e-15)
pulse = pypret.Pulse(ft, 800e-9)

pulses = []
for i in range(100):
    pypret.random_pulse(pulse, 2.0)
    pulses.append(pulse.copy())

pypret.save(pulses, "pulse_bank.hdf5")
