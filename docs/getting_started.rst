Getting started
===============

pypret is a package to simulate and retrieve from measurements such as
frequency-resolved optical gating (FROG), dispersion scan (d-scan),
interferometric FROG (iFROG), time-domain ptychography (TDP) and even
multiphoton intrapulse interference phase scan (MIIPS). These are all
measurements used for ultrashort (sub-ps) laser pulse measurement. More
generally the package can handle all kinds of parametrized nonlinear
process spectra (PNPS) measurements.

A good place to start reading on the algorithms and the used notation is
our paper [Geib2019]_ and its supplement. pypret can be thought to accompany
this publication and can be used to reproduce most of the results shown there.

Basic Use
---------
pypret can be used to simulate PNPS measurements. This is useful for designing
experiments and necessary for retrieval, of course.

In a first step you have to set up the simulation grid in time and frequency::
    
    ft = pypret.FourierTransform(256, dt=2.5e-15)
    
which generates a 256 elements grid with a temporal spacing of 2.5 fs centered
around t=0. The frequency grid is chosen to match the reciprocity relation
``dt * dw = 2 * pi / N``. Alternatively you can specify the frequency spacing.
See the documentation at :doc:`apidoc/pypret.fourier`.
Next you can instantiate a :class:`pypret.Pulse` object::

    pulse = pypret.Pulse(ft, 800e-9)
    
where we used a central wavelength of 800 nm. This class can already be used
for small but useful calculations::

    # generate pulse with Gaussian spectrum and field standard deviation
    # of 20 nm
    pulse.spectrum = pypret.lib.gaussian(pulse.wl, x0=800e-9, sigma=20e-9)
    # print the accurate FWHM of the temporal intensity envelope
    print(pulse.fwhm(dt=pulse.dt/100))
    # propagate it through 1cm of BK7 (remove first ord)
    phase = np.exp(1.0j * pypret.material.BK7.k(pulse.wl) * 0.01)
    pulse.spectrum = pulse.spectrum * phase
    # print the temporal FWHM again
    print(pulse.fwhm(dt=pulse.dt/100))
    # finally plot the pulse
    pypret.graphics.PulsePlot(pulse)

You can now instantiate a PNPS class with that pulse object::

    insertion = np.linspace(-0.025, 0.025, 128)  # insertion in m
    pnps = pypret.PNPS(pulse, "dscan", "shg", material=pypret.material.BK7)
    # calculate the measurement trace
    pnps.calculate(pulse.spectrum, delay)
    original_spectrum = pulse.spectrum
    # and plot it
    pypret.MeshDataPlot(pnps.trace)
    
The PNPS constructor supports a lot of different PNPS measurements (see docs
at :doc:`apidoc/pypret.pnps`). Furthermore, it is easy to implement your own.

Finally, you can use pypret for pulse retrieval by instantiating a Retriever
object::

    # do the retrieval
    ret = pypret.Retriever(pnps, "copra", verbose=True, max_iterations=300)
    # start with a Gaussian spectrum with random phase as initial guess
    pypret.random_gaussian(pulse, 50e-15, phase_max=0.0)
    # now retrieve from the synthetic trace simulated above
    ret.retrieve(pnps.trace, pulse.spectrum)
    # and print the retrieval results
    ret.result(original_spectrum)
    
A lot of different retrieval algorithms besides the default, COPRA, are
implemented (see docs at :doc:`apidoc/pypret.retrieval`). While COPRA should
work for all PNPS measurements, you may try one of the others for verification.

Storage
-------
The :doc:`apidoc/pypret.io` subpackage supports saving almost arbitrary Python
structures and all pypret classes to HDF5 files. You can either use the
:func:`pypret.save` function or the `save` method on classes::
 
    pnps.calculate(pulse.spectrum, insertion)
    pnps.trace.save("trace.hdf5")
    # or
    pypret.save(pnps.trace, "trace.hdf5")
    # load it with
    trace = pypret.load("trace.hdf5")

This should make storing intermediate or final results almost effortless.

Experimental data
-----------------
As this question is surely going to come: you can use pypret to retrieve pulses
from experimental data, however, it currently has no pre-processing functions
to make that convenient. The data fed to the retrieval functions has to be
properly dark-subtracted and interpolated. Furthermore, some features that are
very useful for retrieval from experimental data (e.g., handling non-calibrated
traces) are not yet implemented. This is on the top of the ToDo-list, though.
