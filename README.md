# Python for Pulse Retrieval

This project aims to provide numerical algorithms for ultrashort laser pulse measurement methods such as frequency-resolved optical gating (FROG), dispersion scan (d-scan), or time-domain ptychography (TDP) and more. Specifically, it provides a reference implementation of the algorithms presented in our paper "Common pulse retrieval algorithm: a fast and universal method to retrieve ultrashort pulses". The paper is currently available as a [pre-print](https://arxiv.org/abs/1810.04780).

![Example output](scripts/result.png?raw=true "Result")

## Notes

This code is a complete re-implentation of the (rather messy) code used in our research. It was created with the expressive purpose to be well-documented and educational. The notation in the code tries to match the notation in the paper and references it. I would strongly recommend reading the publication before diving into the code.

As a down-side the code is not optimized and on many occasions I deliberately decided to go with the less efficient but more expressive and straightforward solution. This pains me somewhat and I do not recommend to use the code as an example for high-performance, numerical Python. It creates unecessarily many temporal copies and re-calculates many values that could be stored.

## Documentation

The full documentation can be found at [pypret.readthedocs.io](https://pypret.readthedocs.io). The ``scripts`` folder contains examples of how the package is used.

### Usage example
The code is contained in the plain Python package ``pypret`` (PYthon for Pulse Retrieval). Point your PYTHONPATH to it and you can use it. The package contains many classes that are in general useful for ultrashort pulse simulations. The most iconic usage, however, is to simulate pulse measurement schemes and perform pulse retrieval:

```python
import numpy as np
import pypret
# create simulation grid
ft = pypret.FourierTransform(256, dt=5.0e-15)
# instantiate a pulse object, central wavelength 800 nm
pulse = pypret.Pulse(ft, 800e-9)
# create a random pulse with time-bandwidth product of 2.
pypret.random_pulse(pulse, 2.0)
# plot the pulse
pypret.PulsePlot(pulse)

# simulate a d-scan measurement
insertion = np.linspace(-0.025, 0.025, 128)  # insertion in m
pnps = pypret.PNPS(pulse, "dscan", "shg", material=pypret.material.BK7)
# calculate the measurement trace
pnps.calculate(pulse.spectrum, insertion)
original_spectrum = pulse.spectrum
# and plot it
pypret.MeshDataPlot(pnps.trace)

# and do the retrieval
ret = pypret.Retriever(pnps, "copra", verbose=True, max_iterations=300)
# start with a Gaussian spectrum with random phase as initial guess
pypret.random_gaussian(pulse, 50e-15, phase_max=0.0)
# now retrieve from the synthetic trace simulated above
ret.retrieve(pnps.trace, pulse.spectrum)
# and print the retrieval results
ret.result(original_spectrum)
```
The text output should look similar to this:
```
Retrieval report
trace error                    R = 2.07853378314943602e-07
min. trace error               R0 = 0.00000000000000000e+00
                          R - R0 = 2.07853378314943602e-07

pulse error                    ε = 3.24609262599545714e-05
```
This shows that the retrieval converged to the input trace within almost the numerical accuracy of the underlying calculations. This is, of course, only possible if no noise was added to the input data. Occasionally, the algorithm will not converge - in which case you would have to run it again.

More elaborate examples of how to use this package can be found in the scripts directory.

## Author

This project was developed by Nils C. Geib at the [Institute of Applied Physics](https://www.iap.uni-jena.de) of the [University of Jena](https://www.uni-jena.de), Germany.

For any questions or comments, you can contact me via email: nils.geib@uni-jena.de

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
