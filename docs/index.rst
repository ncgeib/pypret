pypret
======

:Release: |release|
:Date: |today|

This is the documentation of `Python for pulse retrieval`. It is a Python
package that aims to provide algorithms and tools to retrieve ultrashort
laser pulses from parametrized nonlinear process spectra, such as
frequency-resolved optical gating (FROG), dispersion scan (d-scan),
time-domain ptychography (TDP) or multiphoton intrapulse interference phase
scan (MIIPS).

The package is currently in an early alpha state. It provides the
algorithms but still requires thorough understanding of what they do to apply
them correctly on measured data.

Background
----------

The package was developed at the `Institute of Applied Physics`_ at the
`Friedrich Schiller University Jena`_. Main author is Nils C. Geib. You
can reach me at nils.geib@uni-jena.de if you have questions or comments on
the code.

The current capabilities of the package reflect mostly what we
presented in our publication on a common pulse retrieval algorithm [Geib2019]_.
If you want to reference this package you may cite that paper.

The code in its current state mainly serves to give a reference implementation
of the algorithms discussed within and allow the reproduction of our results.
It is planned, however, to expand the package to make it a more full-fledged
solution for pulse retrieval.

.. _`Institute of Applied Physics`: https://www.iap.uni-jena.de
.. _`Friedrich Schiller University Jena`: https://www.uni-jena.de


User documentation
------------------

.. toctree::
    :maxdepth: 1
    
    installation
    getting_started
    references

API documentation
-----------------

.. toctree::
    :maxdepth: 1
    
    apidoc/pypret.fourier
    apidoc/pypret.pulse
    apidoc/pypret.pnps
    apidoc/pypret.retrieval
    apidoc/pypret.pulse_error
    apidoc/pypret.io
    apidoc/pypret.lib
    apidoc/pypret.frequencies
    apidoc/pypret.material
    apidoc/pypret.mesh_data
    apidoc/pypret.graphics
