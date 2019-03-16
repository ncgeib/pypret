""" This sub-package implements different pulse retrieval algorithms
using a common interface. This facilitates direct comparison.

All algorithms are implemented as a subclass of :class:`BaseRetriever`. The
algorithms which are implemented step-by-step, i.e., do not rely on some
monolithic minimization algorithm implemented elsewhere, are further
subclassed from :class:`StepRetriever`.

The public function to instantiate the correct retriever is :func:`Retriever`.

This sub-package does not implement any form of data pre-processing. It expects
correctly interpolated data in form of a MeshData object.
"""
from .retriever import Retriever
from .step_retriever import (COPRARetriever, PCGPARetriever, GPARetriever,
                             GPDSCANRetriever, PIERetriever)
from .nlo_retriever import LMRetriever, NMRetriever, DERetriever, BFGSRetriever