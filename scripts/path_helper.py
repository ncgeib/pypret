""" A small helper that allows to import from the pypret package
in this git repository without any changes to PYTHONPATH.
"""
# first try if pypret is installed
import importlib
spec = importlib.util.find_spec("pypret")
if spec is None or spec.origin == "namespace":
    # if pypret is not installed
    # add relative path with high priority
    import sys
    from pathlib import Path
    pypret_folder = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(pypret_folder))
