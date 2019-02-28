""" A small helper that allows to import from the pypret package
without installing it.
"""
import sys
from pathlib import Path
pypret_folder = Path(__file__).resolve().parents[1]
# add with high priority
sys.path.insert(0, str(pypret_folder))
