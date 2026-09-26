"""Compat shim: the macOS pymicro-features fork (puddly/minimum-cpp-version)
only exposes PascalCase ProcessSamples, but microwakeword calls process_samples.
"""
from pymicro_features import MicroFrontend

MicroFrontend.process_samples = MicroFrontend.ProcessSamples
