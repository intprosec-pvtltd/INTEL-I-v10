"""INTEL-I TensorRT production integration.

TensorRT is optional at Python import time. The application must never silently
fall back to an unvalidated engine when production TensorRT mode is requested.
"""
