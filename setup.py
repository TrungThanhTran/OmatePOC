# Compatibility shim for older pip/setuptools versions (Python 3.10).
# All configuration lives in pyproject.toml — this file just enables
# the legacy editable-install path for pip < 21.3.
from setuptools import setup

setup()
