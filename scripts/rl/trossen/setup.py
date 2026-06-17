"""Editable-install package for the Trossen Stationary AI cube-pickup task.

Install into the Isaac Lab container venv:
    podman exec isaaclab bash -lc "/opt/venv/bin/pip install -e /repo/scripts/rl/trossen"
Isaac Lab + rsl_rl are provided by the container image; not declared here.
"""

from setuptools import find_packages, setup

setup(
    name="trossen_cube",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
)
