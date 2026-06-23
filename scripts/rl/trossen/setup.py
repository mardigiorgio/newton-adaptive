"""Editable-install package for the Trossen Stationary AI cube-pickup task.

Install into the native Isaac Sim interpreter (the binary ships its own python3.11):
    ~/Documents/code/IsaacLab/isaaclab.sh -p -m pip install -e scripts/rl/trossen
Isaac Lab + rsl_rl are provided by ``isaaclab.sh --install``; not declared here.
"""

from setuptools import find_packages, setup

setup(
    name="trossen_cube",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
)
