"""Trossen Stationary AI cube-pickup teacher/student task package.

Importing this package registers the gym ids (via :mod:`trossen_cube.tasks`).

IMPORTANT: import this only AFTER launching ``isaaclab.app.AppLauncher`` /
``SimulationApp`` -- the task configs import ``isaaclab.*``, which needs the USD
(``pxr``) runtime that only exists once Kit has booted.
"""

from . import tasks  # noqa: F401  (registers gym ids on import)
