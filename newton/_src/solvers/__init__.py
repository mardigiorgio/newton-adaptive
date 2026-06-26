# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from .featherstone import SolverFeatherstone
from .flags import SolverNotifyFlags
from .implicit_mpm import SolverImplicitMPM
from .kamino import SolverKamino
from .mujoco import SolverMuJoCo, SolverMuJoCoAdaptive
from .semi_implicit import SolverSemiImplicit
from .solver import SolverBase
from .style3d.solver_style3d import SolverStyle3D
from .vbd import SolverVBD
from .xpbd import SolverXPBD

# SAP is vendored from an external ``sap_warp`` checkout via a sys.path shim, so its
# import is guarded: a missing/broken sap_warp must not break the rest of Newton.
try:
    from .sap import SolverSAP, SolverSAPAdaptive
except Exception as _sap_exc:  # noqa: BLE001
    SolverSAP = None
    SolverSAPAdaptive = None
    _SAP_IMPORT_ERROR = _sap_exc
else:
    _SAP_IMPORT_ERROR = None

__all__ = [
    "SolverBase",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverKamino",
    "SolverMuJoCo",
    "SolverMuJoCoAdaptive",
    "SolverNotifyFlags",
    "SolverSAP",
    "SolverSAPAdaptive",
    "SolverSemiImplicit",
    "SolverStyle3D",
    "SolverVBD",
    "SolverXPBD",
]
