# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""ActiveSetMaskingMixin — for XPBD / SemiImplicit step_fns that accept
a world_active mask in their kernels."""

import warp as wp

from scripts.adaptive import kernels as K


class ActiveSetMaskingMixin:
    """Maintain a world_active mask and pass it through step_fn.

    Concrete class composition: class AdaptiveMaskingWrapper(ActiveSetMaskingMixin, AdaptiveWrapper).
    Mixin MUST come before AdaptiveWrapper in MRO so its __init__ runs and
    overrides participate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = self.model.world_count
        device = self.model.device
        self._world_active = wp.full(n, True, dtype=wp.bool, device=device)

    def _reset_active_mask(self):
        self._world_active.fill_(True)

    def _update_active_after_advance(self):
        n = self.model.world_count
        device = self.model.device
        wp.launch(
            K._update_active_mask, dim=n,
            inputs=[self._sim_time, self._next_time, self._world_active],
            device=device,
        )

    def _run_iteration_body(self, effective_dt_max: float) -> None:
        # Call base implementation, then update active mask.
        super()._run_iteration_body(effective_dt_max)
        self._update_active_after_advance()

    def step_dt(self, dt_outer, state_0, state_1, control):
        self._reset_active_mask()
        return super().step_dt(dt_outer, state_0, state_1, control)

    @property
    def world_active(self) -> wp.array:
        return self._world_active
