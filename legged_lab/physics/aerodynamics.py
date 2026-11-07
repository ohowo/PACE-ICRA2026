# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Renhong Zhang

import math
import torch
from typing import Literal, Optional, Tuple

from legged_lab.physics.aero_model import (
    MeasuredAeroModel
)
from legged_lab.physics.magnus_model import ContiMagnusModel

class AeroForceField:
    """
    Isaac Lab adapter for the GPU-batched aerodynamic models.
    Works with either a single RigidObject (one body per env) or a RigidObjectCollection.

    air_velocity_w is the optional uniform air velocity (x, y, z) in world-frame m/s.
    By default the air is still.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        # ball + air properties
        radius_m: float = 0.020,
        air_density: float = 1.20,
        drag_coeff: float = 0.43,
        magnus_factor: float = 0.0,
        *,
        air_velocity_w: Optional[Tuple[float, float, float]] = None,
        magnus_model: Literal["constant", "conti2026"] = "constant",
    ):
        if magnus_model not in ("constant", "conti2026"):
            raise ValueError("magnus_model must be 'constant' or 'conti2026'")
        if magnus_model == "conti2026" and not math.isclose(radius_m, 0.020, rel_tol=1e-6):
            raise ValueError("conti2026 is calibrated for a nominal 0.020 m radius")
        # Constant mode preserves the old magnus_factor behavior. Dynamic mode
        # computes kappa from current speed/spin and does not use magnus_factor.
        self.magnus_model = magnus_model
        self._dynamic_magnus = ContiMagnusModel() if magnus_model == "conti2026" else None
        self.device = torch.device(device)
        self.env_props = {"air_density": air_density}
        self.r = radius_m
        self.a = math.pi * (radius_m ** 2)

        self._air_velocity_w = None
        if air_velocity_w is not None:
            air_velocity = torch.tensor(air_velocity_w, dtype=torch.float32, device=self.device)
            if air_velocity.shape != (3,):
                raise ValueError("air_velocity_w must contain three world-frame components")
            self._air_velocity_w = air_velocity

        
        self.model = MeasuredAeroModel(device=device)
        self._fixed_params = {"drag_coefficients": drag_coeff, "magnus_factors": magnus_factor}

        # Buffers (lazily sized to match the asset)
        self._ball_props = None  # dict[str, torch.Tensor]
        self._model_params = None

    def drag_acceleration_coefficient(self, mass_kg: float) -> float:
        """Return k = rho * A * Cd / (2 * mass) in inverse meters."""
        return 0.5 * self.env_props["air_density"] * self.a * self._fixed_params["drag_coefficients"] / mass_kg

    @property
    def magnus_enabled(self) -> bool:
        """Whether the configured force law can produce Magnus lift."""
        return self._dynamic_magnus is not None or self._fixed_params["magnus_factors"] != 0.0

    def compute_forces(self, v_w: torch.Tensor, omega_w: torch.Tensor):
        """Evaluate flight forces without staging forces on a live body."""
        return self._compute_forces(v_w, omega_w)

    def _ensure_buffers(self, batch_size: int):
        """Create constant parameter tensors on the correct device and shape (N,)."""
        # (Re)build only if sizes changed
        if self._ball_props is None or next(iter(self._ball_props.values())).numel() != batch_size:
            dev = self.device
            self._ball_props = {
                "radii":     torch.full((batch_size,), self.r, dtype=torch.float32, device=dev),
                "areas":     torch.full((batch_size,), self.a, dtype=torch.float32, device=dev),
            }
            # Model params
            
            self._model_params = {
                "drag_coefficients": torch.full((batch_size,), self._fixed_params["drag_coefficients"], dtype=torch.float32, device=dev),
                "magnus_factors":    torch.full((batch_size,), self._fixed_params["magnus_factors"],    dtype=torch.float32, device=dev),
            }

    def _compute_forces(
        self, v_w: torch.Tensor, omega_w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v_w:     (..., 3) world linear velocities
        omega_w: (..., 3) world angular velocities
        returns: (forces_w, torques_w) both (..., 3) world-frame
        """
        # Flatten leading dims to a (N, 3) batch for the aero model
        shape = v_w.shape[:-1]
        N = math.prod(shape)

        v_flat     = v_w.reshape(N, 3).to(self.device)
        omega_flat = omega_w.reshape(N, 3).to(self.device)
        # Both drag and Magnus use the ball's velocity relative to the air.
        if self._air_velocity_w is not None:
            v_flat = v_flat - self._air_velocity_w.to(dtype=v_flat.dtype)
        self._ensure_buffers(N)

        params = self._model_params
        if self._dynamic_magnus is not None:
            # Recompute at every call, including when the batch size is unchanged.
            kappa = self._dynamic_magnus(
                torch.linalg.vector_norm(v_flat, dim=-1),
                torch.linalg.vector_norm(omega_flat, dim=-1),
            )
            params = {**params, "magnus_factors": kappa}

        F_flat, T_flat = self.model.calculate_forces_and_torques(
            v_flat, omega_flat, self._ball_props, self.env_props, params
        )
        return F_flat.reshape(*shape, 3), T_flat.reshape(*shape, 3)

    # --- Public apply helpers -------------------------------------------------

    def apply_to_rigid_object(self, ball_asset) -> None:
        """
        ball_asset: isaaclab.assets.RigidObject with one body per env.
        Stages world-frame forces/torques and leaves pushing to sim to caller's scene.write_data_to_sim().
        """
        # (num_envs, 3)
        v_w = ball_asset.data.root_lin_vel_w
        w_w = ball_asset.data.root_ang_vel_w
        F_w, T_w = self._compute_forces(v_w, w_w)  # (num_envs, 3)
        
        self.last_forces_w = F_w.detach()
        self.last_torques_w = T_w.detach()

        # Expand to (num_envs, num_bodies=1, 3) and stage in GLOBAL frame
        ball_asset.set_external_force_and_torque(
            forces=F_w.unsqueeze(1), torques=T_w.unsqueeze(1), is_global=True
        )
