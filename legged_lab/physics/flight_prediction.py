# SPDX-License-Identifier: MIT

import math

import torch

from legged_lab.physics.aerodynamics import AeroForceField


def predict_flight_apex(
    position: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    *,
    aero: AeroForceField,
    mass_kg: float,
    gravity: float = 9.81,
    step_s: float = 0.02,
    max_time_s: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict first ascent apex with batched midpoint integration (SI/world).

    Inputs have matching (..., 3) shapes. Return position, elapsed seconds,
    and a valid mask. Initially descending/stationary samples return their
    starting position and zero time. Unresolved or nonfinite samples are
    invalid; their last state must not be interpreted as an apex.

    Each stage recomputes vector drag and Magnus from velocity and spin using
    the live force law. Spin is held at its supplied value because that law
    has zero aerodynamic torque. No collision or live force is applied here.
    The fixed loop budget avoids per-ball loops and device scalar readbacks.
    """
    for name, value in (("mass_kg", mass_kg), ("gravity", gravity),
                        ("step_s", step_s), ("max_time_s", max_time_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if position.shape != linear_velocity.shape or position.shape != angular_velocity.shape:
        raise ValueError("position and velocities must have matching shapes")
    if position.shape[-1] != 3:
        raise ValueError("last dimension must contain three world components")

    steps = math.ceil(max_time_s / step_s)
    dt = max_time_s / steps
    p, v = position.clone(), linear_velocity.clone()
    finite = (torch.isfinite(p).all(-1) & torch.isfinite(v).all(-1)
              & torch.isfinite(angular_velocity).all(-1))
    valid = finite & (v[..., 2] <= 0)
    active = finite & ~valid
    elapsed = torch.zeros_like(v[..., 2])
    gravity_vector = v.new_tensor((0.0, 0.0, -gravity))

    def acceleration(velocity):
        force, _ = aero.compute_forces(velocity, angular_velocity)
        return force / mass_kg + gravity_vector

    for _ in range(steps):
        a0 = acceleration(v)
        mid_v = v + 0.5 * dt * a0
        a_mid = acceleration(mid_v)
        next_v = v + dt * a_mid
        next_p = p + dt * mid_v
        finite_step = torch.isfinite(next_v).all(-1) & torch.isfinite(next_p).all(-1)
        crossing = active & finite_step & (next_v[..., 2] <= 0)
        # Locate vz=0 inside the crossing step, avoiding a full-step overshoot.
        denominator = torch.where(crossing, v[..., 2] - next_v[..., 2], 1.0)
        fraction = torch.where(crossing, v[..., 2] / denominator, 1.0)
        event_dt = (dt * fraction).unsqueeze(-1)
        event_p = p + event_dt * v + 0.5 * event_dt.square() * a0
        p = torch.where(active.unsqueeze(-1),
                        torch.where(crossing.unsqueeze(-1), event_p, next_p), p)
        v = torch.where(active.unsqueeze(-1), next_v, v)
        elapsed = elapsed + torch.where(active, dt * fraction, 0.0)
        valid = valid | crossing
        active = active & finite_step & ~crossing
    return p, elapsed, valid
