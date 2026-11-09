# SPDX-License-Identifier: MIT

import math

import torch

from legged_lab.physics.aerodynamics import AeroForceField


def _validate_flight_inputs(position, linear_velocity, angular_velocity,
                            mass_kg, gravity, step_s, max_time_s):
    for name, value in (("mass_kg", mass_kg), ("gravity", gravity),
                        ("step_s", step_s), ("max_time_s", max_time_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if position.shape != linear_velocity.shape or position.shape != angular_velocity.shape:
        raise ValueError("position and velocities must have matching shapes")
    if not position.ndim or position.shape[-1] != 3:
        raise ValueError("last dimension must contain three world components")


def predict_table_contact(
    position: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    *,
    table_height: float,
    aero: AeroForceField,
    mass_kg: float,
    gravity: float = 9.81,
    step_s: float = 0.02,
    max_time_s: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return position, incident velocity, time and validity at the table plane.

    SI/world inputs have matching (..., 3) shapes. table_height is the ball
    CENTER height at contact, including radius. This infinite-plane predictor
    does not test table edges or the net. It locates the first descending
    crossing using midpoint integration of the live drag/Magnus force law.
    Spin stays constant under the existing zero aerodynamic torque model.

    At-plane incoming/resting samples return their input at t=0; at-plane
    upward samples are followed to their return. Initially below-plane,
    nonfinite or horizon-exhausted samples are invalid. Callers must check
    validity rather than interpreting the last state as a contact.
    """
    _validate_flight_inputs(position, linear_velocity, angular_velocity,
                            mass_kg, gravity, step_s, max_time_s)
    if not math.isfinite(table_height):
        raise ValueError("table_height must be finite")
    steps = math.ceil(max_time_s / step_s)
    dt = max_time_s / steps
    p, v = position.clone(), linear_velocity.clone()
    # Match the tensor dtype (not a promoted Python scalar) for contact tests.
    height = p.new_tensor(table_height)
    finite = (torch.isfinite(p).all(-1) & torch.isfinite(v).all(-1)
              & torch.isfinite(angular_velocity).all(-1))
    valid = finite & (p[..., 2] == height) & (v[..., 2] <= 0)
    active = finite & (p[..., 2] >= height) & ~valid
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
        crossing = active & finite_step & (next_p[..., 2] <= height)
        # Solve the quadratic midpoint position interpolant for the descending
        # plane crossing. The rational form avoids cancellation near contact.
        gap = (p[..., 2] - height).clamp(min=0)
        discriminant = (v[..., 2].square() - 2 * a0[..., 2] * gap).clamp(min=0)
        denominator = -v[..., 2] + torch.sqrt(discriminant)
        regular = denominator > 0
        safe_a = torch.where(a0[..., 2] != 0, a0[..., 2], 1.0)
        root = torch.where(regular, 2 * gap / torch.where(regular, denominator, 1.0),
                           -2 * v[..., 2] / safe_a)
        event_dt = torch.where(crossing, root.clamp(min=0, max=dt), dt).unsqueeze(-1)
        fraction = event_dt / dt
        event_p = p + event_dt * v + 0.5 * event_dt.square() * a0
        event_p = torch.cat((event_p[..., :2], torch.full_like(event_p[..., 2:3], table_height)), dim=-1)
        # Dense midpoint velocity: endpoints match v and next_v, while the
        # initial derivative is a0. No extra force evaluation at contact.
        event_v = v + event_dt * (a0 + fraction * (a_mid - a0))
        p = torch.where(active.unsqueeze(-1),
                        torch.where(crossing.unsqueeze(-1), event_p, next_p), p)
        v = torch.where(active.unsqueeze(-1),
                        torch.where(crossing.unsqueeze(-1), event_v, next_v), v)
        elapsed = elapsed + torch.where(active, event_dt.squeeze(-1), 0.0)
        valid = valid | crossing
        active = active & finite_step & ~crossing
    return p, v, elapsed, valid


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
    _validate_flight_inputs(position, linear_velocity, angular_velocity,
                            mass_kg, gravity, step_s, max_time_s)

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
