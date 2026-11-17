# SPDX-License-Identifier: MIT
"""Timed physical ball states, independent of controller reachability targets."""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from legged_lab.physics.aerodynamics import AeroForceField
from legged_lab.physics.ball_impact import resolve_ball_impact


@dataclass(frozen=True)
class TrajectoryContacts:
    """Contact-event samples in chronological slots (..., max_bounces + 1).

    Vector fields add a final dimension of 3. Times are relative to the input
    state. Each valid slot retains both sides of the impulse; ``bounced`` is
    false for the terminating contact, whose after-state equals its incident
    state (no further response is modeled). Unused numeric slots contain NaN.
    """

    time: torch.Tensor
    position: torch.Tensor
    velocity_before: torch.Tensor
    velocity_after: torch.Tensor
    spin_before: torch.Tensor
    spin_after: torch.Tensor
    valid: torch.Tensor
    bounced: torch.Tensor


@dataclass(frozen=True)
class TimedTrajectory:
    """Regular time grid plus explicit contact events, in the input frame.

    time has shape (S,); position/velocity/spin have shape (..., S, 3).
    valid and bounce_count have shape (..., S). The first sample is exactly
    the supplied state, even for an impact at t=0. Later samples at a rebound
    are post-impulse. Invalid numeric states are NaN and counts are -1;
    samples after the terminating contact must not be used. An off-grid
    terminal state is available in contacts rather than assigned a false
    grid time. Input tensors are never mutated. No workspace clamp applies.
    """

    time: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    spin: torch.Tensor
    valid: torch.Tensor
    bounce_count: torch.Tensor
    contacts: TrajectoryContacts


def _advance(position, velocity, spin, duration, aero, mass, gravity, height):
    """Midpoint step and dense descending plane event (batched durations)."""
    dt = duration.unsqueeze(-1)
    a0 = aero.compute_forces(velocity, spin)[0] / mass + gravity
    mid_v = velocity + 0.5 * dt * a0
    amid = aero.compute_forces(mid_v, spin)[0] / mass + gravity
    next_p, next_v = position + dt * mid_v, velocity + dt * amid
    if height is None:
        return next_p, next_v, None
    gap = (position[..., 2] - height).clamp(min=0)
    discriminant = (velocity[..., 2].square() - 2 * a0[..., 2] * gap).clamp(min=0)
    denominator = -velocity[..., 2] + torch.sqrt(discriminant)
    regular = denominator > 0
    safe_a = torch.where(a0[..., 2] != 0, a0[..., 2], 1.0)
    root = torch.where(regular, 2 * gap / torch.where(regular, denominator, 1.0),
                       -2 * velocity[..., 2] / safe_a)
    tau = torch.minimum(root.clamp(min=0), duration)
    fraction = tau / torch.where(duration > 0, duration, 1.0)
    event_p = position + tau.unsqueeze(-1) * velocity + 0.5 * tau.unsqueeze(-1).square() * a0
    event_p = torch.cat((event_p[..., :2], torch.full_like(event_p[..., 2:3], height)), dim=-1)
    event_v = velocity + tau.unsqueeze(-1) * (a0 + fraction.unsqueeze(-1) * (amid - a0))
    return next_p, next_v, (tau, event_p, event_v)


@torch.no_grad()
def predict_timed_trajectory(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    *,
    aero: AeroForceField,
    mass_kg: float,
    gravity: float = 9.81,
    step_s: float = 0.02,
    max_time_s: float = 1.5,
    table_height: Optional[float] = None,
    radius: float = 0.02,
    max_bounces: int = 1,
    restitution: float = 0.98,
    restitution_speed_slope: float = 0.02,
    friction: float = 0.25,
    tangential_restitution: float = 0.0,
) -> TimedTrajectory:
    """Predict full flight samples, optionally across ONE future table bounce.

    Units are meters, seconds, m/s and rad/s. Spin is constant in free flight;
    the shared thin-shell impact updates it at contact. The same vector force
    integrator is used with or without Magnus. This is an inference API.

    With table_height=None there are no contact events. Otherwise the height
    is the BALL CENTER contact plane (surface height + radius), not the table
    surface. The plane is infinite, stationary and horizontal in the input
    frame. Finite-table/net geometry and reachability are not evaluated here.
    max_bounces=0 stops at the first contact; 1 continues after that rebound
    and stops at the second contact. A contact producing no upward velocity
    terminates immediately (no low-speed settling model). The impact substep
    is split at the event and its remaining time is integrated with new spin.

    Invalid/nonfinite input or integration states are masked, never converted
    to robot-relative fallback targets. Grid spacing is max_time_s / ceil(
    max_time_s / step_s), so the final grid time is exactly the horizon.
    """
    for value in (mass_kg, gravity, step_s, max_time_s, radius):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("mass, gravity, step, horizon and radius must be positive and finite")
    if table_height is not None and not math.isfinite(table_height):
        raise ValueError("table_height must be finite or None")
    if type(max_bounces) is not int or max_bounces not in (0, 1):
        raise ValueError("max_bounces must be 0 or 1")
    if (position.shape != velocity.shape or position.shape != spin.shape
            or position.ndim == 0 or position.shape[-1] != 3):
        raise ValueError("position, velocity and spin must have matching (..., 3) shapes")
    if (not position.is_floating_point() or velocity.dtype != position.dtype
            or spin.dtype != position.dtype or velocity.device != position.device
            or spin.device != position.device):
        raise ValueError("state tensors must share a floating dtype and device")
    if (not 0 <= restitution <= 1 or not 0 <= tangential_restitution <= 1
            or not math.isfinite(friction) or friction < 0
            or not math.isfinite(restitution_speed_slope) or restitution_speed_slope < 0):
        raise ValueError("invalid restitution, friction or restitution speed slope")

    steps = math.ceil(max_time_s / step_s)
    time = torch.linspace(0, max_time_s, steps + 1, dtype=position.dtype, device=position.device)
    batch = position.shape[:-1]
    finite = torch.isfinite(torch.cat((position, velocity, spin), dim=-1)).all(-1)
    active = finite if table_height is None else finite & (position[..., 2] >= table_height)
    # Finite placeholders are internal only; every invalid public sample is NaN.
    p, v, w = (torch.where(active.unsqueeze(-1), x, 0.0).clone() for x in (position, velocity, spin))
    counts = torch.zeros(batch, dtype=torch.long, device=position.device)
    event_slots = 0 if table_height is None else max_bounces + 1
    event_shape = batch + (event_slots,)
    event_time = position.new_full(event_shape, float("nan"))
    vectors = [position.new_full(event_shape + (3,), float("nan")) for _ in range(5)]
    event_valid = torch.zeros(event_shape, dtype=torch.bool, device=position.device)
    event_bounced = torch.zeros_like(event_valid)
    ps, vs, ws, masks, bounce_counts = [], [], [], [], []

    def record(mask):
        for history, value in ((ps, p), (vs, v), (ws, w)):
            history.append(torch.where(mask.unsqueeze(-1), value, float("nan")))
        masks.append(mask.clone())
        bounce_counts.append(torch.where(mask, counts, -1))

    record(active)
    gravity_vector = position.new_tensor((0.0, 0.0, -gravity))
    normal = position.new_tensor((0.0, 0.0, 1.0))
    for step in range(steps):
        remaining = torch.ones_like(counts, dtype=position.dtype) * (time[step + 1] - time[step])
        consumed = torch.zeros_like(remaining)
        terminal_at_end = torch.zeros_like(active)
        # At most one rebound and one terminal contact can occur in a step.
        for _ in range(1 if table_height is None else max_bounces + 1):
            running = active & (remaining > 0)
            dt = torch.where(running, remaining, 0.0)
            next_p, next_v, event = _advance(p, v, w, dt, aero, mass_kg, gravity_vector, table_height)
            good = torch.isfinite(torch.cat((next_p, next_v), dim=-1)).all(-1)
            active = active & (~running | good)
            if table_height is None:
                p = torch.where((running & good).unsqueeze(-1), next_p, p)
                v = torch.where((running & good).unsqueeze(-1), next_v, v)
                break
            tau, event_p, event_v = event
            good_event = torch.isfinite(torch.cat((event_p, event_v), dim=-1)).all(-1)
            crossing = running & good & (next_p[..., 2] <= table_height)
            active = active & (~crossing | good_event)
            crossing = crossing & good_event
            outgoing, outgoing_spin = resolve_ball_impact(
                event_v, w, normal, radius=radius, restitution=restitution,
                friction=friction, restitution_speed_slope=restitution_speed_slope,
                tangential_restitution=tangential_restitution)
            response_finite = torch.isfinite(torch.cat((outgoing, outgoing_spin), dim=-1)).all(-1)
            bad_response = crossing & (counts < max_bounces) & ~response_finite
            active = active & ~bad_response
            crossing = crossing & ~bad_response
            bounce = crossing & (counts < max_bounces) & (outgoing[..., 2] > 0)
            terminal = crossing & ~bounce
            terminal_at_end |= terminal & (tau == remaining)
            for slot in range(event_slots):
                selected = crossing & (counts == slot)
                event_time[..., slot] = torch.where(selected, time[step] + consumed + tau, event_time[..., slot])
                for target, value in zip(vectors, (event_p, event_v,
                                                  torch.where(bounce.unsqueeze(-1), outgoing, event_v),
                                                  w, torch.where(bounce.unsqueeze(-1), outgoing_spin, w))):
                    target[..., slot, :] = torch.where(selected.unsqueeze(-1), value, target[..., slot, :])
                event_valid[..., slot] |= selected
                event_bounced[..., slot] |= selected & bounce
            advancing = running & active
            p = torch.where(advancing.unsqueeze(-1), torch.where(crossing.unsqueeze(-1), event_p, next_p), p)
            v = torch.where(advancing.unsqueeze(-1),
                            torch.where(bounce.unsqueeze(-1), outgoing,
                                        torch.where(crossing.unsqueeze(-1), event_v, next_v)), v)
            w = torch.where(bounce.unsqueeze(-1), outgoing_spin, w)
            counts = counts + bounce.long()
            active = active & ~terminal
            consumed = consumed + torch.where(crossing, tau, remaining)
            remaining = torch.where(bounce, (remaining - tau).clamp(min=0), 0.0)
        record(active | terminal_at_end)

    contacts = TrajectoryContacts(event_time, *vectors, event_valid, event_bounced)
    return TimedTrajectory(time, torch.stack(ps, dim=-2), torch.stack(vs, dim=-2),
                           torch.stack(ws, dim=-2), torch.stack(masks, dim=-1),
                           torch.stack(bounce_counts, dim=-1), contacts)
