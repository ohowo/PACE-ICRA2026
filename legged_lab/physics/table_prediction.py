# SPDX-License-Identifier: MIT

import torch

from legged_lab.physics.ball_impact import resolve_ball_impact


def predict_table_rebound_state(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    time_to_impact: torch.Tensor,
    *,
    drag_k: float | torch.Tensor,
    gravity: float,
    radius: float,
    restitution: float = 0.9,
    friction: float | None = None,
    tangential_restitution: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rebound (linear, angular) velocities in the world frame.

    Velocities have shape (..., 3), in m/s and rad/s. time_to_impact has
    shape (...) in seconds and must be nonnegative. drag_k = rho*A*Cd/(2*m)
    is nonnegative in 1/m, gravity is positive in m/s^2, and radius is in m.

    Match the analytical predictor's per-axis horizontal drag and ballistic
    vertical motion. The horizontal solution v(t) = v0/(1+k*abs(v0)*t)
    supplies the incident velocity to BOTH rebound modes. With friction=None,
    retain the inherited horizontal speed multiplier of 0.7 and preserve
    incoming spin: that legacy mode has no angular impulse law. Otherwise use
    the spin-aware thin-shell impulse model. The vertical incident velocity
    should be approaching the table for a physical rebound prediction.

    This is a batched closed-form approximation, without time-stepping or a
    second live contact impulse. Vector drag coupling and Magnus are omitted.
    """
    horizontal = linear_velocity[..., :2]
    horizontal_impact = horizontal / (
        1 + drag_k * horizontal.abs() * time_to_impact.unsqueeze(-1)
    )
    vertical_impact = linear_velocity[..., 2] - gravity * time_to_impact

    if friction is None:
        rebound = torch.cat(
            (0.7 * horizontal_impact, (-restitution * vertical_impact).unsqueeze(-1)), dim=-1
        )
        return rebound, angular_velocity.clone()

    incident = torch.cat((horizontal_impact, vertical_impact.unsqueeze(-1)), dim=-1)
    return resolve_ball_impact(
        incident,
        angular_velocity,
        incident.new_tensor((0.0, 0.0, 1.0)),
        radius=radius,
        restitution=restitution,
        friction=friction,
        tangential_restitution=tangential_restitution,
    )


def predict_table_rebound(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    time_to_impact: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Compatibility API returning only linear velocity; new callers use state."""
    return predict_table_rebound_state(
        linear_velocity, angular_velocity, time_to_impact, **kwargs
    )[0]
