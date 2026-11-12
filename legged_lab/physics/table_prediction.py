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
    restitution: float = 0.98,
    restitution_speed_slope: float = 0.02,
    friction: float = 0.25,
    tangential_restitution: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rebound (linear, angular) velocities in the world frame.

    Velocities have shape (..., 3), in m/s and rad/s. time_to_impact has
    shape (...) in seconds and must be nonnegative. drag_k = rho*A*Cd/(2*m)
    is nonnegative in 1/m, gravity is positive in m/s^2, and radius is in m.
    If a numerical flight predictor already supplies the incident velocity,
    pass time_to_impact=0 to apply only the collision, with no second flight.

    Match the analytical predictor's per-axis horizontal drag and ballistic
    vertical motion. The horizontal solution v(t) = v0/(1+k*abs(v0)*t)
    supplies the incident velocity to the thin-shell impulse model, which
    always updates both linear and angular velocity. Default friction 0.25
    is the nominal average of ball 0.10 and table 0.40 material friction.
    Set friction=0 for a frictionless impulse; None no longer selects the
    removed fixed-horizontal-multiplier model. Separating/grazing incident
    states are left unchanged by the collision solver.

    Normal restitution is evaluated from the INCIDENT normal approach speed:
    e_n = clamp(restitution - restitution_speed_slope*u_n, 0, 1).
    Defaults 0.98 and 0.02 s/m use the Conti et al. v2 base contact model;
    see docs/external-data-sources.md for scope and calibration limitations.
    Set restitution_speed_slope=0 with restitution=<constant> for old fits.

    This is a batched closed-form approximation, without time-stepping or a
    second live contact impulse. Vector drag coupling and Magnus are omitted.
    """
    if friction is None:
        raise ValueError("friction=None is no longer supported; use a nonnegative coefficient (default 0.25)")
    horizontal = linear_velocity[..., :2]
    horizontal_impact = horizontal / (
        1 + drag_k * horizontal.abs() * time_to_impact.unsqueeze(-1)
    )
    vertical_impact = linear_velocity[..., 2] - gravity * time_to_impact

    incident = torch.cat((horizontal_impact, vertical_impact.unsqueeze(-1)), dim=-1)
    return resolve_ball_impact(
        incident,
        angular_velocity,
        incident.new_tensor((0.0, 0.0, 1.0)),
        radius=radius,
        restitution=restitution,
        restitution_speed_slope=restitution_speed_slope,
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
