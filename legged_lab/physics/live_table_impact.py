# SPDX-License-Identifier: MIT

"""Replace a confirmed flat tabletop response, never add a second impulse."""

import math

import torch

from legged_lab.physics.ball_impact import resolve_ball_impact


def replace_table_response(
    free_velocity, incoming_spin, native_velocity, native_spin,
    before_position, after_position, table_force, total_force, table_is_stationary,
    *, dt, mass, radius, surface_height, x_bounds, y_bounds,
    restitution, restitution_speed_slope, friction, tangential_restitution,
    bounce_threshold,
):
    """Return (linear velocity, spin, replaced mask), all batched on device.

    Positions are table-root-local in the existing axis-aligned PACE scene.
    Forces are sensor NORMAL forces on the ball (N), averaged over this same
    physics step. For a verified upward tabletop impulse, reconstruct the
    normal incident velocity as v_native_z - F_table_z*dt/m. This removes the
    native normal impulse, including its old restitution, before computing
    the replacement. Tangential incident velocity includes this step's applied
    flight-force kick; incoming spin is sampled before the native collision.

    Preserve native positions and all unsupported responses: no contact,
    edges, net/sides, multiple-body contact, moving/tilted table, and low-speed
    settling below the engine's bounce threshold. This is a velocity-level
    substep approximation, not continuous-time contact integration.
    """
    if not all(math.isfinite(x) and x > 0 for x in (dt, mass, radius)):
        raise ValueError("dt, mass and radius must be positive and finite")
    if not math.isfinite(bounce_threshold) or bounce_threshold < 0:
        raise ValueError("bounce_threshold must be nonnegative and finite")

    # Only an upward normal from the table, with no other body's normal force.
    # Tolerances are numerical guards, not measured material parameters.
    tolerance = 1e-6 + 1e-4 * table_force[..., 2].abs()
    upward = (table_force[..., 2] > 1e-6) & (
        torch.linalg.vector_norm(table_force[..., :2], dim=-1) <= tolerance)
    only_table = torch.linalg.vector_norm(total_force-table_force, dim=-1) <= tolerance
    incident = torch.cat((free_velocity[..., :2],
                          (native_velocity[..., 2] - table_force[..., 2]*dt/mass)[..., None]), dim=-1)
    approach = -incident[..., 2]
    height = surface_height + radius
    # 5 mm accommodates native contact/rest offsets and small position error.
    # Include the substep travel distance; do not mistake a net-top contact
    # for a tabletop contact merely because it belongs to the same table body.
    reach = approach.clamp(min=0)*dt + 0.005
    near_top = ((before_position[..., 2] >= height-0.005)
                & (before_position[..., 2] <= height+reach)
                & (after_position[..., 2] >= height-0.005)
                & (after_position[..., 2] <= height+reach)
                & (torch.minimum(before_position[..., 2], after_position[..., 2]) <= height+radius))
    interior = torch.ones_like(upward)
    for axis, bounds in enumerate((x_bounds, y_bounds)):
        interior &= ((before_position[..., axis] > bounds[0]+radius)
                     & (before_position[..., axis] < bounds[1]-radius)
                     & (after_position[..., axis] > bounds[0]+radius)
                     & (after_position[..., axis] < bounds[1]-radius))
    finite = torch.isfinite(torch.cat((incident, incoming_spin, native_velocity,
                                      native_spin, before_position, after_position,
                                      table_force, total_force), dim=-1)).all(-1)
    replace = (upward & only_table & near_top & interior & finite
               & table_is_stationary & (approach > bounce_threshold))
    rebound, spin = resolve_ball_impact(
        incident, incoming_spin, incident.new_tensor((0, 0, 1)), radius=radius,
        restitution=restitution, restitution_speed_slope=restitution_speed_slope,
        friction=friction, tangential_restitution=tangential_restitution)
    return (torch.where(replace[..., None], rebound, native_velocity),
            torch.where(replace[..., None], spin, native_spin), replace)


class LiveTableImpact:
    """Isaac asset/sensor adapter with no Isaac import, also testable on CPU."""

    def __init__(self, ball, table, sensor, aero, config, dt, bounce_threshold, gravity):
        self.ball, self.table, self.sensor, self.aero = ball, table, sensor, aero
        self.config, self.dt, self.bounce_threshold = config, dt, bounce_threshold
        self.gravity = gravity
        self.mass = ball.cfg.spawn.mass_props.mass
        self.radius = ball.cfg.spawn.radius
        self._prepared = False
        self.last_replaced = None
        data = sensor.data
        if (data.force_matrix_w is None or data.force_matrix_w.shape[1] != 1
                or data.force_matrix_w.shape[2] == 0):
            raise ValueError("Live table impact requires one ball contact sensor filtered to the table body")

    def prepare(self):
        """Call after staging aero forces and before scene.write_data_to_sim."""
        self.before_position = self.ball.data.root_pos_w.clone()
        self.incoming_spin = self.ball.data.root_ang_vel_w.clone()
        # Reuse the force already computed for this substep; never recompute
        # or apply a second aerodynamic force here. Normal speed is recovered
        # from the native impulse in replace_table_response instead.
        self.free_velocity = self.ball.data.root_lin_vel_w + self.dt * (
            self.aero.last_forces_w / self.mass
            + self.ball.data.root_lin_vel_w.new_tensor(self.gravity))
        self._prepared = True

    def apply(self):
        """Call after scene.update and before perception/rewards read ball state."""
        if not self._prepared:
            raise RuntimeError("prepare must precede each live table response")
        self._prepared = False
        contact = self.sensor.data  # Lazy sensor update for the just-completed substep.
        table_force = contact.force_matrix_w[:, 0].sum(dim=1)
        total_force = contact.net_forces_w[:, 0]
        data = self.table.data
        # The shared prediction law assumes a stationary, axis-aligned table.
        stationary = ((torch.linalg.vector_norm(data.root_lin_vel_w, dim=-1) < 1e-3)
                      & (torch.linalg.vector_norm(data.root_ang_vel_w, dim=-1) < 1e-3)
                      & (torch.linalg.vector_norm(data.root_quat_w[:, 1:], dim=-1) < 1e-4))
        origin = data.root_pos_w
        cfg = self.config
        velocity, spin, self.last_replaced = replace_table_response(
            self.free_velocity, self.incoming_spin,
            self.ball.data.root_lin_vel_w, self.ball.data.root_ang_vel_w,
            self.before_position-origin, self.ball.data.root_pos_w-origin,
            table_force, total_force, stationary,
            dt=self.dt, mass=self.mass, radius=self.radius,
            surface_height=cfg.surface_height,
            x_bounds=(cfg.table_own_contact_x[0], cfg.table_opponent_contact_x[1]),
            y_bounds=cfg.table_own_contact_y,
            restitution=cfg.bounce_restitution,
            restitution_speed_slope=cfg.bounce_restitution_speed_slope,
            friction=cfg.bounce_friction, tangential_restitution=cfg.bounce_tangential_restitution,
            bounce_threshold=self.bounce_threshold)
        # One batched replacement, not a delta added to already bounced state.
        # Non-selected rows contain the exact native velocities. No position
        # teleport or extra simulation step; avoid GPU nonzero/host branching.
        self.ball.write_root_velocity_to_sim(torch.cat((velocity, spin), dim=-1))
