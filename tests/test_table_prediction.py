# SPDX-License-Identifier: MIT

import math
import unittest

import torch

from legged_lab.physics.ball_impact import resolve_ball_impact
from legged_lab.physics.table_prediction import predict_table_rebound, predict_table_rebound_state


def vector(values):
    return torch.tensor(values, dtype=torch.float64, device="cpu")


class TablePredictionTest(unittest.TestCase):
    def test_rebound_state_keeps_contact_angular_impulse(self):
        # With no initial spin, friction must generate rotation at the table.
        v, w = vector([[4, 0, -3], [-4, 0, -3]]), vector([[0, 0, 0], [0, 50, 10]])
        v0, w0 = v.clone(), w.clone()
        kwargs = dict(radius=0.02, restitution=0.9, restitution_speed_slope=0, friction=0.25,
                      tangential_restitution=0.2)
        actual_v, actual_w = predict_table_rebound_state(
            v, w, vector([0, 0]), drag_k=0.125, gravity=9.81, **kwargs)
        expected_v, expected_w = resolve_ball_impact(v, w, vector([0, 0, 1]), **kwargs)
        torch.testing.assert_close(actual_v, expected_v)
        torch.testing.assert_close(actual_w, expected_w)
        self.assertGreater(actual_w[0, 1].item(), 0)
        # Independently check angular momentum about the contact point:
        # delta omega = (-r*n) x delta v_t / ((2/3)*r^2).
        arm = vector([[0, 0, -0.02]]).expand_as(v)
        torch.testing.assert_close(actual_w - w,
                                   torch.linalg.cross(arm, actual_v - v) / ((2/3)*0.02**2))
        torch.testing.assert_close(v, v0)
        torch.testing.assert_close(w, w0)

    def test_linear_api_matches_state_and_outputs_do_not_alias_inputs(self):
        v = vector([[[4, 0, -3], [-4, 0, -3]]])
        w = vector([[[0, 50, 0], [0, -50, 20]]])
        time = vector([[0.1, 0.2]])
        kwargs = dict(drag_k=0.125, gravity=9.81, radius=0.02)
        out_v, out_w = predict_table_rebound_state(v, w, time, **kwargs)
        torch.testing.assert_close(out_v, predict_table_rebound(v, w, time, **kwargs))
        self.assertFalse(torch.equal(out_w, w))
        out_w.zero_()
        self.assertEqual(w[0, 0, 1], 50)  # No output alias into caller state.

    def rebound(self, velocity, time, **kwargs):
        return predict_table_rebound(
            velocity, torch.zeros_like(velocity), time,
            **({"drag_k": 0.1248038406, "gravity": 9.81, "radius": 0.02} | kwargs),
        )

    def test_default_rebound_uses_contact_velocity(self):
        # Current nominal serve, using the analytically propagated incident
        # speed and Coulomb-limited friction rather than a fixed multiplier.
        time = (1.75 + math.sqrt(1.75**2 + 2 * 9.81 * (1.03 - 0.78))) / 9.81
        rebound = self.rebound(vector([[-5.75, 0, 1.75]]), vector([time]))
        incident_x = -5.75 / (1 + 0.1248038406 * 5.75 * time)
        normal_speed = math.sqrt(1.75**2 + 2 * 9.81 * (1.03 - 0.78))
        e = 0.98 - 0.02 * normal_speed
        self.assertGreater(0.4 * abs(incident_x), 0.25 * (1+e) * normal_speed)
        self.assertAlmostEqual(rebound[0, 0].item(), incident_x + 0.25 * (1+e) * normal_speed, places=10)
        # Ballistic normal speed from the vertical energy balance.
        expected_vz = e * math.sqrt(1.75**2 + 2 * 9.81 * (1.03 - 0.78))
        self.assertAlmostEqual(rebound[0, 2].item(), expected_vz, places=10)

    def test_matches_integrated_horizontal_drag(self):
        # Independently integrate dv/dt = -k*abs(v)*v on CPU with RK4.
        # Signed and zero components exercise both axes without a sign branch.
        initial = vector([[6, -2, -1], [-5, 3, -2], [0, 0, -3]])
        incident_xy = initial[:, :2].clone()
        k, time, steps = 0.125, 0.4, 400
        dt = time / steps
        def derivative(v):
            return -k * v.abs() * v
        for _ in range(steps):
            a = derivative(incident_xy)
            b = derivative(incident_xy + dt * a / 2)
            c = derivative(incident_xy + dt * b / 2)
            d = derivative(incident_xy + dt * c)
            incident_xy += dt * (a + 2*b + 2*c + d) / 6
        rebound = self.rebound(initial, vector([time] * 3), drag_k=k, friction=0)
        torch.testing.assert_close(rebound[:, :2], incident_xy, rtol=1e-10, atol=1e-10)

    def test_zero_drag_or_zero_time_preserves_limiting_solution(self):
        initial = vector([[4, -2, -3], [0, 0, -1]])
        at_contact = self.rebound(initial, vector([0, 0]), friction=0)
        torch.testing.assert_close(at_contact, vector([[4, -2, 2.76], [0, 0, 0.96]]))
        no_drag = self.rebound(initial, vector([0.2, 0.4]), drag_k=0.0, restitution=0.8, restitution_speed_slope=0, friction=0)
        torch.testing.assert_close(no_drag, vector([[4, -2, 3.9696], [0, 0, 3.9392]]))

    def test_spin_mode_receives_the_same_incident_velocity(self):
        # In one second at k=0.25, |vx| 4 -> 2 and |vy| 2 -> 4/3.
        velocity = vector([[4, -2, -1], [-4, 2, -1]])
        spin = vector([[0, 50, 0], [0, -50, 0]])
        incident = vector([[2, -4/3, -10.81], [-2, 4/3, -10.81]])
        expected, _ = resolve_ball_impact(
            incident, spin, vector([0, 0, 1]), radius=0.02,
            restitution=0.85, restitution_speed_slope=0, friction=0.25, tangential_restitution=0.2,
        )
        actual = predict_table_rebound(
            velocity, spin, vector([1, 1]), drag_k=0.25, gravity=9.81,
            radius=0.02, restitution=0.85, restitution_speed_slope=0, friction=0.25, tangential_restitution=0.2,
        )
        torch.testing.assert_close(actual, expected)

    def test_batched_inputs_are_unchanged_and_dissipate_at_contact(self):
        initial = vector([[[4, -2, -3], [0, 0, -1]], [[-2, 4, -1], [0, 3, -2]]])
        before = initial.clone()
        out = self.rebound(initial, torch.zeros((2, 2), dtype=torch.float64))
        self.assertEqual(out.shape, initial.shape)
        torch.testing.assert_close(initial, before)
        self.assertTrue(torch.all(out.square().sum(-1) <= initial.square().sum(-1)))

    def test_default_impulse_generates_spin_and_respects_friction_limit(self):
        # e=0.92; qn=1.92*3=5.76 m/s; |qt|=min(1.6, 0.25*5.76)=1.44 m/s.
        velocity, spin = predict_table_rebound_state(
            vector([[4, 0, -3]]), vector([[0, 0, 0]]), vector([0]),
            drag_k=0, gravity=9.81, radius=0.02)
        torch.testing.assert_close(velocity, vector([[2.56, 0, 2.76]]))
        torch.testing.assert_close(spin, vector([[0, 108, 0]]))

    def test_default_impulse_handles_rolling_and_spin_to_translation(self):
        v = vector([[2, 0, -3], [2, 0, -3], [2, 0, -3]])
        w = vector([[0, 100, 30], [0, 300, 30], [0, -100, 30]])
        out_v, out_w = predict_table_rebound_state(
            v, w, vector([0, 0, 0]), drag_k=0, gravity=9.81, radius=0.02)
        torch.testing.assert_close(out_v[:, 0], vector([2, 3.44, 0.56]))
        torch.testing.assert_close(out_w[:, 1], vector([100, 192, 8]))
        torch.testing.assert_close(out_w[:, 2], w[:, 2])
        # Check total kinetic energy, including rotation, when spin increases vx.
        inertia_per_mass = (2/3)*0.02**2
        before = v.square().sum(-1) + inertia_per_mass*w.square().sum(-1)
        after = out_v.square().sum(-1) + inertia_per_mass*out_w.square().sum(-1)
        self.assertTrue(torch.all(after <= before))

    def test_separating_and_grazing_states_do_not_rebound_again(self):
        v, w = vector([[3, 0, 1], [3, 0, 0]]), vector([[0, -100, 0]]*2)
        out_v, out_w = predict_table_rebound_state(
            v, w, vector([0, 0]), drag_k=0, gravity=9.81, radius=0.02)
        torch.testing.assert_close(out_v, v)
        torch.testing.assert_close(out_w, w)

    def test_removed_none_mode_is_rejected_with_migration_message(self):
        with self.assertRaisesRegex(ValueError, "friction=None.*nonnegative"):
            self.rebound(vector([[4, 0, -3]]), vector([0]), friction=None)

    def test_repeated_prediction_recomputes_coefficient_from_new_contact_speed(self):
        velocity = vector([[4, 0, -2.5], [4, 0, -2.6], [4, 0, -2.7]])
        for speeds, coefficients in (([2.5, 2.6, 2.7], [0.930, 0.928, 0.926]),
                                     ([3, 4, 6], [0.92, 0.90, 0.86])):
            velocity[:, 2] = -vector(speeds)
            out = self.rebound(velocity, vector([0, 0, 0]), friction=0)
            torch.testing.assert_close(out[:, 2] / vector(speeds), vector(coefficients))
