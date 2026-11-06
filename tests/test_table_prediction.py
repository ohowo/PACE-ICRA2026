# SPDX-License-Identifier: MIT

import math
import unittest

import torch

from legged_lab.physics.ball_impact import resolve_ball_impact
from legged_lab.physics.table_prediction import predict_table_rebound


def vector(values):
    return torch.tensor(values, dtype=torch.float64, device="cpu")


class TablePredictionTest(unittest.TestCase):
    def rebound(self, velocity, time, **kwargs):
        return predict_table_rebound(
            velocity, torch.zeros_like(velocity), time,
            **({"drag_k": 0.1248038406, "gravity": 9.81, "radius": 0.02} | kwargs),
        )

    def test_default_rebound_uses_contact_velocity(self):
        # Current nominal serve. Regression for using 0.7 * launch vx (-4.025).
        time = (1.75 + math.sqrt(1.75**2 + 2 * 9.81 * (1.03 - 0.78))) / 9.81
        rebound = self.rebound(vector([[-5.75, 0, 1.75]]), vector([time]))
        self.assertAlmostEqual(rebound[0, 0].item(), -3.0161089900, places=8)
        # Ballistic normal speed from the vertical energy balance.
        expected_vz = 0.9 * math.sqrt(1.75**2 + 2 * 9.81 * (1.03 - 0.78))
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
        rebound = self.rebound(initial, vector([time] * 3), drag_k=k)
        torch.testing.assert_close(rebound[:, :2], 0.7 * incident_xy, rtol=1e-10, atol=1e-10)

    def test_zero_drag_or_zero_time_preserves_limiting_solution(self):
        initial = vector([[4, -2, -3], [0, 0, -1]])
        at_contact = self.rebound(initial, vector([0, 0]))
        torch.testing.assert_close(at_contact, vector([[2.8, -1.4, 2.7], [0, 0, 0.9]]))
        no_drag = self.rebound(initial, vector([0.2, 0.4]), drag_k=0.0, restitution=0.8)
        torch.testing.assert_close(no_drag, vector([[2.8, -1.4, 3.9696], [0, 0, 3.9392]]))

    def test_spin_mode_receives_the_same_incident_velocity(self):
        # In one second at k=0.25, |vx| 4 -> 2 and |vy| 2 -> 4/3.
        velocity = vector([[4, -2, -1], [-4, 2, -1]])
        spin = vector([[0, 50, 0], [0, -50, 0]])
        incident = vector([[2, -4/3, -10.81], [-2, 4/3, -10.81]])
        expected, _ = resolve_ball_impact(
            incident, spin, vector([0, 0, 1]), radius=0.02,
            restitution=0.85, friction=0.25, tangential_restitution=0.2,
        )
        actual = predict_table_rebound(
            velocity, spin, vector([1, 1]), drag_k=0.25, gravity=9.81,
            radius=0.02, restitution=0.85, friction=0.25, tangential_restitution=0.2,
        )
        torch.testing.assert_close(actual, expected)

    def test_batched_inputs_are_unchanged_and_dissipate_at_contact(self):
        initial = vector([[[4, -2, -3], [0, 0, -1]], [[-2, 4, -1], [0, 3, -2]]])
        before = initial.clone()
        out = self.rebound(initial, torch.zeros((2, 2), dtype=torch.float64))
        self.assertEqual(out.shape, initial.shape)
        torch.testing.assert_close(initial, before)
        self.assertTrue(torch.all(out.square().sum(-1) <= initial.square().sum(-1)))
