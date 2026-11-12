# SPDX-License-Identifier: MIT

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from legged_lab.physics.aerodynamics import AeroForceField
from legged_lab.physics.flight_prediction import predict_flight_apex, predict_table_contact
from legged_lab.physics.table_prediction import predict_table_rebound_state


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


class FlightPredictionTest(unittest.TestCase):
    def field(self, **kwargs):
        return AeroForceField(**(dict(device="cpu", radius_m=0.02, air_density=1.225,
                                     drag_coeff=0.4378, magnus_model="conti2026") | kwargs))

    def test_vacuum_apex_and_nonascending_samples(self):
        p = tensor([[[0, 0, 0.78], [1, 2, 3], [1, 2, 3]]])
        v = tensor([[[4, -2, 3], [4, -2, 0], [4, -2, -1]]])
        w = torch.zeros_like(v)
        out, times, valid = predict_flight_apex(
            p, v, w, aero=self.field(drag_coeff=0, magnus_model="constant"), mass_kg=0.0027)
        time = 3 / 9.81
        expected = p.clone()
        expected[0, 0] += tensor([4*time, -2*time, 3**2/(2*9.81)])
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(times, tensor([[time, 0, 0]]))
        self.assertTrue(valid.all())
        torch.testing.assert_close(p, tensor([[[0, 0, 0.78], [1, 2, 3], [1, 2, 3]]]))

    def test_spin_sign_changes_ascent_and_sideways_deflection(self):
        v = tensor([[5, 0, 3]] * 5)
        w = tensor([[0, 100, 0], [0, 0, 0], [0, -100, 0], [0, 0, 100], [0, 0, -100]])
        out, times, valid = predict_flight_apex(
            torch.zeros_like(v), v, w, aero=self.field(), mass_kg=0.0027)
        self.assertTrue(valid.all())
        self.assertLess(times[0], times[1])
        self.assertLess(times[1], times[2])
        self.assertLess(out[0, 2], out[1, 2])
        self.assertLess(out[1, 2], out[2, 2])
        self.assertGreater(out[3, 1], 0)
        torch.testing.assert_close(out[3, 1], -out[4, 1])

    def test_midpoint_matches_independent_fine_rk4(self):
        v = tensor([[5, 0, 3], [-5, 1, 4], [6, -2, 2]])
        w = tensor([[0, 100, 0], [0, 100, 50], [0, 0, -200]])
        p = torch.zeros_like(v)
        field = self.field()
        out, times, valid = predict_flight_apex(p, v, w, aero=field, mass_kg=0.0027)
        reference_p, reference_v = p.clone(), v.clone()
        active = torch.ones(3, dtype=torch.bool)
        reference_t = torch.zeros(3, dtype=v.dtype)
        dt = 0.0005
        def acceleration(velocity):
            return field.compute_forces(velocity, w)[0] / 0.0027 + tensor([0, 0, -9.81])
        for step in range(3000):
            a = acceleration(reference_v)
            b = acceleration(reference_v + 0.5*dt*a)
            c = acceleration(reference_v + 0.5*dt*b)
            d = acceleration(reference_v + dt*c)
            next_v = reference_v + dt*(a+2*b+2*c+d)/6
            next_p = reference_p + dt*reference_v + dt**2*(a+b+c)/6
            crossing = active & (next_v[:, 2] <= 0)
            fraction = reference_v[:, 2] / (reference_v[:, 2] - next_v[:, 2])
            event_p = reference_p + fraction[:, None]*(next_p-reference_p)
            reference_p = torch.where(active[:, None],
                                      torch.where(crossing[:, None], event_p, next_p), reference_p)
            reference_t = torch.where(crossing, (step+fraction)*dt, reference_t)
            reference_v = next_v
            active &= ~crossing
            if not active.any():
                break
        self.assertFalse(active.any())
        self.assertTrue(valid.all())
        torch.testing.assert_close(out, reference_p, rtol=0, atol=0.003)
        torch.testing.assert_close(times, reference_t, rtol=0, atol=0.001)

    def test_horizon_and_invalid_inputs_are_flagged(self):
        p = tensor([[0, 0, 0], [0, 0, float("nan")]])
        v = tensor([[0, 0, 20], [0, 0, 1]])
        _, _, valid = predict_flight_apex(p, v, torch.zeros_like(v), aero=self.field(),
                                        mass_kg=0.0027, max_time_s=0.02)
        self.assertFalse(valid.any())
        with self.assertRaises(ValueError):
            predict_flight_apex(p, v, v, aero=self.field(), mass_kg=0)

    def test_environment_routes_rebound_spin_and_measured_spin_into_ascent(self):
        # Execute the actual prediction block without importing Isaac Lab.
        source = Path(__file__).resolve().parents[1] / "legged_lab/envs/base/tt_env.py"
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "compute_intermediate_values")
        start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "has_bounced")
        end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == "mask_invalid")
        wrapper = ast.parse("def predict(self):\n    pass").body[0]
        wrapper.body = method.body[start:end+1]
        namespace = dict(torch=torch, predict_table_rebound_state=predict_table_rebound_state,
                         predict_flight_apex=predict_flight_apex,
                         predict_table_contact=predict_table_contact)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                     str(source), "exec"), namespace)
        # Execute the actual task defaults, so the integration test catches a
        # regression back to a disabled/legacy contact model in configuration.
        config_tree = ast.parse((source.parent / "tt_config.py").read_text())
        config_class = next(n for n in config_tree.body if isinstance(n, ast.ClassDef)
                            and n.name == "TableCfg")
        config_class.decorator_list = []
        exec(compile(ast.Module(body=[config_class], type_ignores=[]), "TableCfg", "exec"), namespace)
        table = namespace["TableCfg"]()
        self.assertEqual(table.bounce_friction, 0.25)
        self.assertEqual(table.bounce_restitution, 0.98)
        self.assertEqual(table.bounce_restitution_speed_slope, 0.02)
        env = SimpleNamespace(
            aero=self.field(), device="cpu", cfg=SimpleNamespace(table=table),
            ball=SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(
                mass_props=SimpleNamespace(mass=0.0027)))),
            has_touch_own_table_prev=tensor([[0], [1]]),
            ball_pos=tensor([[0, 0, 0.78], [-1, 0, 0.9]]),
            ball_linvel=tensor([[-4, 0, -3], [-4, 0, 3]]),
            ball_angvel=tensor([[0, 0, 0], [0, 100, 50]]),
            has_touch_paddle=torch.zeros(2, dtype=torch.bool))
        env.ball_drag_k = env.aero.drag_acceleration_coefficient(0.0027)
        namespace["predict"](env)
        bounced_v, bounced_w = predict_table_rebound_state(
            env.ball_linvel[:1], env.ball_angvel[:1], tensor([0]),
            drag_k=env.ball_drag_k, gravity=9.81, radius=0.02, friction=0.25)
        self.assertLess(bounced_w[0, 1], 0)  # Friction generates topspin.
        expected, expected_t, _ = predict_flight_apex(
            env.ball_pos, torch.cat((bounced_v, env.ball_linvel[1:])),
            torch.cat((bounced_w, env.ball_angvel[1:])), aero=env.aero, mass_kg=0.0027)
        expected[:, 0].clamp_(max=-1.6)  # Existing task reachability clamp.
        torch.testing.assert_close(env.ball_future_pose, expected)
        torch.testing.assert_close(env.t_before[:1], expected_t[:1])
        torch.testing.assert_close(env.t_after[1:], expected_t[1:])
        wrong, _, _ = predict_flight_apex(env.ball_pos[:1], bounced_v,
                                         env.ball_angvel[:1], aero=env.aero, mass_kg=0.0027)
        self.assertGreater(abs(expected[0, 2] - wrong[0, 2]), 0.01)
        self.assertFalse(env.mask_invalid.any())

        # Incoming spin is modified by the same impulse before ascent.
        env.ball_angvel[0, 1] = 70
        namespace["predict"](env)
        contact_v, contact_w = predict_table_rebound_state(
            env.ball_linvel[:1], env.ball_angvel[:1], tensor([0]),
            drag_k=env.ball_drag_k, gravity=9.81, radius=0.02)
        self.assertFalse(torch.equal(contact_w, env.ball_angvel[:1]))
        expected, _, _ = predict_flight_apex(env.ball_pos[:1], contact_v, contact_w,
                                            aero=env.aero, mass_kg=0.0027)
        torch.testing.assert_close(env.pos_pred_before[:1, 1:], expected[:, 1:])

        # The Magnus-disabled path keeps its existing closed-form apex.
        env.ball_pos[0, 0] = -1.1  # Keep the resulting X beyond the task clamp.
        env.aero = self.field(magnus_model="constant", magnus_factor=0)
        namespace["predict_flight_apex"] = Mock(side_effect=AssertionError("unexpected integration"))
        namespace["predict"](env)
        k = env.ball_drag_k
        expected_height = 0.78 + 0.5/k * torch.log1p(tensor(k/9.81 * 2.76**2))
        torch.testing.assert_close(env.pos_pred_before[0, 2], expected_height)
        expected_time = env.t_before[0]
        expected_x = env.ball_pos[0, 0] + torch.sign(contact_v[0, 0])/k * torch.log1p(
            k * contact_v[0, 0].abs() * expected_time)
        self.assertLess(expected_x, -1.6)
        torch.testing.assert_close(env.pos_pred_before[0, 0], expected_x.clamp(max=-1.6))
        namespace["predict_flight_apex"].assert_not_called()
