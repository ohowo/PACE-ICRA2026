# SPDX-License-Identifier: MIT

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from legged_lab.physics.aerodynamics import AeroForceField
from legged_lab.physics.flight_prediction import predict_flight_apex, predict_table_contact
from legged_lab.physics.table_prediction import predict_table_rebound_state


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


class TableContactTest(unittest.TestCase):
    def field(self, **kwargs):
        return AeroForceField(**(dict(device="cpu", radius_m=0.02, air_density=1.225,
                                     drag_coeff=0.4378, magnus_model="conti2026") | kwargs))

    def contact(self, p, v, w, **kwargs):
        return predict_table_contact(p, v, w, **(dict(table_height=0.78, mass_kg=0.0027,
                                                     aero=self.field()) | kwargs))

    def test_vacuum_contact_time_position_velocity_and_boundary_states(self):
        # Includes a return to the plane after launching upwards from it.
        p = tensor([[[0, 0, 1.03], [0, 0, 0.78], [0, 0, 0.78], [0, 0, 0.78]]])
        v = tensor([[[4, -2, 1.75], [4, 0, 2], [4, 0, -2], [0, 0, 0]]])
        w = torch.zeros_like(v)
        p0, v0 = p.clone(), v.clone()
        out, incident, time, valid = self.contact(
            p, v, w, aero=self.field(drag_coeff=0, magnus_model="constant"))
        expected_t = (v[..., 2] + torch.sqrt(v[..., 2]**2 + 2*9.81*(p[..., 2]-0.78))) / 9.81
        expected_p = p + expected_t[..., None]*v
        expected_p[..., 2] = 0.78
        expected_v = v.clone()
        expected_v[..., 2] -= 9.81*expected_t
        torch.testing.assert_close(time, expected_t, rtol=0, atol=1e-12)
        torch.testing.assert_close(out, expected_p, rtol=0, atol=1e-12)
        torch.testing.assert_close(incident, expected_v, rtol=0, atol=1e-12)
        self.assertTrue(valid.all())
        torch.testing.assert_close(p, p0)
        torch.testing.assert_close(v, v0)

    def test_spin_changes_landing_time_position_and_incident_velocity(self):
        p = tensor([[1.35, 0, 1.03]] * 5)
        v = tensor([[-5.75, 0, 1.75]] * 5)
        w = tensor([[0, -100, 0], [0, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, -100]])
        out, incident, time, valid = self.contact(p, v, w)
        self.assertTrue(valid.all())
        self.assertLess(time[0], time[1])  # Topspin lands sooner.
        self.assertLess(time[1], time[2])  # Backspin lands later.
        self.assertGreater(out[0, 0], out[1, 0])
        self.assertGreater(out[1, 0], out[2, 0])
        self.assertLess(out[3, 1], -0.1)
        self.assertLess(incident[3, 1], 0)
        torch.testing.assert_close(out[3, 1], -out[4, 1])
        torch.testing.assert_close(incident[3, 1], -incident[4, 1])
        for actual, expected in zip(self.contact(p.float(), v.float(), w.float()),
                                    (out, incident, time, valid)):
            if actual.dtype == torch.bool:
                torch.testing.assert_close(actual, expected)
            else:
                torch.testing.assert_close(actual.double(), expected, rtol=0, atol=1e-5)

    def test_contact_matches_independent_fine_rk4(self):
        p = tensor([[1.35, 0, 1.03]] * 4)
        v = tensor([[-5.75, 0, 1.75]] * 4)
        w = tensor([[0, 0, 0], [0, -100, 0], [0, 100, 0], [0, 0, 100]])
        field = self.field()
        out, incident, time, valid = self.contact(p, v, w, aero=field)
        rp, rv = p.clone(), v.clone()
        rt = torch.zeros(4, dtype=v.dtype)
        active = torch.ones(4, dtype=torch.bool)
        dt = 0.0005
        def acceleration(velocity):
            return field.compute_forces(velocity, w)[0] / 0.0027 + tensor([0, 0, -9.81])
        for step in range(3000):
            a = acceleration(rv)
            b = acceleration(rv + 0.5*dt*a)
            c = acceleration(rv + 0.5*dt*b)
            d = acceleration(rv + dt*c)
            nv = rv + dt*(a+2*b+2*c+d)/6
            np = rp + dt*rv + dt**2*(a+b+c)/6
            crossing = active & (np[:, 2] <= 0.78)
            fraction = (rp[:, 2]-0.78) / torch.where(crossing, rp[:, 2]-np[:, 2], 1.0)
            rt = torch.where(crossing, (step+fraction)*dt, rt)
            rp = torch.where(active[:, None],
                             torch.where(crossing[:, None], rp+fraction[:, None]*(np-rp), np), rp)
            rv = torch.where(active[:, None],
                             torch.where(crossing[:, None], rv+fraction[:, None]*(nv-rv), nv), rv)
            active &= ~crossing
            if not active.any():
                break
        self.assertFalse(active.any())
        self.assertTrue(valid.all())
        torch.testing.assert_close(out, rp, rtol=0, atol=0.003)
        torch.testing.assert_close(incident, rv, rtol=0, atol=0.005)
        torch.testing.assert_close(time, rt, rtol=0, atol=0.001)

    def test_air_relative_force_and_world_velocity(self):
        p, v, w = tensor([[0, 0, 1]]), tensor([[5, 1, 2]]), tensor([[0, 100, 30]])
        wind = tensor([2, -1, 0])
        out, incident, time, valid = self.contact(p, v, w)
        moved, moving_v, moving_t, moving_valid = self.contact(
            p, v+wind, w, aero=self.field(air_velocity_w=tuple(wind.tolist())))
        torch.testing.assert_close(moving_t, time)
        torch.testing.assert_close(moved, out+time[:, None]*wind)
        torch.testing.assert_close(moving_v, incident+wind)
        torch.testing.assert_close(moving_valid, valid)

    def test_invalid_and_unresolved_contacts_and_float32_batch(self):
        p = tensor([[[0, 0, 0.7], [0, 0, 2], [float("nan"), 0, 1], [0, 0, 0.78]]]).float()
        v = tensor([[[0, 0, -1], [0, 0, 20], [0, 0, 0], [0, 0, -1]]]).float()
        out, incident, time, valid = self.contact(p, v, torch.zeros_like(v), max_time_s=0.02)
        self.assertEqual(valid.tolist(), [[False, False, False, True]])
        self.assertEqual(out.shape, p.shape)
        self.assertEqual(incident.dtype, torch.float32)
        self.assertEqual(time[0, 3], 0)
        with self.assertRaises(ValueError):
            self.contact(p, v, v, table_height=float("nan"))
        with self.assertRaises(ValueError):
            self.contact(p, v, v, step_s=0)

    def test_environment_passes_numerical_contact_to_rebound_once(self):
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
                         predict_flight_apex=predict_flight_apex, predict_table_contact=predict_table_contact)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                     str(source), "exec"), namespace)
        env = SimpleNamespace(
            aero=self.field(), device="cpu",
            cfg=SimpleNamespace(table=SimpleNamespace(surface_height=0.76, bounce_restitution=0.98,
                                bounce_restitution_speed_slope=0.02,
                                bounce_friction=0.25, bounce_tangential_restitution=0)),
            ball=SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(
                mass_props=SimpleNamespace(mass=0.0027)))),
            has_touch_own_table_prev=tensor([[0], [0], [0], [1]]),
            ball_pos=tensor([[1.35, 0, 1.03], [1.35, 0, 1.03], [0, 0, 50], [0, 0, 50]]),
            ball_linvel=tensor([[-5.75, 0, 1.75]]*2 + [[-4, 0, 1]]*2),
            ball_angvel=tensor([[0, 100, 0], [0, 0, 100], [0, 0, 0], [0, 0, 0]]),
            has_touch_paddle=torch.zeros(4, dtype=torch.bool))
        env.ball_drag_k = env.aero.drag_acceleration_coefficient(0.0027)
        namespace["predict"](env)
        p, incident, time, valid = self.contact(env.ball_pos, env.ball_linvel, env.ball_angvel)
        self.assertEqual(env.valid_land.tolist(), [True, True, False, False])
        self.assertEqual(env.mask_invalid.tolist(), [False, False, True, False])
        torch.testing.assert_close(env.predict_land_t[:2], time[:2])
        torch.testing.assert_close(env.predict_x_land[:2], p[:2, 0])
        torch.testing.assert_close(env.predict_y_land[:2], p[:2, 1])
        # The rebound input is already at contact: zero flight time prevents a
        # second drag or gravity adjustment of the incident velocity.
        rebound, spin = predict_table_rebound_state(
            incident[:2], env.ball_angvel[:2], tensor([0, 0]), drag_k=0,
            gravity=9.81, radius=0.02, friction=0.25)
        # Use the Magnus/drag-evolved incident speed, not the upward launch vz.
        approach_speed = -incident[:2, 2]
        torch.testing.assert_close(rebound[:, 2],
                                   (0.98 - 0.02*approach_speed)*approach_speed)
        apex, ascent_time, apex_valid = predict_flight_apex(
            p[:2], rebound, spin, aero=env.aero, mass_kg=0.0027)
        self.assertTrue(apex_valid.all())
        apex[:, 0].clamp_(max=-1.6)
        torch.testing.assert_close(env.pos_pred_before[:2], apex)
        torch.testing.assert_close(env.t_before[:2], time[:2]+ascent_time)
        # Legacy mode retains the old ballistic contact-time formula.
        env.aero = self.field(magnus_model="constant")
        namespace["predict"](env)
        expected_t = (1.75 + (1.75**2 + 2*9.81*(1.03-0.78))**0.5) / 9.81
        torch.testing.assert_close(env.predict_land_t[:2], tensor([expected_t]*2))
