# SPDX-License-Identifier: MIT

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from legged_lab.physics.aerodynamics import AeroForceField
from legged_lab.physics.trajectory_prediction import predict_timed_trajectory


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


class TimedTrajectoryTest(unittest.TestCase):
    def field(self, **kwargs):
        return AeroForceField(**(dict(device="cpu", radius_m=0.02, air_density=1.225,
                                     drag_coeff=0.4378, magnus_model="conti2026") | kwargs))

    def trajectory(self, p, v, w=None, **kwargs):
        return predict_timed_trajectory(p, v, torch.zeros_like(v) if w is None else w,
                                        **(dict(aero=self.field(), mass_kg=0.0027) | kwargs))

    def vacuum(self):
        return self.field(drag_coeff=0, magnus_model="constant")

    def test_free_flight_grid_matches_exact_vacuum_and_preserves_inputs(self):
        p = tensor([[[3, 1, 2], [-5, 2, 1]]])
        v = tensor([[[4, -1, 3], [2, 1, -2]]])
        w = tensor([[[0, 200, 0], [20, 0, 0]]])
        originals = [x.clone() for x in (p, v, w)]
        result = self.trajectory(p, v, w, aero=self.vacuum(), max_time_s=0.103, step_s=0.02)
        t = result.time[:, None]
        g = tensor([0, 0, -9.81])
        expected_p = p[..., None, :] + t*v[..., None, :] + 0.5*t*t*g
        expected_v = v[..., None, :] + t*g
        torch.testing.assert_close(result.position, expected_p)
        torch.testing.assert_close(result.velocity, expected_v)
        torch.testing.assert_close(result.spin, w[..., None, :].expand_as(result.spin))
        self.assertTrue(result.valid.all())
        self.assertEqual(result.time[0], 0)
        self.assertEqual(result.time[-1], 0.103)
        self.assertTrue(torch.all(result.time.diff() > 0))
        self.assertLessEqual(result.time.diff().max(), 0.02)
        self.assertEqual(result.contacts.time.shape, (1, 2, 0))
        self.assertEqual(result.bounce_count.max(), 0)
        result.position[..., 0, :] = 999
        for x, original in zip((p, v, w), originals):
            torch.testing.assert_close(x, original)

    def test_vacuum_bounce_events_and_samples_match_independent_analytic_solution(self):
        p, v = tensor([[0, 0, 1]]), tensor([[4, 0, -3]])
        result = self.trajectory(p, v, aero=self.vacuum(), table_height=0.78)
        t_hit = (-3 + (9 + 2*9.81*0.22)**0.5)/9.81
        speed = 3 + 9.81*t_hit
        restitution = 0.98 - 0.02*speed
        up = restitution * speed
        tangential_dv = min(0.4*4, 0.25*(1+restitution)*speed)
        out_v = tensor([4-tangential_dv, 0, up])
        out_w = tensor([0, tangential_dv*0.02/((2/3)*0.02**2), 0])
        hit_p = tensor([4*t_hit, 0, 0.78])
        t_second = t_hit + 2*up/9.81
        torch.testing.assert_close(result.contacts.time, tensor([[t_hit, t_second]]))
        torch.testing.assert_close(result.contacts.position[0, 0], hit_p)
        torch.testing.assert_close(result.contacts.velocity_before[0, 0], tensor([4, 0, -speed]))
        torch.testing.assert_close(result.contacts.velocity_after[0, 0], out_v)
        torch.testing.assert_close(result.contacts.spin_after[0, 0], out_w)
        self.assertEqual(result.contacts.bounced.tolist(), [[True, False]])
        valid = result.time <= t_second
        torch.testing.assert_close(result.valid[0], valid)
        before = result.time < t_hit
        t = result.time[:, None]
        dt = t - t_hit
        g = tensor([0, 0, -9.81])
        expected_p = torch.where(before[:, None], p + t*v + 0.5*t*t*g,
                                 hit_p + dt*out_v + 0.5*dt*dt*g)
        expected_v = torch.where(before[:, None], v+t*g, out_v+dt*g)
        torch.testing.assert_close(result.position[0, valid], expected_p[valid], rtol=0, atol=1e-12)
        torch.testing.assert_close(result.velocity[0, valid], expected_v[valid], rtol=0, atol=1e-12)
        self.assertTrue(torch.isnan(result.position[0, ~valid]).all())
        self.assertTrue((result.bounce_count[0, ~valid] == -1).all())
        # The physical x coordinates are never clipped to a robot workspace.
        self.assertGreater(result.position[0, valid, 0].max(), 1)

    def test_two_contacts_in_one_step_and_terminal_exactly_at_grid_time(self):
        p, v = tensor([[0, 0, 0.78]]), tensor([[0, 0, -1]])
        short = self.trajectory(p, v, aero=self.vacuum(), table_height=0.78,
                                restitution=0.001, restitution_speed_slope=0,
                                step_s=0.02, max_time_s=0.04)
        torch.testing.assert_close(short.contacts.time, tensor([[0, 0.002/9.81]]))
        self.assertEqual(short.valid.tolist(), [[True, False, False]])
        endpoint = self.trajectory(p, v, aero=self.vacuum(), table_height=0.78,
                                   restitution=1, restitution_speed_slope=0,
                                   gravity=1, step_s=2, max_time_s=4)
        torch.testing.assert_close(endpoint.contacts.time, tensor([[0, 2]]))
        self.assertEqual(endpoint.valid.tolist(), [[True, True, False]])
        torch.testing.assert_close(endpoint.position[0, 1], p[0])
        # Sample zero remains the input; event metadata carries the t=0 jump.
        torch.testing.assert_close(endpoint.velocity[0, 0], v[0])
        self.assertEqual(endpoint.contacts.velocity_after[0, 0, 2], 1)

    def test_stop_at_first_contact_and_zero_restitution(self):
        p, v = tensor([[0, 0, 1]]), tensor([[1, 0, -1]])
        stopped = self.trajectory(p, v, aero=self.vacuum(), table_height=0.78, max_bounces=0)
        self.assertEqual(stopped.contacts.time.shape, (1, 1))
        self.assertTrue(stopped.contacts.valid.all())
        self.assertFalse(stopped.contacts.bounced.any())
        torch.testing.assert_close(stopped.contacts.velocity_after, stopped.contacts.velocity_before)
        zero = self.trajectory(p, v, aero=self.vacuum(), table_height=0.78, restitution=0,
                                restitution_speed_slope=0)
        self.assertEqual(zero.contacts.valid.tolist(), [[True, False]])
        self.assertFalse(zero.contacts.bounced.any())
        torch.testing.assert_close(zero.valid, stopped.valid)

    def test_mixed_invalid_boundary_and_horizon_states(self):
        p = tensor([[0, 0, 0.7], [0, 0, float("nan")], [0, 0, 50], [0, 0, 0.78], [0, 0, 0.78]])
        v = tensor([[0, 0, -1], [0, 0, 0], [0, 0, 20], [0, 0, 0], [0, 0, 2]])
        result = self.trajectory(p, v, table_height=0.78, max_time_s=0.02)
        self.assertEqual(result.valid.tolist(), [[False, False], [False, False], [True, True], [True, False], [True, True]])
        self.assertTrue(torch.isnan(result.position[:2]).all())
        self.assertTrue(torch.isnan(result.contacts.time[:3]).all())
        self.assertEqual(result.contacts.time[3, 0], 0)
        self.assertTrue(result.contacts.valid[3, 0])
        overflow = self.trajectory(tensor([[0, 0, 1]]).float(), tensor([[1e30, 0, 0]]).float(), max_time_s=0.02)
        self.assertEqual(overflow.valid.tolist(), [[True, False]])

    def test_full_free_trajectory_matches_independent_fine_rk4(self):
        p = tensor([[0, 0, 2], [1, -1, 3], [-2, 0, 1]])
        v = tensor([[5, 0, 3], [-5, 1, 4], [6, -2, 2]])
        w = tensor([[0, 100, 0], [0, 100, 50], [0, 0, -200]])
        field = self.field()
        result = self.trajectory(p, v, w, aero=field, max_time_s=0.4)
        rp, rv = p.clone(), v.clone()
        states_p, states_v = [rp], [rv]
        dt = 0.0005
        def acceleration(velocity):
            return field.compute_forces(velocity, w)[0]/0.0027 + tensor([0, 0, -9.81])
        for index in range(800):
            a = acceleration(rv)
            b = acceleration(rv + 0.5*dt*a)
            c = acceleration(rv + 0.5*dt*b)
            d = acceleration(rv + dt*c)
            rp = rp + dt*rv + dt**2*(a+b+c)/6
            rv = rv + dt*(a+2*b+2*c+d)/6
            if (index+1) % 40 == 0:
                states_p.append(rp)
                states_v.append(rv)
        torch.testing.assert_close(result.position, torch.stack(states_p, dim=1), rtol=0, atol=0.003)
        torch.testing.assert_close(result.velocity, torch.stack(states_v, dim=1), rtol=0, atol=0.005)

    def test_post_bounce_remainder_uses_outgoing_spin(self):
        p, v = tensor([[0, 0, 0.79]]), tensor([[4, 0, -3]])
        field = self.field()
        result = self.trajectory(p, v, aero=field, table_height=0.78, max_time_s=0.1)
        self.assertTrue(result.contacts.bounced[0, 0])
        self.assertGreater(result.contacts.spin_after[0, 0, 1], 0)
        # Reintegrate the interval after impact with a much smaller step.
        duration = float(result.time[1] - result.contacts.time[0, 0])
        initial = (result.contacts.position[:, 0], result.contacts.velocity_after[:, 0])
        reference = self.trajectory(*initial, result.contacts.spin_after[:, 0], aero=field,
                                    max_time_s=duration, step_s=0.0001)
        wrong = self.trajectory(*initial, torch.zeros_like(v), aero=field,
                                max_time_s=duration, step_s=0.0001)
        torch.testing.assert_close(result.position[:, 1], reference.position[:, -1], rtol=0, atol=0.0001)
        torch.testing.assert_close(result.velocity[:, 1], reference.velocity[:, -1], rtol=0, atol=0.001)
        self.assertGreater(abs(result.velocity[0, 1, 2]-wrong.velocity[0, -1, 2]), 0.01)
        fine = self.trajectory(p, v, aero=field, table_height=0.78, step_s=0.0005, max_time_s=0.1)
        torch.testing.assert_close(result.position, fine.position[:, ::40], rtol=0, atol=0.003)
        torch.testing.assert_close(result.velocity, fine.velocity[:, ::40], rtol=0, atol=0.005)

    def test_wind_and_float32_batched_shapes(self):
        p, v, w = tensor([[[0, 0, 1], [1, 2, 3]]]), tensor([[[5, 0, 2], [4, 1, 1]]]), tensor([[[0, 100, 0], [0, 0, -100]]])
        baseline = self.trajectory(p, v, w, max_time_s=0.1)
        wind = tensor([2, -1, 0])
        moved = self.trajectory(p, v+wind, w, aero=self.field(air_velocity_w=tuple(wind.tolist())), max_time_s=0.1)
        torch.testing.assert_close(moved.position, baseline.position + baseline.time[:, None]*wind)
        torch.testing.assert_close(moved.velocity, baseline.velocity + wind)
        single = self.trajectory(p.float(), v.float(), w.float(), max_time_s=0.1)
        self.assertEqual(single.position.shape, (1, 2, 6, 3))
        self.assertEqual(single.time.dtype, torch.float32)
        torch.testing.assert_close(single.position.double(), baseline.position, rtol=0, atol=1e-5)

    def test_bad_configuration_and_shapes_fail_explicitly(self):
        p, v = tensor([[0, 0, 1]]), tensor([[1, 0, 0]])
        for override in (dict(step_s=0), dict(mass_kg=-1), dict(max_time_s=float("nan")),
                         dict(table_height=float("nan")), dict(max_bounces=2), dict(max_bounces=True),
                         dict(friction=-1), dict(restitution_speed_slope=-1), dict(restitution=2)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.trajectory(p, v, **override)
        with self.assertRaises(ValueError):
            self.trajectory(p, v.float())
        with self.assertRaises(ValueError):
            self.trajectory(p, v[:, :2])

    def test_environment_api_uses_current_state_and_does_not_mutate_targets(self):
        path = Path(__file__).resolve().parents[1]/"legged_lab/envs/base/tt_env.py"
        tree = ast.parse(path.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "predict_ball_trajectory")
        namespace = dict(predict_timed_trajectory=predict_timed_trajectory)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
        origin = tensor([[10, 20, 0]])
        p, v, w = tensor([[1, 0, 1]]), tensor([[4, 0, -1]]), tensor([[0, 50, 0]])
        table = SimpleNamespace(surface_height=0.76, bounce_restitution=0.98,
                                bounce_restitution_speed_slope=0.02, bounce_friction=0.25,
                                bounce_tangential_restitution=0)
        env = SimpleNamespace(aero=self.field(), cfg=SimpleNamespace(table=table),
                              table=SimpleNamespace(data=SimpleNamespace(root_pos_w=origin)),
                              ball=SimpleNamespace(data=SimpleNamespace(root_pos_w=p+origin,
                                  root_lin_vel_w=v, root_ang_vel_w=w),
                                  cfg=SimpleNamespace(spawn=SimpleNamespace(mass_props=SimpleNamespace(mass=0.0027)))),
                              ball_pos=tensor([[-99, 0, 1]]), ball_future_pose=tensor([[-1.6, 0, 1]]))
        target = env.ball_future_pose.clone()
        result = namespace["predict_ball_trajectory"](env, max_time_s=0.1)
        direct = self.trajectory(p, v, w, aero=env.aero, table_height=0.78, max_time_s=0.1)
        torch.testing.assert_close(result.position, direct.position)
        torch.testing.assert_close(env.ball_future_pose, target)
        torch.testing.assert_close(env.ball.data.root_pos_w, p+origin)
        self.assertGreater(result.position[0, 0, 0], -1.6)
