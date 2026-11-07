# SPDX-License-Identifier: MIT

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from legged_lab.physics.aerodynamics import AeroForceField
from legged_lab.physics.magnus_model import ContiMagnusModel


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


class DynamicMagnusTest(unittest.TestCase):
    def field(self, **kwargs):
        return AeroForceField(device="cpu", radius_m=0.02, air_density=1.225,
                              drag_coeff=0.0, magnus_model="conti2026", **kwargs)

    def test_task_default_reaches_force_adapter(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "legged_lab/envs/base/tt_config.py").read_text())
        cfg_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BallCfg")
        cfg_class.decorator_list = []
        namespace = {"AeroForceField": AeroForceField}
        exec(compile(ast.Module(body=[cfg_class], type_ignores=[]), "BallCfg", "exec"), namespace)
        env = SimpleNamespace(device="cpu", cfg=SimpleNamespace(ball=namespace["BallCfg"]()),
                              ball=SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(radius=0.02))))
        namespace["self"] = env
        tree = ast.parse((root / "legged_lab/envs/base/tt_env.py").read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                          and n.value.func.id == "AeroForceField")
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "aero_init", "exec"), namespace)
        self.assertEqual(env.aero.magnus_model, "conti2026")
        self.assertEqual(env.cfg.ball.serve_spin_w, (0, 0, 0))
        f, _ = env.aero._compute_forces(tensor([[7.5, 0, 0]]), tensor([[0, 200, 0]]))
        self.assertAlmostEqual(f[0, 2].item(), -0.01293079536, places=8)

    def test_published_reference_and_volume_to_area_conversion(self):
        speed = tensor([7.5, 7.5, 7.5, 7.5])
        spin = tensor([100., 200., 300., 800.])
        # Eq. (5), Tables II-III: C_M = 0.37 - 0.0008*w up to 350;
        # at 800, C_M = -2e-7*800^2 + 1.7e-4*800 + 0.0587 = 0.0667.
        cm = tensor([0.29, 0.21, 0.13, 0.0667])
        torch.testing.assert_close(ContiMagnusModel()(speed, spin), cm * (8 / 3))
        v = torch.stack((speed, speed * 0, speed * 0), dim=-1)
        w = torch.stack((spin * 0, spin, spin * 0), dim=-1)
        force, _ = self.field()._compute_forces(v, w)
        expected = -1.225 * (4 / 3 * math.pi * 0.02**3) * cm * spin * speed
        torch.testing.assert_close(force[:, 2], expected)

    def test_all_speed_knots_interpolation_and_lookup_guards(self):
        model = ContiMagnusModel()
        speed = tensor([2, 3.5, 7.5, 10.5, 13.5, 17])
        cm_at_100 = tensor([0.08, 0.20, 0.29, 0.3092, 0.327, 0.3262])
        torch.testing.assert_close(model(speed, torch.full_like(speed, 100)), cm_at_100 * (8 / 3))
        mid = model(tensor([5.5]), tensor([100]))
        torch.testing.assert_close(mid, tensor([(0.20 + 0.29) / 2 * (8 / 3)]))
        torch.testing.assert_close(model(tensor([0, 30]), tensor([100, 100])),
                                   tensor([0.08, 0.3262]) * (8 / 3))
        torch.testing.assert_close(model(tensor([7.5]), tensor([1500])),
                                   model(tensor([7.5]), tensor([1000])))

    def test_spin_breakpoints_use_published_branches_and_nonnegative_coefficient(self):
        model = ContiMagnusModel()
        # Published rounded curves are not forced to be continuous at a breakpoint.
        expected = tensor([0.09, -1.667e-7 * 201**2 - 3.333e-5 * 201 + 0.1, 0.0])
        actual = model(tensor([3.5, 3.5, 2]), tensor([200, 201, 1000]))
        torch.testing.assert_close(actual, expected * (8 / 3))

    def test_force_directions_zero_limits_and_no_air_relative_work(self):
        v = tensor([[7.5, 0, 0]] * 5 + [[0, 0, 0]])
        w = tensor([[0, 200, 0], [0, -200, 0], [0, 0, 200], [200, 0, 0], [0, 0, 0], [0, 200, 0]])
        f, torque = self.field()._compute_forces(v, w)
        self.assertLess(f[0, 2], 0)
        torch.testing.assert_close(f[0], -f[1])
        self.assertGreater(f[2, 1], 0)
        torch.testing.assert_close(f[3:], torch.zeros_like(f[3:]))
        torch.testing.assert_close((f * v).sum(-1), torch.zeros(6, dtype=f.dtype))
        torch.testing.assert_close(torque, torch.zeros_like(torque))

    def test_relative_airflow_changes_coefficient_as_well_as_cross_product(self):
        v, w = tensor([[9.5, 0, 0]]), tensor([[0, 200, 0]])
        moving_air = self.field(air_velocity_w=(2, 0, 0))
        f, _ = moving_air._compute_forces(v, w)
        reference, _ = self.field()._compute_forces(tensor([[7.5, 0, 0]]), w)
        torch.testing.assert_close(f, reference)

    def test_same_batch_recomputes_after_speed_and_spin_changes(self):
        field = self.field()
        for speed, spin in ((7.5, 100), (3.5, 100), (3.5, 200), (7.5, 100)):
            f, _ = field._compute_forces(tensor([[speed, 0, 0]]), tensor([[0, spin, 0]]))
            cm = (0.37 - 0.0008 * spin) if speed == 7.5 else (0.31 - 0.0011 * spin)
            expected = -1.225 * (4 / 3 * math.pi * 0.02**3) * cm * spin * speed
            self.assertAlmostEqual(f[0, 2].item(), expected, places=8)

    def test_batch_shapes_dtype_and_constant_mode(self):
        field = self.field()
        for dtype in (torch.float32, torch.float64):
            v = torch.zeros((2, 3, 3), dtype=dtype)
            w = torch.zeros_like(v)
            v[..., 0], w[..., 1] = 7.5, 200
            f, _ = field._compute_forces(v, w)
            self.assertEqual(f.shape, v.shape)
            self.assertEqual(f.dtype, dtype)
            torch.testing.assert_close(f[..., 2], torch.full((2, 3), -0.01293079536, dtype=dtype))
        legacy = AeroForceField(device="cpu", drag_coeff=0, magnus_factor=0)
        f, _ = legacy._compute_forces(tensor([[7.5, 0, 0]]), tensor([[0, 200, 0]]))
        torch.testing.assert_close(f, torch.zeros_like(f))
        with self.assertRaises(ValueError):
            AeroForceField(device="cpu", magnus_model="unknown")
        with self.assertRaises(ValueError):
            AeroForceField(device="cpu", magnus_model="conti2026", radius_m=0.03)

    def test_actual_environment_substep_loop_recomputes_and_integrates_once(self):
        # Execute the actual substep loop with CPU stand-ins for robot and PhysX.
        source = Path(__file__).resolve().parents[1] / "legged_lab/envs/base/tt_env.py"
        tree = ast.parse(source.read_text())
        step = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "step")
        loop = next(n for n in step.body if isinstance(n, ast.For))
        wrapper = ast.parse("def advance(self, processed_actions):\n    pass").body[0]
        wrapper.body = [loop]
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(source), "exec"), namespace)
        field = self.field()
        ball = Mock()
        ball.data = SimpleNamespace(root_lin_vel_w=tensor([[7.5, 0, 0]]),
                                    root_ang_vel_w=tensor([[0, 200, 0]]))
        forces, order = [], []
        dt = 0.002

        def stage(**kwargs):
            forces.append(kwargs["forces"][:, 0].clone())
            self.assertTrue(kwargs["is_global"])
            order.append("force")

        def integrate(render):
            order.append("step")
            ball.data.root_lin_vel_w += forces[-1] / 0.0027 * dt

        ball.set_external_force_and_torque.side_effect = stage
        env = SimpleNamespace(
            cfg=SimpleNamespace(sim=SimpleNamespace(decimation=3)), sim_step_counter=0,
            robot=Mock(), action_joint_ids=[], aero=field, ball=ball, physics_dt=dt,
            sim=SimpleNamespace(step=integrate),
            scene=SimpleNamespace(write_data_to_sim=lambda: order.append("write"),
                                  update=lambda dt: order.append("update")),
            compute_perception=lambda: None, compute_paddle_touch=lambda: None,
        )
        namespace["advance"](env, None)
        self.assertEqual(order, ["force", "write", "step", "update"] * 3)
        self.assertEqual(env.sim_step_counter, 3)
        self.assertFalse(torch.equal(forces[0], forces[1]))
        self.assertAlmostEqual(forces[0][0, 2].item(), -0.01293079536, places=8)
        expected_velocity = tensor([[7.5, 0, 0]]) + sum(forces) / 0.0027 * dt
        torch.testing.assert_close(ball.data.root_lin_vel_w, expected_velocity)
