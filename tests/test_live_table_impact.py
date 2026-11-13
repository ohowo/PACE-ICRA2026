# SPDX-License-Identifier: MIT

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from legged_lab.physics.aerodynamics import AeroForceField
from legged_lab.physics.live_table_impact import LiveTableImpact, replace_table_response
from legged_lab.physics.table_prediction import predict_table_rebound_state


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def table_config():
    path = Path(__file__).resolve().parents[1] / 'legged_lab/envs/base/tt_config.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TableCfg')
    cls.decorator_list = []
    ns = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
    return ns['TableCfg']()


class LiveTableImpactTest(unittest.TestCase):
    def case(self, count=1, native_e=0.925):
        free = tensor([[4, 0, -3]] * count)
        native = tensor([[2, 0, native_e*3]] * count)
        force = tensor([[0, 0, 0.0027*(1+native_e)*3/0.002]] * count)
        return dict(free_velocity=free, incoming_spin=torch.zeros_like(free),
                    native_velocity=native, native_spin=torch.full_like(free, 10),
                    before_position=tensor([[0.5, 0, 0.784]]*count),
                    after_position=tensor([[0.504, 0, 0.781]]*count),
                    table_force=force, total_force=force.clone(),
                    table_is_stationary=torch.ones(count, dtype=torch.bool),
                    dt=0.002, mass=0.0027, radius=0.02, surface_height=0.76,
                    x_bounds=(-1.37, 1.37), y_bounds=(-0.7625, 0.7625),
                    restitution=0.98, restitution_speed_slope=0.02,
                    friction=0.25, tangential_restitution=0, bounce_threshold=0.5)

    def test_replaces_native_response_and_matches_prediction(self):
        for native_e in (0, 0.2, 0.925, 1):
            case = self.case(native_e=native_e)
            v, w, mask = replace_table_response(**case)
            self.assertTrue(mask.all())
            # Independent impulse check: e=.92, qn=5.76, qt=1.44; I/m=2r²/3.
            torch.testing.assert_close(v, tensor([[2.56, 0, 2.76]]))
            torch.testing.assert_close(w, tensor([[0, 108, 0]]))
            predicted_v, predicted_w = predict_table_rebound_state(
                case['free_velocity'], case['incoming_spin'], tensor([0]),
                drag_k=0, gravity=9.81, radius=0.02)
            torch.testing.assert_close(v, predicted_v)
            torch.testing.assert_close(w, predicted_w)

    def test_speed_dependence_uses_contact_impulse_not_launch_or_native_e(self):
        case = self.case(4)
        u = tensor([1, 2.5, 2.7, 6])
        case['free_velocity'][:, 2] = 1.7  # Deliberately wrong launch-like vz.
        case['native_velocity'][:, 2] = 0.925*u
        case['table_force'][:, 2] = 0.0027*1.925*u/0.002
        case['total_force'] = case['table_force'].clone()
        v, w, mask = replace_table_response(**case)
        self.assertTrue(mask.all())
        torch.testing.assert_close(v[:, 2]/u, tensor([0.96, 0.93, 0.926, 0.86]))
        self.assertTrue(torch.all(w[:, 1] > 0))

    def test_unrelated_contacts_and_settling_keep_native_state(self):
        changes = [
            {'table_force': tensor([[0, 0, 0]]), 'total_force': tensor([[0, 0, 0]])},
            {'table_force': tensor([[4, 0, 0]]), 'total_force': tensor([[4, 0, 0]])},
            {'total_force': tensor([[2, 0, 7.79625]])},  # Other body's contact.
            {'before_position': tensor([[1.36, 0, .784]])},  # Table edge.
            {'after_position': tensor([[0, .76, .781]])},
            {'before_position': tensor([[0, 0, .935]]),
             'after_position': tensor([[0, 0, .935]])},  # Net top.
            {'before_position': tensor([[0, 0, .70]])},  # Underneath table.
            {'table_is_stationary': torch.tensor([False])},
            {'native_velocity': tensor([[2, 0, 6]])},  # Separating reconstructed speed.
            {'native_velocity': tensor([[2, 0, 5.575]])},  # u=.2; resting regime.
            {'incoming_spin': tensor([[float('nan'), 0, 0]])},
        ]
        for change in changes:
            with self.subTest(change=change):
                case = self.case() | change
                v, w, mask = replace_table_response(**case)
                self.assertFalse(mask.any())
                torch.testing.assert_close(v, case['native_velocity'])
                torch.testing.assert_close(w, case['native_spin'])

    def test_mixed_batch_preserves_inputs_shape_dtype_and_energy(self):
        for dtype in (torch.float32, torch.float64):
            case = self.case(4)
            for key, value in case.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    case[key] = value.to(dtype)
            case['incoming_spin'][0, 1] = 300  # Spin can increase translation.
            case['table_force'][2:] = 0
            case['total_force'][2:] = 0
            before = {k: v.clone() for k, v in case.items() if isinstance(v, torch.Tensor)}
            v, w, mask = replace_table_response(**case)
            self.assertEqual(mask.tolist(), [True, True, False, False])
            self.assertEqual(v.dtype, dtype)
            self.assertEqual(v.shape, (4, 3))
            torch.testing.assert_close(v[2:], case['native_velocity'][2:])
            torch.testing.assert_close(w[2:], case['native_spin'][2:])
            i = (2/3)*.02**2
            e0 = case['free_velocity'].square().sum(-1) + i*case['incoming_spin'].square().sum(-1)
            e1 = v.square().sum(-1) + i*w.square().sum(-1)
            self.assertTrue(torch.all(e1[:2] <= e0[:2]))
            for key, original in before.items():
                torch.testing.assert_close(case[key], original)

    def test_constant_override_and_extreme_speed_do_not_select_net_top(self):
        v, _, mask = replace_table_response(**(self.case() | {'restitution': .9, 'restitution_speed_slope': 0}))
        self.assertTrue(mask.all())
        torch.testing.assert_close(v, tensor([[2.575, 0, 2.7]]))
        case = self.case()
        case['native_velocity'][:, 2] = .925*100
        case['table_force'][:, 2] = .0027*1.925*100/.002
        case['total_force'] = case['table_force'].clone()
        case['before_position'][:, 2] = .935
        case['after_position'][:, 2] = .935
        v, _, mask = replace_table_response(**case)
        self.assertFalse(mask.any())
        torch.testing.assert_close(v, case['native_velocity'])

    def test_adapter_and_actual_substep_loop_apply_once_before_observation(self):
        cfg = table_config()
        self.assertTrue(cfg.live_impact_enabled)
        field = AeroForceField(device='cpu', radius_m=.02, air_density=1.225,
                               drag_coeff=.4378, magnus_model='conti2026')
        # Second env is translated and has no table contact (a paddle response).
        origins = tensor([[10, 20, 0], [30, 20, 0]])
        pos = tensor([[.5, 0, .784]]*2) + origins
        v0, w0 = tensor([[4, 0, -3]]*2), tensor([[0, 0, 0]]*2)
        ball = Mock()
        ball.cfg = SimpleNamespace(spawn=SimpleNamespace(radius=.02, mass_props=SimpleNamespace(mass=.0027)))
        ball.data = SimpleNamespace(root_pos_w=pos.clone(), root_lin_vel_w=v0.clone(), root_ang_vel_w=w0.clone())
        table = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=origins, root_quat_w=tensor([[1, 0, 0, 0]]*2),
            root_lin_vel_w=torch.zeros_like(v0), root_ang_vel_w=torch.zeros_like(v0)))
        sensor = SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.zeros((2, 1, 1, 3), dtype=torch.float64),
            net_forces_w=torch.zeros((2, 1, 3), dtype=torch.float64)))
        dt, order, native_e = .002, [], .925
        live = LiveTableImpact(ball, table, sensor, field, cfg, dt, .5, (0, 0, -9.81))
        def stage(**kw):
            order.append('force')
        ball.set_external_force_and_torque.side_effect = stage
        free = None
        def step(render):
            nonlocal free
            order.append('step')
            free = v0 + dt*(field.last_forces_w/.0027 + tensor([0, 0, -9.81]))
            ball.data.root_lin_vel_w = tensor([[2, 0, -native_e*free[0, 2]], [7, 1, 2]])
            ball.data.root_ang_vel_w = torch.full_like(w0, 12)
            ball.data.root_pos_w = pos + tensor([[.004, 0, -.003]]*2)
        def update(dt):
            order.append('sensor_update')
            impulse = .0027*(1+native_e)*(-free[0, 2])
            sensor.data.force_matrix_w[0, 0, 0, 2] = impulse/dt
            sensor.data.net_forces_w.copy_(sensor.data.force_matrix_w.sum(2))
            sensor.data.net_forces_w[1, 0, 0] = 5  # Paddle contact only.
        def write_velocity(velocity):
            order.append('replace')
            ball.data.root_lin_vel_w = velocity[:, :3].clone()
            ball.data.root_ang_vel_w = velocity[:, 3:].clone()
        ball.write_root_velocity_to_sim.side_effect = write_velocity
        def observe():
            order.append('observe')
            expected_v, expected_w = predict_table_rebound_state(
                free[:1], w0[:1], tensor([0]), drag_k=0, gravity=9.81, radius=.02)
            torch.testing.assert_close(ball.data.root_lin_vel_w[:1], expected_v)
            torch.testing.assert_close(ball.data.root_ang_vel_w[:1], expected_w)
            torch.testing.assert_close(ball.data.root_lin_vel_w[1], tensor([7, 1, 2]))
            torch.testing.assert_close(ball.data.root_pos_w, pos + tensor([[.004, 0, -.003]]*2))
        env = SimpleNamespace(cfg=SimpleNamespace(sim=SimpleNamespace(decimation=1)),
                              sim_step_counter=0, robot=Mock(), action_joint_ids=[],
                              aero=field, ball=ball, physics_dt=dt, live_table_impact=live,
                              sim=SimpleNamespace(step=step),
                              scene=SimpleNamespace(write_data_to_sim=lambda: order.append('write'), update=update),
                              compute_perception=observe, compute_paddle_touch=lambda: None)
        source = Path(__file__).resolve().parents[1] / 'legged_lab/envs/base/tt_env.py'
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'step')
        loop = next(n for n in method.body if isinstance(n, ast.For))
        wrapper = ast.parse('def advance(self, processed_actions):\n    pass').body[0]
        wrapper.body = [loop]
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(source), 'exec'), ns)
        ns['advance'](env, None)
        self.assertEqual(order, ['force', 'write', 'step', 'sensor_update', 'replace', 'observe'])
        self.assertEqual(live.last_replaced.tolist(), [True, False])
        ball.write_root_velocity_to_sim.assert_called_once()
        with self.assertRaises(RuntimeError):
            live.apply()  # Reusing this contact without a new substep is rejected.
        # A later free-flight substep cannot apply the previous contact again.
        field.apply_to_rigid_object(ball)
        live.prepare()
        sensor.data.force_matrix_w.zero_()
        sensor.data.net_forces_w.zero_()
        before = ball.data.root_lin_vel_w.clone()
        live.apply()
        self.assertFalse(live.last_replaced.any())
        torch.testing.assert_close(ball.data.root_lin_vel_w, before)

    def test_scene_enables_sensor_on_ball_and_filters_actual_table_body(self):
        path = Path(__file__).resolve().parents[1] / 'legged_lab/envs/base/tt_env.py'
        tree = ast.parse(path.read_text())
        scene = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TTSceneCfg')
        setup = next(n for n in ast.walk(scene) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == 'live_table_impact_enabled')
        env = SimpleNamespace(ball=SimpleNamespace(spawn=SimpleNamespace()))
        ns = dict(self=env, physics_dt=.002, ContactSensorCfg=SimpleNamespace, live_table_impact_enabled=True)
        code = compile(ast.Module(body=[setup], type_ignores=[]), str(path), 'exec')
        exec(code, ns)
        self.assertTrue(env.ball.spawn.activate_contact_sensors)
        self.assertEqual(env.ball_table_contact.prim_path, '{ENV_REGEX_NS}/Ball')
        self.assertEqual(env.ball_table_contact.filter_prim_paths_expr, ['{ENV_REGEX_NS}/Table/base_link'])
        self.assertEqual(env.ball_table_contact.update_period, .002)
        self.assertEqual(env.ball_table_contact.history_length, 0)
        disabled = SimpleNamespace(ball=SimpleNamespace(spawn=SimpleNamespace()))
        exec(code, ns | dict(self=disabled, live_table_impact_enabled=False))
        self.assertFalse(hasattr(disabled, 'ball_table_contact'))
        self.assertFalse(hasattr(disabled.ball.spawn, 'activate_contact_sensors'))
