# SPDX-License-Identifier: MIT
"""CPU USD-authoring checks; Isaac Lab spawning is stubbed, not simulated."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None

from legged_lab.physics.ball_impact import resolve_ball_impact

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(Usd is None, "USD authoring tests require pxr (Isaac Sim or usd-core)")
class BallInertiaTest(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1.0)
        self.cfg = SimpleNamespace(radius=0.02, mass_props=SimpleNamespace(mass=0.0027))

        def spawn_sphere(path, cfg, translation=None, orientation=None):
            prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass_props.mass)
            sphere = UsdGeom.Sphere.Define(self.stage, path + "/geometry/mesh")
            sphere.CreateRadiusAttr(cfg.radius)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            return prim

        self.spawn = Mock(side_effect=spawn_sphere)
        self.clone = Mock(side_effect=lambda function: function)
        sim = SimpleNamespace(spawn_sphere=self.spawn)
        stubs = {"isaaclab": SimpleNamespace(sim=sim), "isaaclab.sim": sim,
                 "isaaclab.sim.utils": SimpleNamespace(clone=self.clone)}
        spec = importlib.util.spec_from_file_location(
            "pace_ball_spawner_test", ROOT / "legged_lab/assets/table_tennis/ball_spawner.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        self.spawn_ball = module.spawn_thin_shell_ball

    def test_nominal_inertia_on_rigid_root_and_pose_forwarding(self):
        position, orientation = (1.35, 0, 1.03), (1, 0, 0, 0)
        prim = self.spawn_ball("/Ball", self.cfg, position, orientation)
        self.spawn.assert_called_once_with("/Ball", self.cfg, translation=position, orientation=orientation)
        api = UsdPhysics.MassAPI(prim)
        for component in api.GetDiagonalInertiaAttr().Get():
            self.assertAlmostEqual(component / 7.2e-7, 1.0, places=6)
        self.assertEqual(api.GetCenterOfMassAttr().Get(), Gf.Vec3f(0))
        self.assertEqual(api.GetPrincipalAxesAttr().Get(), Gf.Quatf(1))
        self.assertAlmostEqual(api.GetMassAttr().Get(), 0.0027)
        self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertEqual(UsdGeom.Sphere(self.stage.GetPrimAtPath("/Ball/geometry/mesh")).GetRadiusAttr().Get(), 0.02)

    def test_inertia_tracks_configured_mass_and_radius(self):
        for i, (mass, radius, ratio) in enumerate(((0.0054, 0.02, 2), (0.0027, 0.04, 4))):
            cfg = SimpleNamespace(mass_props=SimpleNamespace(mass=mass), radius=radius)
            inertia = UsdPhysics.MassAPI(self.spawn_ball(f"/Ball{i}", cfg)).GetDiagonalInertiaAttr().Get()
            self.assertAlmostEqual(inertia[0] / 7.2e-7, ratio, places=6)

    def test_invalid_inputs_fail_before_spawning(self):
        for field in ("mass", "radius"):
            for value in (0, -1, float("inf"), float("nan")):
                cfg = SimpleNamespace(radius=0.02, mass_props=SimpleNamespace(mass=0.0027))
                setattr(cfg.mass_props if field == "mass" else cfg, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.spawn_ball("/Ball", cfg)
        for mass_props in (None, SimpleNamespace(mass=None)):
            with self.assertRaises(ValueError):
                self.spawn_ball("/Ball", SimpleNamespace(radius=0.02, mass_props=mass_props))
        self.spawn.assert_not_called()

    def test_authored_inertia_survives_usd_copy_and_inheritance(self):
        # Check USD behavior after authoring; actual Isaac Cloner is not run here.
        self.clone.assert_called_once()
        self.spawn_ball("/Source", self.cfg)
        layer = self.stage.GetRootLayer()
        Sdf.CopySpec(layer, "/Source", layer, "/Copy")
        self.stage.DefinePrim("/Inherited").GetInherits().AddInherit("/Source")
        for path in ("/Source", "/Copy", "/Inherited"):
            api = UsdPhysics.MassAPI(self.stage.GetPrimAtPath(path))
            self.assertTrue(api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion())
            self.assertAlmostEqual(api.GetDiagonalInertiaAttr().Get()[0] / 7.2e-7, 1, places=6)

    def test_offline_spin_change_matches_authored_inertia(self):
        api = UsdPhysics.MassAPI(self.spawn_ball("/Ball", self.cfg))
        inertia = api.GetDiagonalInertiaAttr().Get()[1]
        vector = lambda x: torch.tensor(x, dtype=torch.float64)
        incoming = vector([4, 0, -1])
        velocity, spin = resolve_ball_impact(
            incoming, vector([0, 0, 0]), vector([0, 0, 1]),
            radius=self.cfg.radius, restitution=0.5, friction=0.2,
        )
        impulse = self.cfg.mass_props.mass * (velocity - incoming)
        expected_spin = torch.linalg.cross(vector([0, 0, -self.cfg.radius]), impulse) / inertia
        torch.testing.assert_close(spin, expected_spin, rtol=1e-6, atol=1e-8)


class BallSpawnerWiringTest(unittest.TestCase):
    def test_active_ball_config_uses_thin_shell_spawner(self):
        tree = ast.parse((ROOT / "legged_lab/assets/table_tennis/ball.py").read_text())
        spheres = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "SphereCfg"]
        self.assertEqual(len(spheres), 1)
        func = next(keyword.value for keyword in spheres[0].keywords if keyword.arg == "func")
        self.assertIsInstance(func, ast.Name)
        self.assertEqual(func.id, "spawn_thin_shell_ball")
