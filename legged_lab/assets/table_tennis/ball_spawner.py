# SPDX-License-Identifier: MIT

import math

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from pxr import Gf, UsdPhysics


@clone
def spawn_thin_shell_ball(prim_path, cfg, translation=None, orientation=None):
    """Spawn a sphere with uniform thin-shell inertia, in PACE's SI units.

    Author inertia on the rigid-body root before the outer clone decorator
    copies it to other environments. No per-step mass-property update is needed.
    """
    mass = None if cfg.mass_props is None else cfg.mass_props.mass
    if mass is None or not math.isfinite(mass) or mass <= 0:
        raise ValueError("thin-shell ball requires an explicit positive finite mass")
    if not math.isfinite(cfg.radius) or cfg.radius <= 0:
        raise ValueError("thin-shell ball radius must be positive and finite")

    # The outer decorator supplies a concrete path, so the stock spawner only
    # creates one sphere here. Its geometry, materials and rigid settings remain.
    prim = sim_utils.spawn_sphere(prim_path, cfg, translation=translation, orientation=orientation)
    inertia = (2.0 / 3.0) * mass * cfg.radius**2
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    return prim
