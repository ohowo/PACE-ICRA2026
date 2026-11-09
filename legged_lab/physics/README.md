
# Isaac Lab Ball Aerodynamics

[External data and parameter provenance](../../docs/external-data-sources.md)


GPU batched aerodynamic drag and Magnus forces for spinning balls in Isaac Lab. Implemented in PyTorch. Drop-in force field for `RigidObject` assets.


## Quickstart


```python
from legged_lab.physics.aerodynamics import AeroForceField
from isaaclab.assets.rigid_object import RigidObject

class DummyEnv:
    def __init__(self, sim, scene, ball: RigidObject):
        self.sim = sim
        self.scene = scene
        self.ball = ball

        # Initialize aerodynamics AFTER scene is built, BEFORE resets/buffers
        self.aero = AeroForceField(
            device="cuda:0",
            radius_m=0.020,
            air_density=1.225,
            drag_coeff=0.43,   # set 0.0 to disable
            magnus_factor=0.0
        )

    def step(self):
        # Call once per physics substep, BEFORE scene.write_data_to_sim()
        self.aero.apply_to_rigid_object(self.ball)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=0.01)
```

### Example Ball Definition

The ball used in above example can be defined as follows:
```python
from isaaclab.assets.rigid_object import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

ball_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.02,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.2)),
    ),
)

scene = InteractiveScene(scene_cfg)     # scene_cfg includes ball_cfg
ball: RigidObject = scene["ball"]       # handle used by AeroForceField
```


## Features

* Drag and Magnus forces in world frame
* GPU batched PyTorch compute
* Works with `RigidObject` or collections
* Pluggable model API for custom aero



# Aerodynamic Models

Math and usage for the batched aerodynamic models used by the adapter in `aerodynamics.py`.

## Model

Let $\mathbf{v}\in\mathbb{R}^3$ be world linear velocity, $\boldsymbol{\omega}\in\mathbb{R}^3$ world angular velocity, $\rho$ air density, $A=\pi r^2$ area, $r$ radius, $C_d$ drag coefficient, $\kappa$ Magnus factor.

$$
\mathbf{F}_\text{drag} = -\tfrac{1}{2}\rho A C_d \|\mathbf{v}\| \mathbf{v}, \quad
\mathbf{F}_\text{magnus} = \tfrac{1}{2}\rho A r \kappa (\boldsymbol{\omega}\times \mathbf{v})
$$
$$
\mathbf{F}_\text{total}=\mathbf{F}_\text{drag}+\mathbf{F}_\text{magnus},\quad \boldsymbol{\tau}=\mathbf{0}
$$

Shapes: `(N, 3)` for forces and torques. World frame.


## Adapter usage

```python
from legged_lab.physics.aerodynamics import AeroForceField

aero = AeroForceField(
    device="cuda:0",
    radius_m=0.020,
    air_density=1.225,
    drag_coeff=0.47,
    magnus_factor=0.0
)

# Called once per substep before staging to the sim
aero.apply_to_rigid_object(ball_asset)   # ball_asset: isaaclab.assets.RigidObject
F_w = aero.last_forces_w    # (N, 3)
T_w = aero.last_torques_w   # (N, 3), zero in this model
```

## Where to place calls in an env

* Initialize the adapter **after** the scene is created and assets are resolved.
* Initialize **before** buffer allocation and any `reset(...)` calls.
* In the step loop, call `apply_to_rigid_object(...)` **before** `scene.write_data_to_sim()` on each physics substep.


## Quick example: defining `self.ball`


```python
from isaaclab.assets.rigid_object import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

ball_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.02,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
)

scene = InteractiveScene(scene_cfg)            # scene_cfg includes ball_cfg
self.scene = scene
self.ball: RigidObject = self.scene["ball"]    # RigidObject handle used by the aero adapter

```


## Direct model API


```python
import torch
from legged_lab.physics.aero_model import MeasuredAeroModel

model = MeasuredAeroModel(device="cuda:0")
v = torch.randn(1024, 3, device="cuda")
w = torch.randn(1024, 3, device="cuda")
ball_props = {"radii": torch.full((1024,), 0.02, device="cuda"),
              "areas": torch.full((1024,), 3.1415926*0.02**2, device="cuda")}
env_props = {"air_density": 1.225}
params = {"drag_coefficients": torch.full((1024,), 0.43, device="cuda"),
          "magnus_factors":    torch.full((1024,), 0.0,  device="cuda")}
F, T = model.calculate_forces_and_torques(v, w, ball_props, env_props, params)
```


## Tuning

* `drag_coeff`: fit from deceleration vs. speed; typical spheres \~0.4–0.5.
* With `magnus_model="constant"`, `magnus_factor` is the fixed kappa; `0.0` disables Magnus.
* Use SI units and world-frame velocities.

## Dynamic Magnus force in the live task

`cfg.ball.magnus_model = "conti2026"` is now the task default. At each
physics substep the force adapter reads current world velocity and angular
velocity, subtracts air velocity, and recomputes

`kappa(t) = (8/3) * C_M(|v(t) - u|, |omega(t)|)`

`F_M(t) = 0.5 * rho * pi * r^3 * kappa(t) * (omega(t) x (v(t) - u))`.

The scalar coefficient uses the piecewise curves in Conti et al.,
arXiv:2606.28805v2, Eq. (5), Tables II-III. Evaluate the two neighboring
speed curves at current spin, then interpolate their nonnegative C_M
values linearly in speed. The adapter caches only table constants, not
the changing coefficient. Computation is batched with no per-ball Python
loop. See the [source register](../../docs/external-data-sources.md#dynamic-magnus-force-2026-11-06)
for all coefficients, units, transformations and limitations.

The lookup speed is clamped to 2-17 m/s and lookup spin to 0-1000 rad/s.
Actual vectors still enter the cross product; forces outside that envelope
are unvalidated extensions. The rounded source curves have small jumps
at some spin breakpoints, which are retained rather than silently refitted.
This reference model is restricted to the nominal 0.020 m radius. It is
not a calibration for the exact PACE ball or a dynamic-stall model.

The task applies force before `sim.step` and refreshes state afterward,
every `physics_dt` (currently 0.002 s). PhysX integrates forces over time:
the force in newtons is not multiplied by dt in the adapter. Time dependence
comes from the evolving state; no arbitrary elapsed-time multiplier or
spin-decay law is added. Aerodynamic torque remains zero.

The standalone `AeroForceField` constructor keeps its `"constant"` default
for compatibility with existing callers. In the task, select
`cfg.ball.magnus_model = "constant"` to use the existing `magnus_factor`
(zero disables the force). In `"conti2026"` mode that fixed factor is ignored.
Initial serve spin remains zero; Magnus becomes nonzero when contact or
configuration produces a spin component transverse to relative velocity.

**Scope:** with Magnus enabled, live flight, approach-to-table prediction and
post-bounce/current-ascent prediction use the same drag/Magnus force law.
CPU tests verify equations and environment routing with stand-in physics;
contact-model accuracy, real-world flight and training throughput remain
unverified. No Isaac Sim/GPU run has been performed.

## Approach-to-table prediction

`predict_table_contact` integrates current position and velocity under gravity,
vector drag and Magnus. It returns the first descending crossing of the
horizontal ball-center contact plane: position, incident velocity, elapsed
time and validity. The task uses `0.760 m + radius` as the plane height.
This is an infinite-plane flight prediction, not a test of table edges or
net collisions. Spin is held constant under the existing zero-torque law.

The explicit-midpoint solver uses the same maximum 0.020 s step and 1.50 s
horizon as the ascent predictor. Each step recomputes force twice using the
current and midpoint velocities. It solves the quadratic position interpolant
for contact within a step and interpolates incident velocity consistently;
the returned center height is exactly the contact plane. No per-ball Python
loop, scalar device readback or extra live force is introduced.

The task passes the numerical incident velocity to the rebound model with
zero remaining approach time, preventing double application of drag/gravity.
It then uses the returned linear/angular rebound velocities for ascent.
At-plane incoming/resting states contact immediately; upward states at the
plane are followed until they return. Initially below-plane, nonfinite and
horizon-exhausted states are invalid. `valid_land` exposes this status and
invalid pre-bounce targets use the existing task fallback. A failed landing
estimate does not invalidate the independent ascent of an already bounced
ball. With Magnus disabled, the old analytical flight path remains available.

CPU regression cases compare landing time, position and incident velocity
with independent 0.0005 s RK4 integration of the same force law, using 1 ms,
3 mm and 0.005 m/s tolerances. They also cover spin direction, moving air,
float32/float64 batches, boundary/invalid states and contact-to-rebound-to-apex
routing through the actual task code. These are numerical checks, not measured
accuracy bounds for real table-tennis flight.

## Rebound spin and ascent prediction

`predict_table_rebound_state` returns both linear velocity (m/s) and angular
velocity (rad/s). With configured `bounce_friction`, it retains both outputs
of the thin-shell impulse model, including spin generated by table friction:

`omega_out = omega_in + ((-r*n) x delta_v_t) / ((2/3)*r^2)`.

The default `bounce_friction=None` keeps the inherited 0.70 horizontal
multiplier and preserves incoming spin. This legacy mode has no angular
impulse law; preserving spin is a compatibility assumption, not a calibrated
prediction of contact-induced rotation. The old `predict_table_rebound` API
continues to return only linear velocity for external callers.

When Magnus is enabled, the task passes the returned rebound spin to
`predict_flight_apex`. For a ball that has already bounced, it instead uses
the measured current spin. The helper integrates gravity plus the same
vector drag and Magnus force used in live flight, recomputing force at each
midpoint stage. Spin remains constant in free flight because the current
aerodynamic model has zero torque. No extra live contact impulse is applied.

The batched midpoint solver uses a maximum 0.020 s step, a 1.50 s horizon
(75 iterations), and interpolation within the first step crossing `vz=0`.
These are numerical settings, not measured physical parameters. Initially
nonascending samples return their starting state at time zero. Samples
without a finite apex in the horizon are marked invalid and use the task's
existing invalid-target fallback. With Magnus disabled (`constant` mode,
zero factor), the existing analytical ascent path is retained.

CPU regression cases check spin directions, post-impact angular momentum,
propagation through the actual task prediction block, and comparison with
0.0005 s RK4 integration of the same force law (3 mm position / 1 ms time
tolerances for the tested cases). These tolerances are not real-world error
bounds. The solver has no per-ball Python loop or GPU scalar readback, but
training throughput and high-spin/horizon coverage remain unverified.

## Match ball and rebound reference

The live `BALL_CFG` uses `spawn_thin_shell_ball` to author the ball's inertia
on its USD rigid-body root before cloning environments. Each principal
component is `I = (2/3) * mass * radius**2`, or `7.2e-7 kg m²` for the
nominal ball. The center of mass is the sphere center. This uniform thin-shell
assumption matches `resolve_ball_impact`; it is not a measured inertia or an
ITTF inertia requirement. Mass and radius remain configurable, and inertia
is derived from them at spawn time with no per-step work. PACE uses SI stage
units and an unscaled sphere. Geometry and contact materials are unchanged.

CPU checks: `python -m unittest discover -s tests -p 'test_*.py'`.
The inertia authoring tests need `pxr` (available in Isaac Sim or `usd-core`);
without it those five tests are explicitly skipped. They use real USD stages
with a stub sphere spawner, verify copy/inheritance and compare angular impulse
with the offline model. They do not validate PhysX dynamics. Reading back
the initialized PhysX inertia and checking contact-induced spin remain runtime
validation tasks.

The `t1_tt` ball uses the ITTF nominal diameter of 40 mm, mass of 2.7 g,
and a matte white appearance. The simulation uses quadratic vector drag with
the existing PACE values of 1.225 kg/m³ for air density and 0.4378 for `Cd`.
The analytical predictor now gets `k = ρ A Cd / (2m)` from those same inputs
and the configured ball mass, giving 0.124804 m⁻¹. The Magnus-disabled
closed-form approach treats the axes separately; the enabled predictor uses
vector drag. The constant `Cd` is a PACE model
parameter. Experimental drag can depend on speed and spin.

The analytical fallback computes the incident horizontal velocity before
applying either rebound model. For each horizontal component,
`v_impact = v_now / (1 + k * abs(v_now) * time_to_impact)`, the closed-form
solution of `dv/dt = -k * abs(v) * v`. The default rebound retains 70% of
this incident velocity and uses `bounce_restitution` (default 0.90) for the
normal component. Previously the default branch applied 70% directly to the
current velocity, restoring speed lost during the approach to the table.

`predict_table_rebound` uses batched tensor operations with no integration
loop, new dependencies, or live PhysX impulse. The optional spin-aware branch
receives the same incident velocity. In that fallback, vertical approach is
ballistic and the separate-axis approach omits vector drag coupling. When
Magnus is enabled, the task instead supplies the numerical contact velocity
at zero remaining approach time. Neither path recalibrates the contact law
or aerodynamic coefficients.

ITTF Law 2.1.3 specifies about 230 mm rebound from a 300 mm drop onto the
playing surface. In the one-dimensional quadratic-drag model above, that
corresponds to a normal restitution of about 0.905. The existing analytical
table value of 0.90 gives about 228 mm in that model. PhysX resolves contact
using the rigid sphere and the contact materials.

The separate ITTF M3 ball approval test specifies a 240–265 mm rebound from a
305 mm drop onto a standard steel block. M3 also specifies a slow 50 N
indentation test, with 0.68–0.81 mm at the poles for a seamed ball and
0.70–0.90 mm for a seamless ball. These test conditions characterize ball
hardness and steel-block rebound. Dynamic stiffness, damping, and tangential
restitution for a particular table or racket rubber require separate data.

[ITTF Laws 2.1 and 2.3](https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf) · [ITTF M3 Ball Manual](https://equipments.ittf.com/#/equipment/manuals) · [Measured speed and spin dependence](https://arxiv.org/abs/2606.28805)

## Symmetric allround racket rubber

Both faces of the `t1_tt` racket use the same allround inverted-rubber
approximation. The reference is andro Power 3 with a 1.8 mm sponge on the two
blades used in the published racket 02 and 04 experiments. The collision
geometry adds a nominal 3.3 mm per face, including sponge, topsheet and adhesive,
as a thickness assumption. The resulting 14.5 mm cylinder uses a single contact
material, including its rim. The original link mass and inertia are retained.

A constant fit to the two published training sets gives about 0.70 normal
restitution. With tangential restitution fixed at zero, fitting the outgoing
tangential velocity gives 0.445 effective friction. The authored PhysX material
values are 0.79 friction and 0.50 restitution to obtain the fitted contact values
when averaged with the PACE ball (0.10 friction and 0.90 restitution).
Startup randomization retains this paddle material and samples the other robot
shapes, including the forearm collider on the same link.

For offline predictions with `resolve_ball_impact`, fitting with tangential
elasticity enabled gives 0.605 friction and 0.66 tangential restitution, with
the same 0.70 normal restitution. On the held-out data, the tangential-speed
RMSE was 1.43 m/s for the rigid impulse fit and 0.17 m/s with tangential
elasticity. These are fits to the published experiments. The live PhysX
material uses rigid contact and Coulomb friction.

[Racket and rubber experiments](https://arxiv.org/abs/2604.11349) · [Public training and test data](https://github.com/gossardt/tt_rackets/tree/main/data)
