# External data and parameter provenance

This register records the external evidence used by PACE code and identifies
values inherited from PACE or chosen for the model. Source links point to a
specific revision or edition where available. Keep the register and audit log
current under the [repository maintenance rules](../AGENTS.md).

## Standards used in the model

| Source and locator | Value used | Implementation |
| --- | --- | --- |
| [ITTF Statutes 2026, Law 2.3](https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf#page=39) | Nominal diameter 40 mm and mass 2.7 g. The model uses matte white from the permitted appearances. | [Ball asset](../legged_lab/assets/table_tennis/ball.py) uses radius 0.020 m and mass 0.0027 kg. [TTEnv](../legged_lab/envs/base/tt_env.py) reads these from the asset for drag and geometric calculations. |
| [ITTF Statutes 2026, Laws 2.1.1 and 2.1.3](https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf#page=39) | Tabletop height 0.760 m. A 300 mm drop should rebound about 230 mm on the playing surface. | [TTEnv](../legged_lab/envs/base/tt_env.py) adds the ball radius to 0.760 m for the predicted contact height. The drop specification is a comparison benchmark. |
| [ITTF Statutes 2026, Laws 2.4.1 and 2.4.3](https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf#page=40) | The blade must be flat and rigid. A sponge covering including adhesive is less than 4.05 mm per playing face, subject to the applicable measurement tolerance. | The [allround paddle asset](../legged_lab/assets/booster/T1_TT/T1_TT_allround_rubber.usda) retains the rigid disk and uses a 3.3 mm nominal covering per face. The limit is a geometry constraint. The 3.3 mm value is a modeling assumption. |

### Standards consulted for comparison

- [ITTF M3 Ball Manual, October 2024](https://equipments.ittf.com/#/equipment/manuals),
  pp. 6, 13 and 14. A 305 mm drop onto the specified steel block should
  rebound 240-265 mm. The 50 N indentation limits are 0.68-0.81 mm at the poles
  for a seamed ball and 0.70-0.90 mm for a seamless ball. These characterize
  the approval test. PACE uses its existing contact settings for the
  ball-table and ball-racket pairs.
- [ITTF M4 Racket Coverings Manual, October 2024](https://sr.bttv.de/fileadmin/bttv/media/SR/pdf/M4_RC_MANUAL_OCTOBER_2024.pdf),
  pp. 10 and 16. The 50 mN friction test and its 0.50 threshold apply to
  outward-pimple and long-pimple coverings in the standards table. The
  inverted-rubber fit below uses measured ball impacts.

## Racket impact data and fit

Thomas Gossard's [racket bounce study, arXiv:2604.11349v1](https://arxiv.org/abs/2604.11349v1)
identifies the two configurations in Tables II-IV as andro Power 3 allround
inverted rubber with a 1.8 mm sponge. Racket 02 uses an offensive blade and
racket 04 uses a defensive blade. PACE uses this pair as a representative
allround reference and fits one shared set of coefficients for both faces.

The files below are pinned to dataset revision
[`159c6f040dd7832570b537b3b5fd00a97fc28c15`](https://github.com/gossardt/tt_rackets/commit/159c6f040dd7832570b537b3b5fd00a97fc28c15).
The [upstream repository includes an MIT license](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/LICENSE).
PACE stores the derived coefficients and links.

| Split and sample | Source file | Valid rows | Git blob SHA |
| --- | --- | ---: | --- |
| Training, racket 02 | [racket_02.csv](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/data/train/racket_02.csv) | 622 | `f3467213317e8133b3a3e75b33a8557976bff575` |
| Training, racket 04 | [racket_04.csv](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/data/train/racket_04.csv) | 660 | `2ea3e033ee257c82e089af72035aed113179199b` |
| Test, racket 02 | [racket_02.csv](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/data/test/racket_02.csv) | 156 | `6a1ea7f9626fad5927aa9a2ad4828c92b8243591` |
| Test, racket 04 | [racket_04.csv](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/data/test/racket_04.csv) | 165 | `63722ef320700354aab75531b029d5898ef8b2d7` |

### Reproducing the fit

1. Read the published CSV values in the racket frame. Keep rows with 14
   finite numeric fields, `v_in_z < 0` and `v_out_z > 0`. Pool the two
   training splits with equal weight per row. Reserve the two test splits for
   evaluation.
2. Fit normal restitution `e` by least squares through the origin for
   `v_out_z = e * (-v_in_z)`. The pooled result is `0.7008378486`.
3. Compute contact slip `s = v_in_x - r * spin_before_y` with the PACE radius
   `r = 0.020 m` and spin in rad/s. For candidate friction `mu` and
   tangential restitution `e_t`, predict tangential velocity with
   `v_out_x = v_in_x - sign(s) * min(0.4 * (1 + e_t) * abs(s), mu * (1 + e) * (-v_in_z))`.
   The factor 0.4 follows from the thin spherical shell inertia used by
   [the PACE impact model](../legged_lab/physics/ball_impact.py).
4. Minimize summed squared error in signed `v_out_x` over a friction grid
   from 0 to 1.5 in steps of 0.005. For the rigid fit, fix `e_t = 0`. For
   the offline elastic fit, also search `e_t` from 0 to 1 in steps of 0.02.
   Evaluate both fits on the 321 test rows using the fitted normal restitution.

| Fit | Normal restitution | Effective friction | Tangential restitution | Test tangential velocity RMSE |
| --- | ---: | ---: | ---: | ---: |
| Rigid contact approximation | 0.70 (rounded for use) | 0.445 | 0 (fixed) | 1.43 m/s |
| Offline elastic impulse | 0.70 (rounded for use) | 0.605 | 0.66 | 0.17 m/s |

### Mapping the data to PACE

- The [paddle asset](../legged_lab/assets/booster/T1_TT/T1_TT_allround_rubber.usda)
  authors 0.79 for both friction fields and 0.50 for material restitution.
  With the [PACE ball](../legged_lab/assets/table_tennis/ball.py) values of
  0.10 and 0.90, the average combine rule gives the fitted pair values of
  0.445 and 0.70. See the [PhysX combine rule](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-rigid-body-material-combine-modes).
  Static and dynamic friction use the same value as a simplification.
- The nominal 3.3 mm total covering per face includes sponge, topsheet and
  adhesive. Only the 1.8 mm sponge thickness comes from the study. The
  14.5 mm collision disk adds the two assumed coverings to the existing
  7.9 mm disk. The link mass and inertia come from the existing asset.
- The elastic fit is available for offline use with `resolve_ball_impact`
  and documented in the [physics README](../legged_lab/physics/README.md).
  The live PhysX setup uses the rigid material approximation. The reported
  errors are from the analytical fit on published data. PhysX contact
  response and transfer to the PACE racket are still unverified in Isaac Sim.

## Inherited settings and analytical assumptions

| Value or behavior | Provenance and status |
| --- | --- |
| Active PACE air density 1.225 kg/m³ and drag coefficient 0.4378 | Inherited from the upstream [PACE training environment](https://github.com/ohowo/PACE-ICRA2026/blob/859b26b411d442704c7b590a94d54cbfb3ba9bb6/legged_lab/envs/base/tt_env.py#L205-L213). The upstream comment describes real-world testing. A source dataset or fit record for 0.4378 was unavailable in the reviewed branch. |
| Analytical drag constant 0.1248038406 m⁻¹ | Derived in [the force adapter](../legged_lab/physics/aerodynamics.py) as `rho * pi * r**2 * Cd / (2 * mass)`. Uses the values above and the ITTF nominal ball. The CPU test checks the constant against the force. [Conti et al., arXiv:2606.28805v2](https://arxiv.org/abs/2606.28805v2) provides context on speed and spin dependence, without supplying a coefficient to PACE. |
| Table material friction 0.40 and restitution 0.95 | Inherited from the [table USD physics layer](../legged_lab/assets/table_tennis/table/configuration/ping_pong_table_urdf_physics.usd). These are authored material inputs. |
| Analytical tabletop restitution 0.90 and horizontal velocity multiplier 0.70 | Inherited from the [PACE predictor](https://github.com/ohowo/PACE-ICRA2026/blob/859b26b411d442704c7b590a94d54cbfb3ba9bb6/legged_lab/envs/base/tt_env.py#L981-L1022). In the idealized vertical quadratic-drag calculation, 0.90 gives a 227.5 mm rebound from a 300 mm drop. That is a derived comparison, with the actual PhysX rebound pending verification. |
| Still air, zero initial serve spin, and thin-shell inertia `I/m = 2r²/3` | Model defaults and an analytical geometry assumption. The optional airflow and spin settings accept values supplied by the caller. The task now selects dynamic Magnus; the retained zero fixed factor applies only in constant mode. |

## Update audit log

### Dynamic Magnus force (2026-11-06)

[Conti et al., arXiv:2606.28805v2](https://arxiv.org/html/2606.28805v2#S3.SS2.SSS2),
Eq. (5), Tables II-III, supplies the following C_M curves. Units: speed in
m/s, angular speed w in rad/s, dimensionless C_M; m1 and b multiply w,
a multiplies w². Source table values are transcribed without a new fit into
[magnus_model.py](../legged_lab/physics/magnus_model.py).

| Speed | m1 | s | w_break | a | b | c |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 0 | 0.08 | 150 | -1.852e-7 | -1.296e-4 | 0.0983 |
| 3.5 | -1.1e-3 | 0.31 | 200 | -1.667e-7 | -3.333e-5 | 0.1 |
| 7.5 | -8.0e-4 | 0.37 | 350 | -2.000e-7 | 1.700e-4 | 0.0587 |
| 10.5 | -6.58e-4 | 0.375 | 440 | -2.604e-7 | 3.646e-4 | -0.0225 |
| 13.5 | -5.6e-4 | 0.383 | 550 | -3.571e-7 | 5.357e-4 | -0.0893 |
| 17.0 | -4.48e-4 | 0.371 | 650 | -1.000e-7 | 2.300e-4 | -0.0375 |

At each tabulated speed use `max(0, m1*w + s)` when `w <= w_break`,
otherwise `max(0, a*w² + b*w + c)`. PACE linearly interpolates the resulting
C_M values between neighboring speed curves. This is an explicit PACE
interpolation choice, not a claim to reproduce the authors' software.
The table uses a reference viscosity of 1.506e-5 m²/s and nominal radius
0.020 m in the source's Reynolds-number interpretation. PACE uses the listed
speed nodes directly and does not vary viscosity with temperature.

The source defines `F_M = rho * (4*pi*r³/3) * C_M * (omega x v)`.
The existing PACE force is `0.5*rho*pi*r³*kappa*(omega x v_rel)`;
equating them yields `kappa = (8/3)*C_M`. Lookup uses current relative speed
and spin magnitude. The source's force-averaged segment-speed fit is treated
as an instantaneous, quasi-steady model in PACE; this transfer is an assumption.
At 7.5 m/s and 200 rad/s, C_M = 0.21 and kappa = 0.56. For transverse
spin and current PACE density, the force magnitude is 0.01293079536 N.

Lookup speed clamps to the tabulated 2-17 m/s range. The source describes
play up to 1000 rad/s; PACE additionally clamps lookup spin at that value
as an engineering guard, not a new empirical law. The force still uses actual
vectors, so behavior outside the lookup envelope is not validated. Rounded
piecewise coefficients can yield jumps at spin breakpoints; these are preserved.
The fit is for Nittaku Nexcel 40+ balls in the source study, not a universal
ITTF coefficient or a calibration for the exact PACE ball.

No raw flight dataset was imported, filtered or refitted for this update;
there is no new project train/test split for this model. Published fits are
used as a reference. The existing racket dataset and its splits are unchanged.

The task selects `magnus_model="conti2026"`. The old `magnus_factor=0.0`
is retained for the selectable `"constant"` mode and remains the standalone
adapter default. The live force is recomputed at every physics substep
from the current state, and PhysX applies its own time integration. No dt
factor is applied to force values and no elapsed-time multiplier is fitted.
Air density 1.225 kg/m³, C_D 0.4378, mass 0.0027 kg, radius 0.020 m,
initial spin zero, material values, inertia and timing retain their prior
provenance and values. Aerodynamic torque is still zero.

CPU checks cover published node values, both spin branches, interpolation,
lookup bounds, units, sign/zero limits, relative airflow, changing-state
updates with fixed batch size, and the actual environment substep loop
executed with stand-in physics. Real PhysX contact/flight, high-spin transfer
and the fit's physical accuracy for PACE remain unverified. The existing
analytical approach-to-table predictor omitted Magnus at that revision.
The subsequent rebound-spin and approach-flight updates below connect
the same Magnus law to both prediction stages.

### Explicit live ball inertia (2026-11-06)

The [ball spawner](../legged_lab/assets/table_tennis/ball_spawner.py), selected
by [BALL_CFG](../legged_lab/assets/table_tennis/ball.py), now writes explicit
USD mass-distribution attributes at spawn time. With the unchanged nominal
mass `m = 0.0027 kg` and radius `r = 0.020 m` from the ITTF entries above,
each principal inertia is `(2/3) * m * r² = 7.2e-7 kg m²`. The factor `2/3`
is the uniform thin spherical shell assumption already used in
`resolve_ball_impact`, obtained by integrating squared distance to a diameter
over the shell. Center of mass is `(0, 0, 0) m` in the body frame and principal
axes use the identity quaternion. These are derived geometry assumptions,
not ITTF-specified inertia values or new experimental fits. A uniform solid
sphere with the same mass/radius would have `4.32e-7 kg m²`; explicitly
authoring the shell value removes reliance on collider-based mass inference.

Implementation references checked against the project's Isaac Lab 2.1.0:

- [MassPropertiesCfg at v2.1.0](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py)
  exposes mass and density, so inertia is authored using USD MassAPI.
- [Sphere spawning at v2.1.0](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/sim/spawners/shapes/shapes.py)
  places mass/rigid-body APIs on the root and the sphere collider below it.
- [Clone decorator at v2.1.0](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/sim/utils.py)
  calls the wrapped function for the first concrete path before cloning.
  The custom spawner writes inertia within that function, before copies exist.
- [OpenUSD MassAPI, 25.11 reference](https://openusd.org/25.11/api/class_usd_physics_mass_a_p_i.html)
  defines nonzero diagonal inertia in stage mass × distance² units, local
  center of mass and principal-axis orientation. This change assumes PACE's
  meter/kilogram stage and unscaled asset; it does not introduce a USD version
  requirement or rescale assets.

CPU checks cover the nominal value, mass/radius scaling, invalid inputs,
root authoring, USD copying/inheritance and consistency with the offline
angular impulse. The sphere spawner is stubbed in these tests; PhysX inertia
readback, contact response and transfer of the existing racket fit to the
live simulation remain unverified. No new data, dataset split, fitted value,
or dependency requirement was added. Ball mass/radius, aerodynamic inputs,
contact materials and the offline impulse formula retain their existing
provenance entries and values.

### Analytical table rebound update (2026-11-06)

The [table prediction helper](../legged_lab/physics/table_prediction.py) now
uses `v_impact = v_now / (1 + k * abs(v_now) * t)` for each horizontal
component before applying either the default rebound or the optional
spin-aware impact. This formula is derived by integrating the existing
per-axis equation `dv/dt = -k * abs(v) * v`. Here `v` is in m/s, `t` is in
seconds, and `k = rho * pi * r^2 * Cd / (2m)` is in inverse meters.

The active inputs remain radius 0.020 m, mass 0.0027 kg, air density
1.225 kg/m³ and `Cd = 0.4378`, with the standards and inheritance recorded
above. The existing horizontal rebound multiplier 0.70 is unchanged.
Normal restitution now reads `bounce_restitution` in both branches, retaining
the default 0.90. Neither coefficient was fitted again, and no external
dataset or published constant was added by this update.

For the representative launch `(1.35, 0, 1.03) m` at
`(-5.75, 0, 1.75) m/s`, the existing ballistic contact-time estimate is
0.466124 s. Predicted rebound X velocity changes from -4.025 to
-3.016109 m/s. The predicted rebound apex X changes from -1.917866 to
-1.688787 m when only this approach-speed inconsistency is corrected.
These are analytical comparisons, not real-world accuracy measurements.

CPU regression tests compare horizontal velocities with independent RK4
integration of the same per-axis drag ODE, exercise zero drag and zero
flight time, and check the optional spin-aware path, batched shapes and
non-mutating inputs. The helper has no time-stepping loop or GPU-to-CPU
scalar readback. At that revision, the ballistic vertical approach and
separate-axis flight approximation remained; these are now retained only
in the Magnus-disabled fallback. Physical rebound and high-spin flight
still need separate validation.

### Rebound spin propagation and ascent (2026-11-06)

At base `4d792ed63f83e38a6139cbcdfdb3d0ef83a15fc4`, reviewed
`table_prediction.py`, `ball_impact.py`, `aerodynamics.py` and the task's
prediction caller. The state API now retains the existing impulse model's
angular result rather than discarding it. Its relation
`delta_omega = (-r*n) cross delta_v_t / ((2/3)*r^2)` follows from angular
impulse and the previously documented uniform thin-shell inertia. This is
a derived relation using existing inputs, with no new empirical coefficient.
The legacy `bounce_friction=None` branch preserves incoming spin while
keeping the original 0.70 horizontal multiplier; because it has no angular
impulse law, this is explicitly a compatibility assumption. It does not
predict friction-generated spin. The optional friction branch does.

`flight_prediction.py` consumes the returned contact spin, or measured spin
for a ball already in flight, and uses the same force adapter and Conti
speed/spin curves documented above. It recomputes force at both stages of
each explicit-midpoint step. Spin stays constant between contacts under the
existing zero-aerodynamic-torque assumption. Inputs are world-frame meters,
m/s, rad/s, kg, and gravity in m/s². Numerical settings are a maximum
0.020 s step and 1.50 s horizon (75 iterations); the apex event is located
within its crossing step. These are implementation choices, not external
measurements. Unresolved/nonfinite apex samples are invalid, and the task
uses its existing fallback. Magnus-disabled ascent retains the old formula.

CPU checks cover angular momentum, unchanged legacy linear outputs, retained
spin, generated spin, ascent/side-force signs, batch shapes, input ownership,
invalid/horizon handling, and actual environment prediction routing without
Isaac imports. An independent RK4 integrator with a 0.0005 s step checks
representative flight cases to 3 mm position and 1 ms time tolerances under
the same force law. This checks numerical consistency, not measured flight
accuracy. At that revision, landing before the bounce still used ballistic
vertical motion and separate-axis drag without Magnus; the update below
addresses that mismatch. Real PhysX contacts, training throughput, high-spin
regimes and the horizon's coverage remain unverified.

No physical parameter, source dataset, split/filter, external dependency or
coefficient fit was added or changed. Reviewed radius 0.020 m, mass 0.0027 kg,
air density 1.225 kg/m³, `Cd=0.4378`, gravity 9.81 m/s², table restitution
0.90, legacy multiplier 0.70, existing friction settings, thin-shell inertia
and the dynamic Magnus curves; their recorded source entries remain unchanged.

### Magnus-aware approach to the table (2026-11-09)

At base `64787794a862aba3a67f913e6f6db93c18c5a5fd`, reviewed
`flight_prediction.py`, `table_prediction.py`, the task caller and the
existing aerodynamic inputs. The Magnus-enabled task now follows the ball
to the contact-height plane with the same force adapter as live flight:

`dp/dt = v`, `dv/dt = g_vector + (F_D(v-u) + F_M(v-u, omega))/m`.

The existing speed/spin-dependent coefficient is recomputed at both stages
of each explicit-midpoint step. Spin remains constant under the existing
zero-torque assumption. The numerical incident velocity is passed to the
collision helper at zero remaining approach time, so approach drag/gravity
are not applied twice. Returned rebound spin still feeds ascent prediction.

The maximum step (0.020 s) and horizon (1.50 s) are inherited numerical
settings, not measured physical parameters. Contact is located inside the
crossing step from `z(t) = z0 + vz0*t + 0.5*az0*t^2`; the downward root is
used and velocity is interpolated from the midpoint step. This is a derived
numerical event calculation with no fitted coefficient. Contact height is
the existing 0.760 m table height plus ball radius (0.780 m nominal), in the
task's environment-relative coordinates. The solver assumes an infinite
horizontal plane and does not resolve net or table-edge collisions.

At-plane incoming/resting states contact at zero time; upward states are
followed to their return. Below-plane/nonfinite states and unresolved
crossings are invalid. The task records `valid_land` and rejects invalid
pre-bounce targets; current post-bounce ascent has independent validity.
The Magnus-disabled compatibility path keeps the old analytical formulas.

CPU checks cover an exact vacuum solution, incoming/upward/invalid boundary
states, horizon exhaustion, moving-air consistency, float32/float64 batches,
spin directions, and the real environment prediction block without Isaac.
An independent RK4 solver (0.0005 s step) checks representative contact
positions, velocities and times to tolerances of 0.003 m, 0.005 m/s and
0.001 s respectively. Existing ascent and collision tests also pass.

For the nominal initial state `(1.35, 0, 1.03) m`, `(-5.75, 0, 1.75) m/s`,
the new solver gives the following comparisons with fixed world-frame spin:

| Angular velocity (rad/s) | Contact time (s) | Contact X (m) | Contact Y (m) |
| --- | ---: | ---: | ---: |
| `(0, 0, 0)` | 0.459824 | -0.928580 | 0.000000 |
| `(0, -100, 0)` (topspin for this direction) | 0.399668 | -0.667580 | 0.000000 |
| `(0, 100, 0)` (backspin for this direction) | 0.541103 | -1.261952 | 0.000000 |
| `(0, 0, 100)` | 0.459824 | -0.918715 | -0.182199 |

These are numerical model outputs, not measured trajectories. Physical
parameters, source versions, data splits/filters and fits are unchanged:
radius 0.020 m, mass 0.0027 kg, density 1.225 kg/m³, `Cd=0.4378`, gravity
9.81 m/s², table restitution 0.90, legacy horizontal multiplier 0.70,
contact friction settings, thin-shell inertia and Conti speed/spin curves
retain their existing provenance. No external data or dependency was added.
Real PhysX agreement, physical accuracy, high-spin/horizon coverage and
training throughput remain unverified; no GPU simulation was run.

| Date | Code or change reviewed | Provenance check |
| --- | --- | --- |
| 2026-11-04 | Baseline `77e8017`, covering [PR #2](https://github.com/ohowo/PACE-ICRA2026/pull/2), [PR #3](https://github.com/ohowo/PACE-ICRA2026/pull/3), [PR #4](https://github.com/ohowo/PACE-ICRA2026/pull/4) and [PR #5](https://github.com/ohowo/PACE-ICRA2026/pull/5) | Created the register, checked the four dataset blobs at the pinned revision and reran the coefficient fit and test-set evaluation. Recorded the standards, inherited values and remaining validation gaps. This change updates the documentation and maintenance rules. |
| 2026-11-06 | Table rebound prediction at base `caf7ade` | Reviewed ball, air and contact inputs. No new external data or coefficient fit. Recorded the derived approach-speed formula, inherited rebound factors, CPU checks and remaining approximations. |
| 2026-11-06 | Live ball inertia at base `5f06748` | Applied the existing thin-shell assumption to the live asset, checked Isaac Lab 2.1.0 spawning/cloning and USD mass-attribute semantics, and recorded derived SI inertia plus CPU/runtime validation limits. Existing physical inputs and their source entries are unchanged. |
| 2026-11-06 | Dynamic Magnus force at base `3cbce09` | Recorded versioned speed/spin curves, coefficient conversion, interpolation and lookup guards, per-substep updating, CPU evidence and predictor/runtime limitations. Other physical parameters retain their recorded sources and values. |
| 2026-11-06 | Rebound spin and ascent at base `4d792ed` | Retained post-impact angular velocity and routed it into the existing Magnus law. Recorded legacy spin preservation, derived angular impulse, numerical step/horizon, CPU checks and remaining approach-flight/runtime limits. No new external data or physical parameter changes. |
| 2026-11-09 | Magnus-aware table approach at base `6478779` | Applied the existing force law through contact, retained numerical incident velocity for rebound, recorded contact-event derivation and validity handling, and checked CPU numerical consistency. No new external input, coefficient fit or physical parameter change. |
