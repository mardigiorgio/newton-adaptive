# SPDX-License-Identifier: Apache-2.0
"""Derive SAP per-shape contact parameters from the LBM/Drake asset properties.

Closed form only: no GPU, no solver, no Warp, no USD. Every number printed here
is a function of (a) values read out of the LBM SDF ``drake:proximity_properties``,
(b) collision-hull facet geometry measured from the task USD, and (c) the SAP
contact law as implemented in ``sap_warp/sim/``.

Run::

    python tools/probes/sap_contact_parameter_derivation.py

SAP contact law being reproduced (sap_warp/sim/contact_solve.py, the
``_sap_contact_*`` kernels; identical to Drake's
``sap_friction_cone_constraint.cc``)::

    rn_hard = beta^2 / (4 pi^2) * w_eff
    rn_soft = 1 / (h * k_pair * (h + tau_pair))
    rn      = max(rn_hard, rn_soft)
    vhat_n  = -phi0 / (h + tau_pair)
    gamma_n = (1/rn) * (vhat_n - v_n)_+

At steady state (v_n = 0, penetration x = -phi0) the normal force f = gamma_n/h
reduces to f = k_eff * x with

    k_eff = min(k_pair, k_cross),   k_cross = 1 / (rn_hard * h * (h + tau_pair))

so k_cross is a hard CEILING on the stiffness the solver can realize, whatever
stiffness is authored.

Combination rules, verbatim from sap_warp/sim/sap_helpers.py -- and each one is
identical to Drake's own rule for the same quantity:

    _sap_combine_stiffness(k0,k1) = k0*k1/(k0+k1)       series
        == Drake discrete_update_manager.cc:940  g = 1/(1/gM + 1/gN)
    _sap_combine_mu(mu0,mu1)      = 2*mu0*mu1/(mu0+mu1) harmonic mean
    _contact_tau_pair(t0,t1)      = t0 + t1             sum
        == Drake contact_properties.cc:153 GetCombinedDissipationTimeConstant
"""

from __future__ import annotations

import math

PI = math.pi

# --------------------------------------------------------------------------- #
# Inputs, each with its provenance.
# --------------------------------------------------------------------------- #

# SAP constants: sap_warp/sim/solver_sap.py ctor defaults, echoed in the
# pass-25 near-rigid dump's "config" block.
BETA, SIGMA = 1.0, 1.0e-3
BETA_FACTOR = BETA * BETA / (4.0 * PI * PI)

# Timestep: trossen_spatula_lift_env_cfg sets sim.dt = 1/120 with num_substeps 2
# on the fixed arm (1 on the adaptive arm, which then subdivides).
H_FIXED, H_OUTER = 1.0 / 240.0, 1.0 / 120.0

# Current authored state: isaaclab_newton NewtonShapeCfg defaults (ke, mu) and
# MJWarpSolverCfg.sap_contact_tau_d (a PER-SHAPE fallback).
KE_SHAPE_NOW, MU_SHAPE_NOW, TAU_SHAPE_NOW = 2.5e3, 1.0, 0.01

# LBM/Drake validated properties, from the SDF drake:proximity_properties.
E_HYDRO = 1.0e8  # [Pa]   drake:hydroelastic_modulus (mug, spatula, plate)
D_HC = 40.0  # [s/m]  drake:hunt_crossley_dissipation (same three)
MU_LBM = 0.2  # drake:mu_static == drake:mu_dynamic (mug, spatula)

# Mug rigid-body properties: SDF <inertial>, mirrored into the task USD.
MUG_MASS = 0.0181  # [kg]
MUG_COM = (0.0017863, 0.0, 0.045564)  # [m] body frame (origin = base centre)
MUG_I = (2.3341e-05, 2.5136e-05, 2.1762e-05)  # [kg m^2] principal, about COM
G = 9.81

# Gripper carriage drive. Vendor USD (stationary_ai.usd,
# follower_left_left_carriage_joint drive:linear:physics:stiffness) vs the task
# override in trossen_spatula_lift/assets.py.
K_DRIVE_VENDOR = 217687.0  # [N/m]
K_DRIVE_NOW = 1000.0  # [N/m]
CARRIAGE_ARMATURE = 0.1  # [kg] task-authored; ADDED to the SAP mass-matrix
CARRIAGE_EQUIV_INERTIA = 9.820041e-4  # [kg] vendor physics:JointEquivalentInertia

# Contact patches: (facet area [m^2], foundation depth H [m]) measured on the
# actual convex hulls in assets/usd/mug_inomata_white.usd. H is Drake's
# make_convex_field construction -- pressure E at the hull's single interior
# vertex, 0 on the boundary, so the gradient into a facet is E / dist(interior
# vertex, facet plane). The facet listed is the one that actually contacts.
PATCHES = {
    #  shape                          A [m^2]    H [m]     contacts
    "/Mug/collisions_base": (2.068e-3, 5.16e-3, "table (-Z facet)"),
    "/Mug/collisions_wall_[0-7]": (1.106e-3, 2.54e-3, "table/arm (side facet)"),
    "/Mug/collisions_handle_[0-2]": (1.13e-4, 2.92e-3, "gripper pad (+-X facet)"),
}
# What we author on the shapes with no validated compliance (steel carriage and
# gripper bodies, arm links, the TableGuard slab, the ground plane). Drake sets
# a rigid body's gradient to +inf so the series rule collapses to the compliant
# side; sap_warp rejects non-finite per-shape ke, so use a large finite stand-in.
K_RIGID = 1.0e9

# Contacts the pipeline emits per shape pair. NOT measured this pass (GPU).
N_PER_PAIR = 4

# Reference dump from a near-rigid pass. Used ONLY to check the formulas below
# reproduce a measurement; never as an input to a result.
W_EFF_MEDIAN_MEASURED = 14.917052269
RN_RATIO_MEDIAN_MEASURED = 0.1115
FRAC_NEAR_RIGID_MEASURED = 0.1109


# --------------------------------------------------------------------------- #
def combine_stiffness(k0, k1):
    return k0 * k1 / (k0 + k1)


def combine_mu(m0, m1):
    return 2.0 * m0 * m1 / (m0 + m1)


def shape_ke_for_pair(k_pair, k_partner):
    """Invert the series rule: what must this shape carry to reach k_pair?"""
    return float("inf") if k_partner <= k_pair else k_pair * k_partner / (k_partner - k_pair)


def w_free_body(r, mass, inertia):
    """(1/3) tr(J M^-1 J^T) at offset r from a free body's COM.

    diag(W)_ab = delta_ab/m + (r x e_a)^T Ic^-1 (r x e_b); SAP's contact weight
    is the mean of the three diagonal entries (contact_jacobian.py:
    ``w_eff = (wt1 + wt2 + wn)/3``).
    """
    rx, ry, rz = r
    i1, i2, i3 = inertia
    rot = (ry * ry + rz * rz) / i1 + (rx * rx + rz * rz) / i2 + (rx * rx + ry * ry) / i3
    return 1.0 / mass + rot / 3.0


def k_cross(w_eff, h, tau_pair):
    return 1.0 / (BETA_FACTOR * w_eff * h * (h + tau_pair))


def hdr(n, s):
    print(f"\n{'=' * 78}\n{n}. {s}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
hdr(0, "CROSS-CHECK the implemented law against the pass-25 measured dump")
k_pair_now = combine_stiffness(KE_SHAPE_NOW, KE_SHAPE_NOW)
tau_pair_now = TAU_SHAPE_NOW + TAU_SHAPE_NOW
print(f"  pair k   = {k_pair_now:.1f} N/m   (dump contact_k_unique:      [1250.0])")
print(f"  pair tau = {tau_pair_now:.4f} s   (dump contact_tau_d_unique:  [0.02])")
print(f"  pair mu  = {combine_mu(MU_SHAPE_NOW, MU_SHAPE_NOW):.3f}")
ratio = (BETA_FACTOR * W_EFF_MEDIAN_MEASURED) / (1.0 / (H_OUTER * k_pair_now * (H_OUTER + tau_pair_now)))
print(f"  rn_hard/rn_soft at h=1/120, w=median: {ratio:.5f}  (dump: {RN_RATIO_MEDIAN_MEASURED})")
print("  -> the law, the constants and the pair values all reproduce the dump.")
for h, lbl in ((H_OUTER, "1/120"), (H_FIXED, "1/240")):
    print(f"  contacts flip near-rigid above w_eff = "
          f"{1.0 / (BETA_FACTOR * h * k_pair_now * (h + tau_pair_now)):7.1f} /kg at h={lbl}")

# --------------------------------------------------------------------------- #
hdr(1, "w_eff for the pairs that matter (closed form from the mug's own inertia)")
cx, cy, cz = MUG_COM
sites = {
    "base rim": (0.0331 - cx, -cy, 0.0001 - cz),
    "base under COM": (-cx + 0.0017863, -cy, 0.0001 - cz),
    "handle strut": (0.0640 - cx, -cy, 0.0580 - cz),
    "wall side": (0.0386 - cx, -cy, 0.0500 - cz),
}
w_mug = {k: w_free_body(r, MUG_MASS, MUG_I) for k, r in sites.items()}
for k, v in w_mug.items():
    print(f"  mug contact at {k:16s} -> w_mug = {v:7.1f} /kg")
W_PAD = 1.0 / (CARRIAGE_ARMATURE + CARRIAGE_EQUIV_INERTIA)
print(f"\n  carriage pad along closing axis: w <= 1/(armature+I_eq) = {W_PAD:.1f} /kg")
print(f"    (armature IS summed into the SAP mass matrix: free_motion.py:1428.")
print(f"     WITHOUT the task's armature=0.1 the vendor inertia alone would give")
print(f"     w = {1.0 / CARRIAGE_EQUIV_INERTIA:.0f} /kg -- a {1.0 / CARRIAGE_EQUIV_INERTIA / W_PAD:.0f}x swing. Sensitivity flagged.)")
print("  static TableGuard / ground plane: w = 0 (no DOFs)")

W_PAIR = {
    "mug_base - table": w_mug["base rim"],
    "mug_wall - table": w_mug["wall side"],
    "mug_handle - pad": w_mug["handle strut"] + W_PAD,
    "mug_wall - pad": w_mug["wall side"] + W_PAD,
}
print()
for k, v in W_PAIR.items():
    print(f"  pair w_eff  {k:20s} = {v:7.1f} /kg  "
          f"({v / W_EFF_MEDIAN_MEASURED:5.1f}x the scene median {W_EFF_MEDIAN_MEASURED:.2f})")
print(f"\n  That {FRAC_NEAR_RIGID_MEASURED:.0%} of contacts already measure near-rigid is consistent:")
print("  the high-w_eff tail IS the mug's own contacts.")

# --------------------------------------------------------------------------- #
hdr(2, "STIFFNESS: hydroelastic modulus -> per-shape ke")
print("  Drake, discrete_update_manager.cc:1025   k = A_e * g   [N/m]")
print("        A_e = contact-surface face area, g = normal pressure gradient,")
print("        g = 1/(1/gM + 1/gN)  (series; rigid side is +inf, so g -> compliant side)")
print("  Pressure field: p = E * phi/H  (hydroelastic_parameters_doxygen.h eq.1)")
print("  H for a convex hull (make_convex_field.h): E at the single interior")
print("  vertex, 0 on the boundary => g into a facet = E / dist(vertex, facet).")
print()
print(f"  {'shape':30s} {'A[mm^2]':>8s} {'H[mm]':>6s} {'g[Pa/m]':>10s} {'k_pair':>10s} {'ke author':>10s}")
targets = {}
for name, (area, hh, what) in PATCHES.items():
    g = E_HYDRO / hh
    kp = area * g
    ke = shape_ke_for_pair(kp, K_RIGID)
    targets[name] = (kp, ke)
    print(f"  {name:30s} {area * 1e6:8.1f} {hh * 1e3:6.2f} {g:10.3e} {kp:10.3e} {ke:10.3e}")
print(f"\n  rigid partners (TableGuard, carriage/gripper, links, ground): ke = {K_RIGID:.0e} N/m")
worst = max(kp for kp, _ in targets.values())
print(f"  -> series error vs a true rigid partner: {100 * worst / K_RIGID:.1f}% on the stiffest piece.")
print("\n  Sensitivity: k_pair is EXACTLY linear in A and in 1/H.")
print("    2x patch area -> 2x k_pair;  0.5x -> 0.5x.  Same for 1/H.")
print(f"  If the pipeline emits N={N_PER_PAIR} contacts across one facet, the faithful")
print("  per-point value is k_pair/N (Drake's quadrature is one point per face):")
for name, (kp, _) in targets.items():
    print(f"    {name:30s} k/N = {kp / N_PER_PAIR:.3e} N/m")

# --------------------------------------------------------------------------- #
hdr(3, "Does the authored ke ever bind? (the near-rigid ceiling)")
TAUS = {
    "current 2 x 0.01": 0.02,
    "Castro/CENIC (beta/pi)*h": (BETA / PI) * H_FIXED,
    "energy-match (sec.4)": 2.42e-3,
    "zero": 0.0,
}
for pair, kp_name in (("mug_base - table", "/Mug/collisions_base"),
                      ("mug_handle - pad", "/Mug/collisions_handle_[0-2]")):
    kp = targets[kp_name][0]
    w = W_PAIR[pair]
    print(f"\n  {pair}:  derived pair k = {kp:.3e} N/m, w_eff = {w:.1f} /kg, h = 1/240 s")
    print(f"    {'tau_pair':30s} {'k_cross':>10s} {'k_eff realized':>15s} {'vs now':>8s}")
    for lbl, tp in TAUS.items():
        kc = k_cross(w, H_FIXED, tp)
        print(f"    {lbl + f'  = {tp * 1e3:6.2f} ms':30s} {kc:10.4g} {min(kp, kc):15.4g} "
              f"{min(kp, kc) / 1250.0:7.2f}x")
    tp = (BETA / PI) * H_FIXED
    h_bind = (-tp + math.sqrt(tp * tp + 4.0 / (BETA_FACTOR * w * kp))) / 2.0
    print(f"    -> the authored ke only stops being clamped below h = {h_bind * 1e6:.1f} us "
          f"({1 / h_bind:.0f} Hz).")
    print(f"       Smallest dt this campaign has observed: 2.24 ms. So the authored ke is")
    print(f"       INERT at every timestep the campaign runs; tau_d and h set the behaviour.")

# --------------------------------------------------------------------------- #
hdr(4, "DISSIPATION: Hunt & Crossley d=40 s/m -> SAP tau_d")
print("  Drake sanctions NO conversion: multibody_plant.h:806 says exactly one of")
print("  hunt_crossley_dissipation / relaxation_time is used, per approximation;")
print("  neither Castro 2021 (SAP) nor Castro 2023 (ICF) gives a formula. Castro")
print("  2023's own empirical pairings have d/tau_d = 5e5, 1e5, 5e4 -- not constant.")
print("  So this is OUR mapping, matched at a stated operating point, three ways.")
print()
print("  (A) match the DAMPING TERM:")
print("      H&C   f = k x (1 + d xdot)_+   -> damping force  k*d*x*xdot")
print("      SAP   f = k (x + tau xdot)_+   -> damping force  k*tau*xdot")
print("      equal at penetration x0  =>  tau = d * x0.   [per shape; pair sums]")
print(f"      {'x0':>12s} {'tau_shape':>12s} {'tau_pair':>12s}")
for x0 in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3):
    print(f"      {x0 * 1e6:9.1f} um {D_HC * x0 * 1e3:9.4f} ms {2 * D_HC * x0 * 1e3:9.4f} ms")
print("      external check: Castro 2023 pairs (k=1e7, d=500) with tau_d=1e-3, i.e.")
print(f"      x0 = tau/d = {1e-3 / 500 * 1e6:.1f} um, and (d=10, tau_d=1e-4) -> x0 = "
      f"{1e-4 / 10 * 1e6:.1f} um. Both are")
print("      physically sane operating penetrations, so the rule reproduces their")
print("      hand-tuned pairs. It is a real mapping, not a dimensional coincidence.")

print("\n  Self-consistent fixed point (near-rigid branch, tau=d*x and x=f/(N k_cross)):")
for pair, n, f_tot in (("mug_base - table", N_PER_PAIR, MUG_MASS * G),):
    w = W_PAIR[pair]
    rn_h = BETA_FACTOR * w
    f_i = f_tot / n
    C = D_HC * f_i * rn_h * H_FIXED
    tau_fp = C * H_FIXED / (1.0 - C) if C < 1.0 else float("inf")
    x_fp = f_i * rn_h * H_FIXED * (H_FIXED + tau_fp)
    print(f"    {pair}: f={f_tot:.4f} N over N={n} -> tau_pair={tau_fp * 1e3:.3f} ms, "
          f"x={x_fp * 1e6:.2f} um")

print("\n  (B) match ENERGY per impact (critical damping at the REALIZED stiffness):")
print("      zeta = tau*omega_n/2, omega_n = sqrt(k_eff/m); zeta=1 -> tau=2 sqrt(m/k_eff)")
for lbl, kk in (("realized k_cross at tau=1.33ms",
                 k_cross(W_PAIR["mug_base - table"], H_FIXED, (BETA / PI) * H_FIXED)),
                ("the authored 4.0e7", targets["/Mug/collisions_base"][0])):
    print(f"      mug on {lbl:32s} k={kk:10.4g} -> tau_pair = "
          f"{2 * math.sqrt(MUG_MASS / kk) * 1e3:7.3f} ms")
kk = k_cross(W_PAIR["mug_base - table"], H_FIXED, (BETA / PI) * H_FIXED)
om = math.sqrt(kk / MUG_MASS)
for tp in (0.02, (BETA / PI) * H_FIXED, 2.42e-3):
    z = tp * om / 2.0
    e = 0.0 if z >= 1.0 else math.exp(-z * PI / math.sqrt(1 - z * z))
    print(f"      tau_pair={tp * 1e3:6.2f} ms -> zeta={z:6.3f}, restitution e={e:.3f}")
print(f"      H&C with d={D_HC:.0f} s/m is fully plastic above v = 1/d = "
      f"{1 / D_HC * 1e2:.1f} cm/s, so e~0 is the target.")

print("\n  (C) Castro/CENIC near-rigid prescription: tau = (beta/pi)*h = "
      f"{(BETA / PI) * H_FIXED * 1e3:.3f} ms per PAIR")
print("\n  VERDICT: the three routes give tau_pair in [0.1, 2.4] ms. They agree to")
print("  within an order of magnitude and all sit ~1 ms. The current 20 ms is")
print(f"  {0.02 / ((BETA / PI) * H_FIXED):.0f}x longer than any of them.")
print(f"  Recommend tau_pair = {(BETA / PI) * H_FIXED * 1e3:.2f} ms, i.e. sap_contact_tau_d = "
      f"{(BETA / PI) * H_FIXED / 2:.2e} s per shape.")
print("  Sensitivity: route (A) moves linearly with the assumed operating")
print("  penetration -- a 10x error in x0 is a 10x error in tau. Routes (B) and")
print("  (C) do not depend on x0 at all, which is why they bound (A).")

# --------------------------------------------------------------------------- #
hdr(5, "FRICTION")
print("  _sap_combine_mu is the HARMONIC MEAN, so equal shapes reproduce their own")
print("  value (unlike stiffness, whose series rule HALVES equal shapes).")
print(f"    now      mu 1.0 & 1.0 -> pair {combine_mu(1.0, 1.0):.3f}")
print(f"    derived  mu 0.2 & 0.2 -> pair {combine_mu(0.2, 0.2):.3f}   "
      f"({combine_mu(1.0, 1.0) / combine_mu(0.2, 0.2):.0f}x error today)")
print(f"    if the pads are elastomer (1.0): 0.2 & 1.0 -> pair {combine_mu(0.2, 1.0):.3f}")
print(f"    LBM stainless partner (0.15):    0.2 & 0.15 -> pair {combine_mu(0.2, 0.15):.3f}")
n_now = MUG_MASS * G / (2.0 * combine_mu(1.0, 1.0))
n_new = MUG_MASS * G / (2.0 * combine_mu(MU_LBM, MU_LBM))
print(f"\n  Two-finger pinch normal force needed to hold the mug against gravity:")
print(f"    pair mu 1.0 -> {n_now:.4f} N;   pair mu 0.2 -> {n_new:.4f} N  "
      f"({n_new / n_now:.0f}x more squeeze)")
print(f"  Lateral force to slide the mug on the table: {1.0 * MUG_MASS * G:.4f} N now vs "
      f"{MU_LBM * MUG_MASS * G:.4f} N derived.")
print(f"  Tip-vs-slide: base radius 33.1 mm, COM height {MUG_COM[2] * 1e3:.1f} mm -> the mug")
print(f"  tips instead of sliding when mu > r/z = {0.0331 / MUG_COM[2]:.3f}. At mu 1.0 it")
print("  ALWAYS tips; at 0.2 it ALWAYS slides. That is the 'sticky' signature.")

# --------------------------------------------------------------------------- #
hdr(6, "CONSEQUENCE: gripper embed and the drive stiffness")
print("  Finger with drive stiffness k_d commanded delta past the surface, resisted")
print("  by contact k_c:   x = delta*k_d/(k_d+k_c),   grip f = k_c*x.")
DELTA = 0.010
w_pinch = W_PAIR["mug_handle - pad"]
kc_now = 1250.0
kc_20 = k_cross(w_pinch, H_FIXED, 0.02)
kc_13 = k_cross(w_pinch, H_FIXED, (BETA / PI) * H_FIXED)
kc_0 = k_cross(w_pinch, H_FIXED, 0.0)
print(f"\n  delta = {DELTA * 1e3:.0f} mm, pinch pair w_eff = {w_pinch:.1f} /kg, h = 1/240 s")
print(f"  {'case':>34s} {'k_c':>10s} {'k_d':>8s} {'embed':>7s} {'x[mm]':>7s} {'f[N]':>8s} {'margin':>7s}")
for lbl, kc, kd in [
    ("now (soft branch, k=1250)", kc_now, K_DRIVE_NOW),
    ("derived ke, tau_pair 20 ms", kc_20, K_DRIVE_NOW),
    ("derived ke, tau_pair 1.33 ms", kc_13, K_DRIVE_NOW),
    ("derived ke, tau_pair 0", kc_0, K_DRIVE_NOW),
    ("derived + tau 1.33ms, k_d 200", kc_13, 200.0),
    ("derived + tau 1.33ms, k_d 100", kc_13, 100.0),
    ("derived + tau 1.33ms, VENDOR k_d", kc_13, K_DRIVE_VENDOR),
]:
    frac = kd / (kd + kc)
    x = DELTA * frac
    f = kc * x
    print(f"  {lbl:>34s} {kc:10.4g} {kd:8.0f} {frac:6.1%} {x * 1e3:7.3f} {f:8.3f} "
          f"{f / n_new:6.1f}x")
print(f"\n  margin = grip force / the {n_new:.3f} N required at the validated pair mu 0.2.")
print(f"  The 'now' row reproduces pass-35's MEASURED 44% embed independently:")
print(f"  k_d/(k_d+k_c) = 1000/2250 = {1000 / 2250:.3f}. That also proves today's pinch")
print("  sits on the SOFT branch (it uses k=1250, not k_cross), as w_eff predicts.")

print(f"\n  Ceiling: with tau -> 0 the realized pinch stiffness cannot exceed")
print(f"    1/(rn_hard*h^2) = {kc_0:.4g} N/m at h=1/240. k_cross scales as h^-2:")
for nsub in (2, 4, 8, 16, 32):
    h = 1.0 / (120.0 * nsub)
    kc = k_cross(w_pinch, h, (BETA / PI) * h)
    print(f"    num_substeps {nsub:2d}: h={h * 1e3:6.4f} ms -> k_cross={kc:10.4g} N/m, "
          f"embed at k_d=200: {200 / (200 + kc):6.2%}, at vendor k_d: "
          f"{K_DRIVE_VENDOR / (K_DRIVE_VENDOR + kc):6.2%}")
print(f"\n  To make the VENDOR drive stiffness ({K_DRIVE_VENDOR:.0f} N/m) embed <=5% you need")
print(f"  k_c >= {K_DRIVE_VENDOR * 19:.3g} N/m, i.e. h <= "
      f"{math.sqrt(1.0 / (BETA_FACTOR * w_pinch * K_DRIVE_VENDOR * 19)) * 1e6:.1f} us "
      f"({1 / math.sqrt(1.0 / (BETA_FACTOR * w_pinch * K_DRIVE_VENDOR * 19)):.0f} Hz).")
print("  That is out of reach. The vendor drive stiffness is UNUSABLE at this")
print("  substep no matter what contact stiffness is authored.")
