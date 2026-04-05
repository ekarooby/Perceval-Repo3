# ============================================================
# qsp_mse_vs_L.py
#
# PURPOSE:
#   Load saved QPU experimental and SLOS simulation results
#   for multiple L values and plot log(MSE) vs L,
#   reproducing Bu et al. (Phys. Rev. Applied 23, 034073, 2025)
#   Figure 3 style.
#
# KEY INSIGHT FROM BU ET AL. PAPER:
#   Their MSE formula (Eq. 3) compares W_L(x) vs f(x) where
#   f(x) is the TARGET FUNCTION. For STEP, since the circuit
#   is trained to approximate the surrogate arctan(Nx)*2/pi,
#   their "Theoretical MSE" is:
#       Perceval exact Z  vs  Surrogate (arctan, N=100)
#   Their "Experimental MSE" is:
#       QPU hardware Z    vs  Surrogate (arctan, N=100)
#
#   Crucially, their theoretical curve uses Perceval's
#   compute_unitary() -- the exact matrix result with ZERO
#   sampling noise. This is NOT recomputed from numpy; it is
#   loaded directly from Perceval for correctness.
#
# PLOTS PRODUCED:
#   Plot 1 -- Bu et al. style (2 curves, mirrors their Fig. 3):
#     Black solid  : Perceval analytic Z vs Surrogate (Theoretical)
#     Red squares  : Experimental QPU Z  vs Surrogate (Experimental)
#     Saved as: mse_vs_L_Bu_style_STEP_pq.png
#
#   Plot 2 -- Full comparison (all curves):
#     Black solid  : Perceval analytic vs Surrogate (Theoretical)
#     Grey dashed  : Perceval analytic vs True STEP
#     Blue solid   : SLOS vs True STEP
#     Blue squares : Exp vs True STEP
#     Red solid    : SLOS vs Surrogate
#     Red squares  : Exp vs Surrogate
#     Saved as: mse_vs_L_combined_STEP_pq.png
#
# FILE NAMING CONVENTION:
#   Loads files saved by qsp_experiment_qpu.py and
#   qsp_slos_simulation.py with FILE_TAG:
#     STEP_L{L}_N{N_SHOTS}_x{N_X}
#   Angle files: theta_step_pq_L{L}.npy, phi_step_pq_L{L}.npy
#
# HOW TO RUN:
#   1. Set L_VALUES, N_SHOTS_EXP, N_SHOTS_SLOS, N_X below.
#   2. All .npy files and angle files must be in RESULTS_DIR.
#   3. Run: python qsp_mse_vs_L.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import perceval as pcvl
import perceval.components as comp

# ============================================================
# SETTINGS
# ============================================================

FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"

L_VALUES = [5, 10, 15, 20, 24, 30, 60, 120, 180, 240, 300, 330, 360]

N_SHOTS_EXP  = 5000
N_SHOTS_SLOS = 5000
N_X          = 100

RESULTS_DIR = "."

# ============================================================
# Reference functions
# ============================================================

N_approx = 100   # must match what was used in QPU/SLOS code

def true_step(x):
    return np.where(x >= 0, 1.0, -1.0)

def surrogate_step(x):
    return (2.0 / np.pi) * np.arctan(N_approx * x)

# ============================================================
# Perceval circuit builder
# Identical to build_qsp_pic() in qsp_experiment_qpu.py
# ============================================================

def build_qsp_pic(theta_arr, phi_arr, x_val, L):
    circuit = pcvl.Circuit(2, name=f"QSP_{FUNC_NAME}_L{L}")
    circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))
    circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
    circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))
    for j in range(1, L + 1):
        circuit.add(0, comp.PS(float(-x_val / 2)))
        circuit.add(1, comp.PS(float( x_val / 2)))
        circuit.add(0,      comp.PS(float(-phi_arr[j] / 2)))
        circuit.add(1,      comp.PS(float( phi_arr[j] / 2)))
        circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[j])))
    return circuit

def perceval_analytic_Z(theta_arr, phi_arr, x_val, L):
    """
    Exact Z = p0 - p1 using Perceval compute_unitary().
    This is the true theoretical result -- same as what Bu et al.
    call 'classically simulated QSP circuits'. Zero sampling noise.
    """
    circuit = build_qsp_pic(theta_arr, phi_arr, x_val, L)
    U   = np.array(circuit.compute_unitary())
    psi = U @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Helper: load .npy file, warn if missing
# ============================================================

def load_npy(filepath, label):
    if not os.path.exists(filepath):
        print(f"  WARNING: {label} not found -- {filepath}")
        return None
    return np.load(filepath)

# ============================================================
# Main loop: collect MSE for each L
# ============================================================

FUNC_LOWER = FUNC_NAME.lower()

# Bu et al. style -- Perceval analytic (exact, zero noise)
mse_perceval_vs_surrogate = {}   # <-- their "Theoretical MSE"
mse_perceval_vs_true      = {}   # exact vs true STEP

# Experimental and SLOS
mse_exp_vs_surrogate      = {}   # <-- their "Experimental MSE"
mse_exp_vs_true           = {}
mse_slos_vs_surrogate     = {}
mse_slos_vs_true          = {}

print("=" * 65)
print(f"  Loading results for {FUNC_NAME}  method={ANGLE_METHOD}")
print(f"  L values: {L_VALUES}")
print("=" * 65)

for L in L_VALUES:

    tag_exp  = f"{FUNC_NAME}_L{L}_N{N_SHOTS_EXP}_x{N_X}"
    tag_slos = f"{FUNC_NAME}_L{L}_N{N_SHOTS_SLOS}_x{N_X}"

    print(f"\n  L = {L}")

    # ── Load x_values ─────────────────────────────────────────
    x_file = os.path.join(RESULTS_DIR, f"x_values_{tag_exp}.npy")
    x_vals = load_npy(x_file, f"x_values L={L}")
    if x_vals is None:
        x_file = os.path.join(RESULTS_DIR, f"x_values_{tag_slos}.npy")
        x_vals = load_npy(x_file, f"x_values slos L={L}")
    if x_vals is None:
        x_vals = np.linspace(-np.pi, np.pi, N_X)
        print(f"    x_values not found -- using linspace(-pi,pi,{N_X})")

    # ── Load angle files ──────────────────────────────────────
    theta_file = os.path.join(RESULTS_DIR,
                              f"theta_{FUNC_LOWER}_{ANGLE_METHOD}_L{L}.npy")
    phi_file   = os.path.join(RESULTS_DIR,
                              f"phi_{FUNC_LOWER}_{ANGLE_METHOD}_L{L}.npy")
    theta_arr  = load_npy(theta_file, f"theta L={L}")
    phi_arr    = load_npy(phi_file,   f"phi L={L}")

    # ── Compute Perceval analytic Z (exact, no sampling noise) ─
    if theta_arr is not None and phi_arr is not None:
        print(f"    Computing Perceval analytic Z (L={L})...",
              end=" ", flush=True)
        f_perceval = np.array([
            perceval_analytic_Z(theta_arr, phi_arr, x, L)
            for x in x_vals
        ])
        print("done.")

        f_true      = true_step(x_vals)
        f_surrogate = surrogate_step(x_vals)

        mse_perceval_vs_surrogate[L] = float(
            np.mean((f_perceval - f_surrogate)**2))
        mse_perceval_vs_true[L]      = float(
            np.mean((f_perceval - f_true)**2))

        print(f"    Perceval analytic vs Surrogate : "
              f"MSE={mse_perceval_vs_surrogate[L]:.6f}  "
              f"log10={np.log10(mse_perceval_vs_surrogate[L]):.3f}")
        print(f"    Perceval analytic vs True STEP : "
              f"MSE={mse_perceval_vs_true[L]:.6f}  "
              f"log10={np.log10(mse_perceval_vs_true[L]):.3f}")
    else:
        f_perceval  = None
        f_true      = true_step(x_vals)
        f_surrogate = surrogate_step(x_vals)
        mse_perceval_vs_surrogate[L] = float('nan')
        mse_perceval_vs_true[L]      = float('nan')
        print(f"    Angle files missing -- Perceval analytic will be NaN")

    # ── Experimental (QPU) results ────────────────────────────
    z_exp_file = os.path.join(RESULTS_DIR, f"z_experimental_{tag_exp}.npy")
    z_exp      = load_npy(z_exp_file, f"z_experimental L={L}")

    if z_exp is not None:
        n_valid = int(np.sum(~np.isnan(z_exp)))
        mse_exp_vs_surrogate[L] = float(
            np.nanmean((z_exp - f_surrogate)**2))
        mse_exp_vs_true[L]      = float(
            np.nanmean((z_exp - f_true)**2))
        print(f"    Exp ({n_valid}/{N_X} valid):  "
              f"vs_surr={mse_exp_vs_surrogate[L]:.4f}  "
              f"vs_true={mse_exp_vs_true[L]:.4f}")
    else:
        mse_exp_vs_surrogate[L] = float('nan')
        mse_exp_vs_true[L]      = float('nan')
        print(f"    Experimental: NOT FOUND")

    # ── SLOS results ──────────────────────────────────────────
    z_slos_file = os.path.join(RESULTS_DIR, f"z_slos_{tag_slos}.npy")
    z_slos      = load_npy(z_slos_file, f"z_slos L={L}")

    if z_slos is not None:
        x_slos_file = os.path.join(RESULTS_DIR,
                                   f"x_values_{tag_slos}.npy")
        x_slos_vals = load_npy(x_slos_file, f"x_values slos L={L}")
        if x_slos_vals is None:
            x_slos_vals = x_vals

        f_true_slos      = true_step(x_slos_vals)
        f_surrogate_slos = surrogate_step(x_slos_vals)

        mse_slos_vs_surrogate[L] = float(
            np.nanmean((z_slos - f_surrogate_slos)**2))
        mse_slos_vs_true[L]      = float(
            np.nanmean((z_slos - f_true_slos)**2))
        print(f"    SLOS:  "
              f"vs_surr={mse_slos_vs_surrogate[L]:.4f}  "
              f"vs_true={mse_slos_vs_true[L]:.4f}")
    else:
        mse_slos_vs_surrogate[L] = float('nan')
        mse_slos_vs_true[L]      = float('nan')
        print(f"    SLOS: NOT FOUND")

print("\n" + "=" * 65)
print("  All data loaded. Generating 2 plots...")
print("=" * 65)

L_arr = np.array(L_VALUES)

# ============================================================
# Shared helpers
# ============================================================

def plot_series(ax, mse_dict, label, color, ls, marker, ms, lw):
    mse_vals = np.array([mse_dict.get(L, np.nan) for L in L_VALUES])
    valid    = (mse_vals > 0) & ~np.isnan(mse_vals)
    ax.plot(L_arr[valid], np.log10(mse_vals[valid]),
            color=color,
            linestyle=ls if ls else 'none',
            marker=marker if marker else '',
            markersize=ms,
            linewidth=lw,
            label=label)

def finish_ax(ax, title):
    ax.set_xlabel("L  (QSP layers)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\mathrm{MSE})$", fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(L_VALUES)
    ax.set_xticklabels([str(l) for l in L_VALUES], rotation=45, fontsize=9)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

# ============================================================
# Plot 1 -- Bu et al. Figure 3 style
#
# BLACK solid  : Perceval analytic Z vs Surrogate
#                = their "Theoretical MSE"
#                Uses compute_unitary() -- zero sampling noise.
#                Decreases with L because PaddleQuantum directly
#                optimizes this quantity.
#
# RED squares  : QPU experimental Z vs Surrogate
#                = their "Experimental MSE"
# ============================================================

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.suptitle(
    f"log(MSE) vs L  —  {FUNC_NAME}  method={ANGLE_METHOD}  "
    f"N_shots_exp={N_SHOTS_EXP}  N_x={N_X}",
    fontsize=11, fontweight='bold'
)

plot_series(ax1, mse_perceval_vs_surrogate,
            "Theoretical MSE  (Perceval exact vs Surrogate)",
            "black", "-", None, 0, 2.5)
plot_series(ax1, mse_exp_vs_surrogate,
            "Experimental MSE  (QPU hardware vs Surrogate)",
            "red", "", "s", 8, 0.0)

finish_ax(ax1,
    f"log(MSE) vs L  —  {FUNC_NAME}  (Bu et al. Fig. 3 style)\n"
    f"Black solid = Perceval exact vs Surrogate  |  "
    f"Red squares = QPU experiment vs Surrogate")

plt.tight_layout()
bu_file = f"mse_vs_L_Bu_style_{FUNC_NAME}_{ANGLE_METHOD}.png"
plt.savefig(bu_file, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Saved: {bu_file}")

# ============================================================
# Plot 2 -- Full comparison
#
# Black solid  : Perceval analytic vs Surrogate (Theoretical)
# Grey dashed  : Perceval analytic vs True STEP
# Blue solid   : SLOS vs True STEP
# Blue squares : Exp vs True STEP
# Red solid    : SLOS vs Surrogate
# Red squares  : Exp vs Surrogate
# ============================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle(
    f"log(MSE) vs L  —  {FUNC_NAME}  method={ANGLE_METHOD}  "
    f"N_shots_exp={N_SHOTS_EXP}  N_shots_slos={N_SHOTS_SLOS}  N_x={N_X}",
    fontsize=11, fontweight='bold'
)

full_series = [
    (mse_perceval_vs_surrogate,
     "Perceval exact vs Surrogate  (Theoretical)",
     "black", "-",   None, 0,  2.5),
    (mse_perceval_vs_true,
     "Perceval exact vs True STEP",
     "grey",  "--",  None, 0,  2.0),
    (mse_slos_vs_true,
     "SLOS vs True STEP",
     "blue",  "-",   None, 0,  2.0),
    (mse_exp_vs_true,
     "Exp vs True STEP",
     "blue",  "",    "s",  8,  0.0),
    (mse_slos_vs_surrogate,
     "SLOS vs Surrogate",
     "red",   "-",   None, 0,  2.0),
    (mse_exp_vs_surrogate,
     "Exp vs Surrogate",
     "red",   "",    "s",  8,  0.0),
]

for mse_dict, label, color, ls, marker, ms, lw in full_series:
    plot_series(ax2, mse_dict, label, color, ls, marker, ms, lw)

finish_ax(ax2,
    f"log(MSE) vs L  —  {FUNC_NAME}\n"
    f"Black = Perceval exact vs Surrogate  |  Grey = Perceval exact vs True STEP  |  "
    f"Solid = SLOS  |  Squares = Experiment  |  Blue = True STEP  |  Red = Surrogate")

plt.tight_layout()
combined_file = f"mse_vs_L_combined_{FUNC_NAME}_{ANGLE_METHOD}.png"
plt.savefig(combined_file, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Saved: {combined_file}")

print("\n" + "=" * 65)
print("  2 plots saved:")
print(f"    {bu_file}")
print(f"      --> Bu et al. style: Perceval exact vs Surrogate + QPU exp")
print(f"    {combined_file}")
print(f"      --> Full comparison: all curves")
print("=" * 65)