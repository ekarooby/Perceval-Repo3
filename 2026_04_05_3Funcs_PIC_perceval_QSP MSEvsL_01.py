# ============================================================
# qsp_mse_vs_L.py
#
# PURPOSE:
#   Load saved QPU experimental and SLOS simulation results
#   for multiple L values and plot log(MSE) vs L for 6
#   reference comparisons:
#
#   Experimental vs:
#     1. True STEP function
#     2. Surrogate (arctan approximation)
#     3. Classical (pure numpy matrix math)
#
#   SLOS simulation vs:
#     4. True STEP function
#     5. Surrogate (arctan approximation)
#     6. Classical (pure numpy matrix math)
#
# FILE NAMING CONVENTION:
#   Loads files saved by qsp_experiment_qpu.py and
#   qsp_slos_simulation.py with FILE_TAG format:
#     STEP_L{L}_N{N_SHOTS}_x{N_X}
#   Example: z_experimental_STEP_L15_N5000_x100.npy
#
# OUTPUTS:
#   6 PNG files, one per plot:
#     mse_vs_L_exp_vs_true.png
#     mse_vs_L_exp_vs_surrogate.png
#     mse_vs_L_exp_vs_classical.png
#     mse_vs_L_slos_vs_true.png
#     mse_vs_L_slos_vs_surrogate.png
#     mse_vs_L_slos_vs_classical.png
#
# HOW TO RUN:
#   1. Set L_VALUES, N_SHOTS_EXP, N_SHOTS_SLOS, N_X below
#      to match your actual saved runs.
#   2. Make sure all .npy files and angle files are in the
#      same directory as this script (or adjust RESULTS_DIR).
#   3. Run: python qsp_mse_vs_L.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# SETTINGS -- adjust to match your saved runs
# ============================================================

FUNC_NAME    = "STEP"
ANGLE_METHOD = "pq"

# L values you have run (both QPU and SLOS)
L_VALUES = [5, 10, 15, 20, 24, 30, 60, 120, 180, 240, 300, 330, 360]

# N_SHOTS used in each run
N_SHOTS_EXP  = 5000     # QPU experimental runs
N_SHOTS_SLOS = 5000     # SLOS simulation runs

# Number of x points used in each run
N_X = 100

# Directory where all .npy files and angle files are stored
# Use "." if they are in the same folder as this script
RESULTS_DIR = "."

# ============================================================
# Reference functions
# ============================================================

N_approx = 100   # arctan sharpness -- must match what was used in QPU/SLOS code

def true_step(x):
    return np.where(x >= 0, 1.0, -1.0)

def surrogate_step(x):
    return (2.0 / np.pi) * np.arctan(N_approx * x)

# ============================================================
# Classical QSP circuit (pure numpy, no Perceval)
# Matches the build_qsp_pic() convention exactly:
#   A(t,p) = Ry(t) @ Rz(p)
#   W = A(t0,p0), then for j=1..L: W = A(tj,pj) @ Rz(x) @ W
# ============================================================

def Ry_mat(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rz_mat(p):
    return np.array([[np.exp(-1j * p / 2), 0],
                     [0, np.exp( 1j * p / 2)]], dtype=complex)

def A_mat(t, p):
    return Ry_mat(t) @ Rz_mat(p)

def classical_qsp_Z(theta_arr, phi_arr, x_val, L):
    """Compute Z = p0 - p1 via pure numpy matrix math."""
    W = A_mat(theta_arr[0], phi_arr[0])
    for j in range(1, L + 1):
        W = A_mat(theta_arr[j], phi_arr[j]) @ Rz_mat(x_val) @ W
    psi = W @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Helper: load a .npy file with a clear error if missing
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

# Storage: dict[L] = mse value (or nan if data missing)
mse_exp_vs_true       = {}
mse_exp_vs_surrogate  = {}
mse_exp_vs_classical  = {}
mse_slos_vs_true      = {}
mse_slos_vs_surrogate = {}
mse_slos_vs_classical = {}

print("=" * 65)
print(f"  Loading results for {FUNC_NAME}  method={ANGLE_METHOD}")
print(f"  L values: {L_VALUES}")
print("=" * 65)

for L in L_VALUES:

    tag_exp  = f"{FUNC_NAME}_L{L}_N{N_SHOTS_EXP}_x{N_X}"
    tag_slos = f"{FUNC_NAME}_L{L}_N{N_SHOTS_SLOS}_x{N_X}"

    print(f"\n  L = {L}")

    # ── Load x_values ────────────────────────────────────────
    x_file = os.path.join(RESULTS_DIR, f"x_values_{tag_exp}.npy")
    x_vals = load_npy(x_file, f"x_values L={L}")
    if x_vals is None:
        # Try SLOS tag
        x_file = os.path.join(RESULTS_DIR, f"x_values_{tag_slos}.npy")
        x_vals = load_npy(x_file, f"x_values (slos tag) L={L}")
    if x_vals is None:
        # Fall back to linspace
        x_vals = np.linspace(-np.pi, np.pi, N_X)
        print(f"    x_values not found -- using linspace(-pi,pi,{N_X})")

    # ── Load angle files to compute classical reference ───────
    theta_file = os.path.join(RESULTS_DIR,
                              f"theta_{FUNC_LOWER}_{ANGLE_METHOD}_L{L}.npy")
    phi_file   = os.path.join(RESULTS_DIR,
                              f"phi_{FUNC_LOWER}_{ANGLE_METHOD}_L{L}.npy")
    theta_arr  = load_npy(theta_file, f"theta L={L}")
    phi_arr    = load_npy(phi_file,   f"phi L={L}")

    # Compute classical reference at x_vals
    if theta_arr is not None and phi_arr is not None:
        print(f"    Computing classical reference (L={L})...", end=" ")
        f_classical = np.array([
            classical_qsp_Z(theta_arr, phi_arr, x, L) for x in x_vals
        ])
        print("done.")
    else:
        f_classical = None
        print(f"    Angle files missing -- classical MSE will be NaN")

    # Reference curves at x_vals
    f_true      = true_step(x_vals)
    f_surrogate = surrogate_step(x_vals)

    # ── Experimental results ──────────────────────────────────
    z_exp_file = os.path.join(RESULTS_DIR, f"z_experimental_{tag_exp}.npy")
    z_exp      = load_npy(z_exp_file, f"z_experimental L={L}")

    if z_exp is not None:
        # np.nanmean excludes NaN points (Fix 2 from QPU code)
        mse_exp_vs_true[L]      = float(np.nanmean((z_exp - f_true)**2))
        mse_exp_vs_surrogate[L] = float(np.nanmean((z_exp - f_surrogate)**2))
        mse_exp_vs_classical[L] = float(
            np.nanmean((z_exp - f_classical)**2)
        ) if f_classical is not None else float('nan')
        n_valid = int(np.sum(~np.isnan(z_exp)))
        print(f"    Experimental: {n_valid}/{N_X} valid points  "
              f"MSE_true={mse_exp_vs_true[L]:.4f}  "
              f"MSE_surr={mse_exp_vs_surrogate[L]:.4f}  "
              f"MSE_clas={mse_exp_vs_classical[L]:.4f}")
    else:
        mse_exp_vs_true[L]      = float('nan')
        mse_exp_vs_surrogate[L] = float('nan')
        mse_exp_vs_classical[L] = float('nan')
        print(f"    Experimental: NOT FOUND")

    # ── SLOS results ──────────────────────────────────────────
    z_slos_file = os.path.join(RESULTS_DIR, f"z_slos_{tag_slos}.npy")
    z_slos      = load_npy(z_slos_file, f"z_slos L={L}")

    if z_slos is not None:
        # Load matching x_values for SLOS (may differ from exp x_vals)
        x_slos_file = os.path.join(RESULTS_DIR, f"x_values_{tag_slos}.npy")
        x_slos_vals = load_npy(x_slos_file, f"x_values slos L={L}")
        if x_slos_vals is None:
            x_slos_vals = x_vals   # fallback

        f_true_slos      = true_step(x_slos_vals)
        f_surrogate_slos = surrogate_step(x_slos_vals)

        if theta_arr is not None and phi_arr is not None:
            f_classical_slos = np.array([
                classical_qsp_Z(theta_arr, phi_arr, x, L) for x in x_slos_vals
            ])
        else:
            f_classical_slos = None

        mse_slos_vs_true[L]      = float(np.nanmean((z_slos - f_true_slos)**2))
        mse_slos_vs_surrogate[L] = float(np.nanmean((z_slos - f_surrogate_slos)**2))
        mse_slos_vs_classical[L] = float(
            np.nanmean((z_slos - f_classical_slos)**2)
        ) if f_classical_slos is not None else float('nan')
        print(f"    SLOS:         MSE_true={mse_slos_vs_true[L]:.4f}  "
              f"MSE_surr={mse_slos_vs_surrogate[L]:.4f}  "
              f"MSE_clas={mse_slos_vs_classical[L]:.4f}")
    else:
        mse_slos_vs_true[L]      = float('nan')
        mse_slos_vs_surrogate[L] = float('nan')
        mse_slos_vs_classical[L] = float('nan')
        print(f"    SLOS:         NOT FOUND")

L_arr = np.array(L_VALUES)

print("\n" + "=" * 65)
print("  All data loaded. Generating 2 plots...")
print("=" * 65)

# ============================================================
# Plot 7 -- Combined: all 6 curves in one plot
#
# Color encodes the REFERENCE:
#   Green  = vs Classical (numpy)
#   Blue   = vs True STEP
#   Red    = vs Surrogate (arctan)
#
# Line style encodes the SOURCE:
#   Solid line  = SLOS simulation
#   Square dots = Experimental (QPU)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    f"log(MSE) vs L  —  {FUNC_NAME}  method={ANGLE_METHOD}  "
    f"N_shots_exp={N_SHOTS_EXP}  N_shots_slos={N_SHOTS_SLOS}  N_x={N_X}",
    fontsize=11, fontweight='bold'
)

# Each entry: (mse_dict, label, color, linestyle, marker, markersize, linewidth)
combined_series = [
    # ── Green: vs Classical ───────────────────────────────────
    (mse_slos_vs_classical,
     "SLOS vs Classical",
     "green", "-",  None, 0,   2.0),
    (mse_exp_vs_classical,
     "Exp vs Classical",
     "green", "",   "s",  8,   0.0),

    # ── Blue: vs True STEP ────────────────────────────────────
    (mse_slos_vs_true,
     "SLOS vs True STEP",
     "blue",  "-",  None, 0,   2.0),
    (mse_exp_vs_true,
     "Exp vs True STEP",
     "blue",  "",   "s",  8,   0.0),

    # ── Red: vs Surrogate ─────────────────────────────────────
    (mse_slos_vs_surrogate,
     "SLOS vs Surrogate",
     "red",   "-",  None, 0,   2.0),
    (mse_exp_vs_surrogate,
     "Exp vs Surrogate",
     "red",   "",   "s",  8,   0.0),
]

for mse_dict, label, color, ls, marker, ms, lw in combined_series:
    mse_vals = np.array([mse_dict.get(L, np.nan) for L in L_VALUES])
    valid    = (mse_vals > 0) & ~np.isnan(mse_vals)
    L_plot   = L_arr[valid]
    log_mse  = np.log10(mse_vals[valid])

    ax.plot(L_plot, log_mse,
            color=color,
            linestyle=ls if ls else 'none',
            marker=marker if marker else '',
            markersize=ms,
            linewidth=lw,
            label=label)

ax.set_xlabel("L  (QSP layers)", fontsize=12)
ax.set_ylabel(r"$\log_{10}(\mathrm{MSE})$", fontsize=12)
ax.set_title(
    f"log(MSE) vs L  —  {FUNC_NAME}\n"
    f"Solid line = SLOS  |  Square dots = Experiment  |  "
    f"Green = Classical  |  Blue = True STEP  |  Red = Surrogate",
    fontsize=10
)
ax.set_xticks(L_VALUES)
ax.set_xticklabels([str(l) for l in L_VALUES], rotation=45, fontsize=9)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
combined_file = f"mse_vs_L_combined_{FUNC_NAME}_{ANGLE_METHOD}.png"
plt.savefig(combined_file, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Saved: {combined_file}")

print("\n" + "=" * 65)
print("=" * 65)

# ============================================================
# Plot 2 -- Blue + Red only: STEP and Surrogate curves
#           (no green / Classical -- focuses on approximation quality)
#
# Blue = vs True STEP
# Red  = vs Surrogate
# Solid line = SLOS  |  Square dots = Experiment
# ============================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle(
    f"log(MSE) vs L  —  {FUNC_NAME}  method={ANGLE_METHOD}  "
    f"N_shots_exp={N_SHOTS_EXP}  N_shots_slos={N_SHOTS_SLOS}  N_x={N_X}",
    fontsize=11, fontweight='bold'
)

approx_series = [
    # ── Blue: vs True STEP ────────────────────────────────────
    (mse_slos_vs_true,
     "SLOS vs True STEP",
     "blue",  "-",  None, 0,   2.0),
    (mse_exp_vs_true,
     "Exp vs True STEP",
     "blue",  "",   "s",  8,   0.0),

    # ── Red: vs Surrogate ─────────────────────────────────────
    (mse_slos_vs_surrogate,
     "SLOS vs Surrogate",
     "red",   "-",  None, 0,   2.0),
    (mse_exp_vs_surrogate,
     "Exp vs Surrogate",
     "red",   "",   "s",  8,   0.0),
]

for mse_dict, label, color, ls, marker, ms, lw in approx_series:
    mse_vals = np.array([mse_dict.get(L, np.nan) for L in L_VALUES])
    valid    = (mse_vals > 0) & ~np.isnan(mse_vals)
    L_plot   = L_arr[valid]
    log_mse  = np.log10(mse_vals[valid])

    ax2.plot(L_plot, log_mse,
             color=color,
             linestyle=ls if ls else 'none',
             marker=marker if marker else '',
             markersize=ms,
             linewidth=lw,
             label=label)

ax2.set_xlabel("L  (QSP layers)", fontsize=12)
ax2.set_ylabel(r"$\log_{10}(\mathrm{MSE})$", fontsize=12)
ax2.set_title(
    f"log(MSE) vs L  —  {FUNC_NAME}  (approximation quality only)\n"
    f"Solid line = SLOS  |  Square dots = Experiment  |  "
    f"Blue = True STEP  |  Red = Surrogate",
    fontsize=10
)
ax2.set_xticks(L_VALUES)
ax2.set_xticklabels([str(l) for l in L_VALUES], rotation=45, fontsize=9)
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
approx_file = f"mse_vs_L_approx_{FUNC_NAME}_{ANGLE_METHOD}.png"
plt.savefig(approx_file, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Saved: {approx_file}")

print("\n" + "=" * 65)
print("  2 plots saved:")
print(f"    {combined_file}  (all 6 curves: blue + red + green)")
print(f"    {approx_file}    (4 curves: blue + red only)")
print("=" * 65)