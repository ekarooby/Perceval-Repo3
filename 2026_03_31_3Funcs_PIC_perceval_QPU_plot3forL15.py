# ============================================================
# STANDALONE PLOT 3 -- Experiments vs SLOS vs Perceval
#                      analytic vs Classical results
# ============================================================
#
# PURPOSE:
#   Generate the 3-way comparison plot (right panel) from
#   already-saved QPU experimental results and SLOS results.
#   Run this AFTER the QPU code has finished and saved data.
#   Does NOT submit any jobs to the QPU -- purely local.
#
# REQUIRED FILES (must be in the same folder as this script):
#   z_experimental_{FILE_TAG}.npy   -- saved by QPU code
#   x_values_{FILE_TAG}.npy         -- saved by QPU code
#   z_slos_{SLOS_TAG}.npy           -- saved by SLOS code
#   x_values_{SLOS_TAG}.npy         -- saved by SLOS code
#   theta_{func}_{method}_L{L}.npy  -- angle file
#   phi_{func}_{method}_L{L}.npy    -- angle file
#
# HOW TO USE:
#   1. Copy SLOS files from Perceval-Repo2 to Perceval-Repo3
#      if they are in a different folder
#   2. Set FUNC_NAME, ANGLE_L, ANGLE_METHOD, N_SHOTS, N_X
#      to match exactly what you used in the QPU run
#   3. Run: python plot3_comparison.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
import os

# ============================================================
# ---- SETTINGS: must match your QPU run exactly ----
# ============================================================

FUNC_NAME    = "STEP"   # <-- same as QPU run
ANGLE_L      = 15       # <-- same as QPU run
ANGLE_METHOD = "pq"     # <-- same as QPU run
N_SHOTS      = 5000     # <-- same as QPU run
N_X          = 100      # <-- same as QPU run (len of x_values)

# ============================================================
# Derived settings -- do not change
# ============================================================

FUNC_LOWER = FUNC_NAME.lower()
N_approx   = 100
FILE_TAG   = f"{FUNC_NAME}_L{ANGLE_L}_N{N_SHOTS}_x{N_X}"
SLOS_TAG   = f"{FUNC_NAME}_L{ANGLE_L}_N{N_SHOTS}_x{N_X}"

print(f"FILE_TAG : {FILE_TAG}")
print(f"SLOS_TAG : {SLOS_TAG}")

# ============================================================
# Step 1: Load saved experimental results
# ============================================================

z_experimental = np.load(f"z_experimental_{FILE_TAG}.npy")
x_values       = np.load(f"x_values_{FILE_TAG}.npy")
L              = ANGLE_L
N_X            = len(x_values)

n_valid   = int(np.sum(~np.isnan(z_experimental)))
n_missing = int(np.sum( np.isnan(z_experimental)))
print(f"\nLoaded experimental data: {n_valid} valid, {n_missing} missing")

# ============================================================
# Step 2: Load QSP angles
# ============================================================

theta = np.load(f"theta_{FUNC_LOWER}_{ANGLE_METHOD}_L{ANGLE_L}.npy")
phi   = np.load(f"phi_{FUNC_LOWER}_{ANGLE_METHOD}_L{ANGLE_L}.npy")
print(f"Loaded angles: {FUNC_NAME}  {ANGLE_METHOD}  L={L}")

# ============================================================
# Step 3: Target functions
# ============================================================

def get_surrogate(func_name):
    if func_name == "STEP":
        return lambda x: (2.0 / np.pi) * np.arctan(N_approx * x)
    elif func_name == "ReLU":
        return lambda x: np.log(1 + np.exp(N_approx * x)) / N_approx
    elif func_name == "SELU":
        alpha, scale = 1.6733, 1.0507
        return lambda x: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def get_true_func(func_name):
    if func_name == "STEP":
        return lambda x: np.where(x >= 0, 1.0, -1.0)
    elif func_name == "ReLU":
        return lambda x: np.maximum(0.0, x)
    elif func_name == "SELU":
        alpha, scale = 1.6733, 1.0507
        return lambda x: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

surrogate_func = get_surrogate(FUNC_NAME)
true_func      = get_true_func(FUNC_NAME)

x_fine      = np.linspace(-np.pi, np.pi, 300)
f_surrogate = surrogate_func(x_fine)
f_true      = true_func(x_fine)

# ============================================================
# Step 4: Classical reference (pure numpy, no Perceval)
# ============================================================

def Ry_mat(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz_mat(p):
    return np.array([[np.exp(-1j*p/2), 0],[0, np.exp(1j*p/2)]], dtype=complex)

def A_mat(t, p):
    return Ry_mat(t) @ Rz_mat(p)

def classical_qsp(theta_arr, phi_arr, x_val, L):
    """Pure numpy QSP circuit -- Bu et al. convention, no Perceval."""
    W = A_mat(theta_arr[0], phi_arr[0])
    for j in range(1, L + 1):
        W = A_mat(theta_arr[j], phi_arr[j]) @ Rz_mat(x_val) @ W
    psi = W @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

print("\nComputing classical reference (pure numpy)...")
f_classical = np.array([classical_qsp(theta, phi, x, L) for x in x_values])

# ============================================================
# Step 5: Perceval analytic reference
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

print("Computing Perceval analytic reference...")
f_perceval_analytic = np.zeros(N_X)
for i, x_val in enumerate(x_values):
    circuit = build_qsp_pic(theta, phi, x_val, L)
    U   = np.array(circuit.compute_unitary())
    psi = U @ np.array([1.0, 0.0])
    f_perceval_analytic[i] = abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Step 6: Load SLOS results
# ============================================================

slos_x_file = f"x_values_{SLOS_TAG}.npy"
slos_z_file = f"z_slos_{SLOS_TAG}.npy"

if not os.path.exists(slos_z_file):
    print(f"\nERROR: SLOS file not found: {slos_z_file}")
    print(f"  Copy it from the folder where you ran the SLOS code.")
    exit()

x_slos = np.load(slos_x_file)
z_slos = np.load(slos_z_file)
print(f"SLOS results loaded: {slos_z_file}")

# ============================================================
# Step 7: Compute MSEs
# ============================================================

z_slos_interp        = np.interp(x_values, x_slos, z_slos)
mse_exp_vs_analytic  = np.nanmean((z_experimental - f_perceval_analytic)**2)
mse_exp_vs_classical = np.nanmean((z_experimental - f_classical)**2)
mse_exp_vs_slos      = np.nanmean((z_experimental - z_slos_interp)**2)
mse_exp_vs_true      = np.nanmean((z_experimental - true_func(x_values))**2)

print(f"\n========== MSE Report  [{FILE_TAG}] ==========")
print(f"  MSE experimental vs Perceval analytic : {mse_exp_vs_analytic:.4f}")
print(f"  MSE experimental vs Classical results : {mse_exp_vs_classical:.4f}")
print(f"  MSE experimental vs SLOS              : {mse_exp_vs_slos:.4f}")
print(f"  MSE experimental vs true {FUNC_NAME:<5}        : {mse_exp_vs_true:.4f}")
print(f"==============================================")

# ============================================================
# Step 8: Plot
# ============================================================

xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
fig.suptitle(
    f"Experiments: Paddle Quantum  {FUNC_NAME}  L={L}  "
    f"N_shots={N_SHOTS}  N_x={N_X}",
    fontsize=12, fontweight='bold'
)

ax.plot(x_fine,   f_true,              'k-',  lw=2.5,
        label=f"True {FUNC_NAME}",             zorder=3)
ax.plot(x_values, f_classical,         'm-',  lw=1.5,
        label=f"Classical  MSE={mse_exp_vs_classical:.4f}",
        zorder=4)
ax.plot(x_values, f_perceval_analytic, 'g--', lw=2,
        label=f"Perceval analytic  MSE={mse_exp_vs_analytic:.4f}",
        zorder=5)
ax.plot(x_slos,   z_slos,              'b.',  ms=6,
        label=f"SLOS  MSE={mse_exp_vs_slos:.4f}",
        zorder=6)
ax.plot(x_values, z_experimental,      'r.',  ms=10,
        label=f"Experimental",                zorder=7)

ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=11)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Z = p0 - p1", fontsize=12)
ax.set_title(
    f"Experiments vs SLOS, Perceval analytics,\n"
    f"Classical results  ({FUNC_NAME}  L={L})",
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plot_filename = f"plot3_comparison_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved: {plot_filename}")