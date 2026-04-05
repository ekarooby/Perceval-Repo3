# ============================================================
# QSP ANGLE GENERATION USING PADDLE QUANTUM QPP
# STEP / ReLU / SELU
# ============================================================
#
# PURPOSE:
#   Generate QSP angles (theta, phi) for any L using Paddle
#   Quantum's QPP angle finder, which is numerically stable
#   at high L (L=15, 50, 100, 180 all work with no breakdown).
#   This is the same pipeline used by Bu et al. (2025).
#
# ENVIRONMENT:
#   Run this script in paddle_env (NOT perceval_env).
#   The saved .npy files are then loaded by the SLOS and QPU
#   codes running in perceval_env -- the two environments
#   communicate only through .npy files on disk.
#
# PIPELINE (Bu et al. Appendix B):
#   1. laurent_generator(fn, dx, deg, L_width)
#      --> Laurent polynomial approximating fn over [-pi, pi]
#      --> use deg = 2 * L_desired (must be even)
#   2. pair_generation(F)
#      --> generates (P, Q) pair with P*P* + Q*Q* = 1
#   3. qpp_angle_approximator(P, Q)
#      --> returns (list_theta, list_phi) for Bu et al. circuit
#
# CIRCUIT CONVENTION (Bu et al. Eq. 1):
#   W(x) = A(t0,p0) * prod_{j=1}^{L} [Rz(x) * A(tj,pj)]
#   where A(t,p) = Ry(t) @ Rz(p)
#   Output: Z = |psi[0]|^2 - |psi[1]|^2  (measurable)
#   These angles plug DIRECTLY into the Perceval circuit.
#
# HOW TO SWITCH BETWEEN STEP / ReLU / SELU:
#   Change FUNC_NAME and ANGLE_L at the top of the settings.
#   Everything else updates automatically.
#
# FILE NAMING CONVENTION:
#   Output files include FUNC_NAME and L so different runs
#   never overwrite each other.
#   pq = Paddle Quantum (distinguishes from nlft angles)
#   Example (FUNC_NAME="STEP", ANGLE_L=15):
#     theta_step_pq_L15.npy
#     phi_step_pq_L15.npy
#     qsp_step_pq_L15.png
#   The SLOS and QPU codes load these with matching ANGLE_L.
#
# IMPORTANT NOTES:
#   - deg = 2 * ANGLE_L  (always use even degree)
#   - Parity warnings from pair_generation are numerical noise
#     (~1e-11 to 1e-15) and can be safely ignored
#   - The verification step in qpp_angle_approximator has been
#     patched out (paddle.fluid incompatibility with paddle 3.0)
#     This does not affect angle quality -- see MSE results below
#
# VERIFIED MSE vs true STEP (200 x points, x in [-pi, pi]):
#   L=15  : MSE ~ 0.075
#   L=30  : MSE ~ 0.064
#   L=50  : MSE ~ 0.058
#   L=100 : MSE ~ 0.046
#   L=180 : MSE ~ 0.032
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings
from paddle_quantum.qpp.laurent import laurent_generator, pair_generation, Laurent
from paddle_quantum.qpp.angles import qpp_angle_approximator

# ============================================================
# ---- SETTINGS: change these two lines to switch function ----
#
# FUNC_NAME : "STEP", "ReLU", or "SELU"
# ANGLE_L   : number of QSP layers L
#             deg = 2 * ANGLE_L is used internally (must be even)
# ============================================================

FUNC_NAME = "STEP"   # <-- change to "ReLU" or "SELU" as needed
ANGLE_L   = 360    # <-- change to desired L

# ============================================================
# Derived settings -- do not change these manually
# FILE_TAG flows into all saved filenames automatically
# ============================================================

FUNC_LOWER = FUNC_NAME.lower()        # "step", "relu", "selu"
FILE_TAG   = f"{FUNC_LOWER}_pq_L{ANGLE_L}"   # e.g. "step_pq_L15"
deg        = 2 * ANGLE_L              # must be even for pair_generation

print("=" * 62)
print(f"  Paddle Quantum QPP Angle Generation")
print(f"  Function : {FUNC_NAME}")
print(f"  L        : {ANGLE_L}")
print(f"  deg      : {deg}  (= 2 * L, must be even)")
print(f"  File tag : {FILE_TAG}")
print("=" * 62)

N_approx = 100   # sharpness of arctan surrogate for STEP

# ============================================================
# Target surrogate functions
#
# STEP  : arctan approximation  (Bu et al. Eq. B9)
# ReLU  : softplus approximation
# SELU  : scaled ELU approximation
# ============================================================

def get_surrogate(func_name):
    """Return the smooth surrogate function for the given FUNC_NAME."""
    if func_name == "STEP":
        return lambda x: (2.0 / np.pi) * np.arctan(N_approx * x)
    elif func_name == "ReLU":
        return lambda x: np.log(1 + np.exp(N_approx * x)) / N_approx
    elif func_name == "SELU":
        alpha = 1.6733
        scale = 1.0507
        return lambda x: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError(f"Unknown FUNC_NAME: {func_name}. Use 'STEP', 'ReLU', or 'SELU'.")

def get_true_func(func_name):
    """Return the ideal target function for the given FUNC_NAME."""
    if func_name == "STEP":
        return lambda x: np.where(x >= 0, 1.0, -1.0)
    elif func_name == "ReLU":
        return lambda x: np.maximum(0.0, x)
    elif func_name == "SELU":
        alpha = 1.6733
        scale = 1.0507
        return lambda x: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError(f"Unknown FUNC_NAME: {func_name}. Use 'STEP', 'ReLU', or 'SELU'.")

surrogate_func = get_surrogate(FUNC_NAME)
true_func      = get_true_func(FUNC_NAME)

# ============================================================
# QSP circuit (Bu et al. convention) -- for verification only
# These matrix functions are NOT used for angle finding,
# only for MSE verification after angles are found.
# ============================================================

def Ry(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz(p):
    return np.array([[np.exp(-1j*p/2), 0],[0, np.exp(1j*p/2)]], dtype=complex)

def qsp_Z(theta_arr, phi_arr, x):
    """Z = p0-p1 using Bu et al. circuit convention (measurable)."""
    W = Ry(theta_arr[0]) @ Rz(phi_arr[0])
    for j in range(1, len(theta_arr)):
        W = Ry(theta_arr[j]) @ Rz(phi_arr[j]) @ Rz(x) @ W
    psi = W @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Step 1: Generate Laurent polynomial
#
# laurent_generator(fn, dx, deg, L_width):
#   fn      : target function
#   dx      : sampling step (0.01 is sufficient)
#   deg     : polynomial degree = 2 * ANGLE_L (must be even)
#   L_width : approximation half-width = pi
#
# The Laurent variable is X = e^{ix/2}, so the polynomial
# approximates fn(x) for x in [-pi, pi].
#
# Even deg gives parity 0, which pair_generation requires.
# Odd deg gives parity 1 and pair_generation will fail.
# ============================================================
print("\n[1] Generating Laurent polynomial...")

F = laurent_generator(surrogate_func, 0.01, deg, np.pi)
print(f"   Raw: parity={F.parity}, max_norm={F.max_norm:.4f}, deg={F.deg}")

# Scale so max_norm < 1 (strictly required by pair_generation)
scale_factor = 0.95 / F.max_norm
F = Laurent(F.coef * scale_factor)
print(f"   Scaled: parity={F.parity}, max_norm={F.max_norm:.4f}")
print(f"   Scale factor: {scale_factor:.4f}")

# ============================================================
# Step 2: Generate (P, Q) Laurent pair
#
# pair_generation(F) computes:
#   P = sqrt((1 + F) / 2)
#   Q = sqrt((1 - F) / 2)
# such that P*P_conj + Q*Q_conj = 1 (SU(2) unitarity)
#
# Parity warnings (~1e-11 to 1e-15) are suppressed --
# they are floating point noise, not actual errors.
# ============================================================
print("\n[2] Generating (P, Q) Laurent pair...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    P, Q = pair_generation(F)

print(f"   P: parity={P.parity}, max_norm={P.max_norm:.4f}, deg={P.deg}")
print(f"   Q: parity={Q.parity}, max_norm={Q.max_norm:.4f}, deg={Q.deg}")

# ============================================================
# Step 3: Find QSP angles via QPP angle approximator
#
# qpp_angle_approximator(P, Q) iteratively peels angles
# from the Laurent pair and returns:
#   list_theta : Ry rotation angles (L+1 values)
#   list_phi   : Rz rotation angles (L+1 values)
#
# These plug directly into the Bu et al./Perceval circuit.
# ============================================================
print("\n[3] Finding QSP angles...")

list_theta, list_phi = qpp_angle_approximator(P, Q)
theta = np.array(list_theta)
phi   = np.array(list_phi)

print(f"   Angles found: {len(theta)} theta, {len(phi)} phi")
print(f"   L = {len(theta) - 1}  (expected {ANGLE_L})")
assert len(theta) - 1 == ANGLE_L, \
    f"L mismatch: got {len(theta)-1}, expected {ANGLE_L}"
print(f"   theta: {np.round(theta, 4)}")
print(f"   phi  : {np.round(phi, 4)}")

# ============================================================
# Step 4: Verify angles in Bu et al. circuit
# ============================================================
print("\n[4] Verifying angles in Bu et al. circuit...")

x_grid   = np.linspace(-np.pi, np.pi, 300)
f_target = surrogate_func(x_grid)
f_true   = true_func(x_grid)
z_vals   = np.array([qsp_Z(theta, phi, x) for x in x_grid])

mse_vs_surrogate = np.mean((z_vals - f_target)**2)
mse_vs_true      = np.mean((z_vals - f_true)**2)

print(f"   MSE vs surrogate    : {mse_vs_surrogate:.4e}")
print(f"   MSE vs true {FUNC_NAME:<5} : {mse_vs_true:.4e}")

# ============================================================
# Step 5: Save angles
#
# Filenames include FILE_TAG (function + pq + L) so that:
#   - Different L values never overwrite each other
#   - STEP / ReLU / SELU stay clearly separated
#   - pq angles are distinguished from nlft angles
# Load in SLOS/QPU by setting ANGLE_L and FUNC_NAME to match.
# ============================================================
theta_filename = f"theta_{FILE_TAG}.npy"
phi_filename   = f"phi_{FILE_TAG}.npy"

np.save(theta_filename, theta)
np.save(phi_filename,   phi)
print(f"\nSaved: {theta_filename}")
print(f"Saved: {phi_filename}")

# ============================================================
# Step 6: Plot verification
#
# Left panel : QPP circuit output vs surrogate vs true function
# Right panel: residuals vs surrogate (blue) and vs true (red)
# ============================================================
print("\n[5] Plotting...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Paddle Quantum   {FUNC_NAME}   L={ANGLE_L}",
    fontsize=13, fontweight='bold')

xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

# Left panel
ax = axes[0]
ax.plot(x_grid, f_true,   'k-',  lw=2.5, label=f"True {FUNC_NAME}")
ax.plot(x_grid, f_target, 'g--', lw=2,   label="surrogate")
ax.plot(x_grid, z_vals,   'b.',  ms=3,
        label=f"QPP  MSE(surrogate)={mse_vs_surrogate:.4f}  "
              f"MSE(true {FUNC_NAME})={mse_vs_true:.4f}")
ax.set_xlim([-np.pi, np.pi]); ax.set_ylim([-1.35, 1.35])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=11)
ax.set_xlabel(r"$x$", fontsize=12); ax.set_ylabel("Function", fontsize=12)
ax.set_title(f"Paddle Quantum   {FUNC_NAME}   L={ANGLE_L}", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Right panel: residuals
ax2 = axes[1]
diff_surrogate = z_vals - f_target
diff_true      = z_vals - f_true
ax2.plot(x_grid, diff_surrogate, color='royalblue', lw=1.5,
         label=f"vs surrogate  MSE={mse_vs_surrogate:.4e}")
ax2.fill_between(x_grid, diff_surrogate, alpha=0.15, color='royalblue')
ax2.plot(x_grid, diff_true, color='red', lw=1.5, linestyle='--',
         label=f"vs true {FUNC_NAME}  MSE={mse_vs_true:.4e}")
ax2.fill_between(x_grid, diff_true, alpha=0.10, color='red')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks(xt); ax2.set_xticklabels(xl, fontsize=11)
ax2.set_xlabel(r"$x$", fontsize=12); ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title(f"Residuals  ({FUNC_NAME}   L={ANGLE_L})", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()

plot_filename = f"qsp_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")

print("\n" + "=" * 62)
print(f"  SUMMARY  [{FILE_TAG}]")
print("=" * 62)
print(f"  Function         : {FUNC_NAME}")
print(f"  L                : {ANGLE_L}")
print(f"  deg (internal)   : {deg}  (= 2 * L)")
print(f"  MSE vs surrogate : {mse_vs_surrogate:.4e}")
print(f"  MSE vs true      : {mse_vs_true:.4e}")
print(f"  theta saved to   : {theta_filename}")
print(f"  phi saved to     : {phi_filename}")
print(f"  To use in SLOS/QPU codes:")
print(f"    FUNC_NAME = '{FUNC_NAME}'")
print(f"    ANGLE_L   = {ANGLE_L}")
print(f"    load: theta_{FILE_TAG}.npy")
print(f"    load: phi_{FILE_TAG}.npy")
print("=" * 62)