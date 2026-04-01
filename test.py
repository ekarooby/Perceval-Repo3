import numpy as np
import warnings
from paddle_quantum.qpp.laurent import laurent_generator, pair_generation, Laurent
from paddle_quantum.qpp.angles import qpp_angle_approximator

fn = lambda x: (2.0 / np.pi) * np.arctan(100 * x)
L  = 70
deg = 2 * L

F_raw = laurent_generator(fn, 0.01, deg, np.pi)
print(f"Raw max_norm: {F_raw.max_norm:.4f}")

# Try different scale factors
for scale_target in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
    F = Laurent(F_raw.coef * (scale_target / F_raw.max_norm))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P, Q = pair_generation(F)
    print(f"  scale={scale_target:.2f}  Q.max_norm={Q.max_norm:.4f}", end="")
    if Q.max_norm < 2.0:
        print("  <-- STABLE, trying angles...")
        try:
            list_theta, list_phi = qpp_angle_approximator(P, Q)
            print(f"  SUCCESS! L={len(list_theta)-1}")
            break
        except Exception as e:
            print(f"  angles failed: {e}")
    else:
        print("  unstable")