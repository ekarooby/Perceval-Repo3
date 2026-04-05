import perceval as pcvl
from perceval import RemoteProcessor
import re
import numpy as np

# ── FILL IN THESE TWO LINES ──────────────────────────────────
MY_TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA2NTkwNjc1LjE4NjQ3NDN9.eCJCJNcF7BqyUtKiUSaOPK1RM0Nd42B-2UXwH0IIRiYy0TOqi9eQWcUM3049KIkcjgyKkHT-xyIxzoqzmuxyaQ"
JOB_ID   = "109711b5-8495-4b32-8eea-3ac69e174551"
# ─────────────────────────────────────────────────────────────

pcvl.RemoteConfig.set_token(MY_TOKEN)
pcvl.RemoteConfig().save()

remote_processor = RemoteProcessor("qpu:belenos")
remote_job       = remote_processor.resume_job(JOB_ID)
results          = remote_job.get_results()

# --- Show computed circuit (Samuel's guide) ---
computed_circ = results['computed_circuit']
print(f"Total components: {computed_circ.ncomponents()}")
pcvl.pdisplay_to_file(computed_circ, path="computed_circuit.png")
print("Circuit diagram saved to computed_circuit.png")

# --- Extract ALL PS values ---
print(f"\n{'#':<6} {'Mode':<14} {'phi (rad)':<16} {'phi (deg)':<12}")
for i, (modes, comp) in enumerate(computed_circ):
    desc = comp.describe()
    if desc.startswith("PS("):
        m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+)", desc)
        if m:
            phi = float(m.group(1))
            print(f"{i:<6} {str(modes):<14} {phi:<16.6f} {np.degrees(phi):<12.4f}")

# --- Show photon count results ---
print("\nPhoton count results:")
raw = results['results']
total = sum(raw.values())
for state, count in sorted(raw.items(), key=lambda kv: -kv[1]):
    print(f"  {str(state):<20} {count:>8}  ({count/total:.4f})")

# --- Filter PS values for modes 0 and 1 only ---
print("\nPS values for mode 0 and mode 1 only:")
print(f"{'#':<6} {'Mode':<14} {'phi (rad)':<16} {'phi (deg)':<12}")
print("-" * 50)
for i, (modes, comp) in enumerate(computed_circ):
    desc = comp.describe()
    if desc.startswith("PS("):
        if modes == (0,) or modes == (1,):
            m = re.search(r"phi\s*=\s*([\-\+]?[\d\.eE\+\-]+)", desc)
            if m:
                phi = float(m.group(1))
                print(f"{i:<6} {str(modes):<14} {phi:<16.6f} {np.degrees(phi):<12.4f}")

# --- Find which x value this job corresponds to ---
print("\nSearching for job ID in file to identify x value...")
job_id_file = "job_ids_STEP_L180_N5000_x30.txt"
with open(job_id_file, "r") as f:
    for line in f:
        if JOB_ID in line:
            print(f"  Found: {line.strip()}")
            break