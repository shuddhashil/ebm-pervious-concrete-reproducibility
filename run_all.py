"""Run every analysis in order, then compare the results with the values in the paper.
Pass --validation-only to run only the validation step (about 15-25 minutes on 4-6 cores)."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("validation", "run_validation.py", []),
    ("flyash analysis/src", "run_flyash_bridge_analysis.py", ["--stability-only"]),
    ("flyash analysis/resampled_perturbation/src", "run_resampled_fa_perturbation.py", []),
    ("08_analysis_code_suite", "analysis_02_optimizer_stability.py", []),
    ("robustness_verification/src", "robustness_verification.py", []),
    ("robustness_verification/robustness_2_reconciliation/src", "final_reconciliation.py", []),
]
if "--validation-only" in sys.argv:
    STEPS = STEPS[:1]
for folder, script, args in STEPS:
    print(f"\n=== {folder}/{script}", flush=True)
    subprocess.run([sys.executable, script, *args], cwd=ROOT / folder, check=True)
subprocess.run([sys.executable, "check_results.py"], cwd=ROOT, check=True)
