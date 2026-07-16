"""
Runs the full Task 4 analysis suite in order.

A-1 and A-2 require the Docker stack to be running (`make up`).
A-3 requires the Docker CLI and a running stack.
A-4 runs fully offline.

Usage:
    python3 analysis/run_all.py
"""
import subprocess
import sys

SCRIPTS = [
    "analysis/a1_load_distribution.py",
    "analysis/a2_scalability.py",
    "analysis/a3_endpoint_and_failure_test.py",
    "analysis/a4_alternate_hash_functions.py",
]

for script in SCRIPTS:
    print(f"\n{'='*70}\nRunning {script}\n{'='*70}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"WARNING: {script} exited with code {result.returncode}")
