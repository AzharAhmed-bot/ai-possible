"""
A-4: Re-run the A-1 / A-2 style experiments using a different, more
uniform hash function, and compare against the assignment-mandated
H(i) = i^2 + 2i + 17 and PHI(i,j) = i^2 + j^2 + 2j + 25.

This script works fully offline (no running stack needed) by directly
exercising ConsistentHashMap with a monkey-patched hash function, which
is the easiest way to isolate the effect of the hash function itself
from network/Docker variability.

Usage:
    python3 analysis/a4_alternate_hash_functions.py
"""
import os
import sys
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "load_balancer"))
from consistent_hash import ConsistentHashMap  # noqa: E402


def run_experiment(H_fn, PHI_fn, label, n_servers=3, num_slots=512, k=9, n_requests=10000):
    chm = ConsistentHashMap(num_slots, k)
    chm.H = H_fn.__get__(chm)
    chm.PHI = PHI_fn.__get__(chm)

    for i in range(n_servers):
        chm.add_server(f"Server_{i+1}", i + 1)

    counts = Counter()
    for _ in range(n_requests):
        rid = random.randint(100000, 999999)
        counts[chm.get_server(rid)] += 1

    print(f"[{label}] counts = {dict(counts)}")
    return counts


# Original (assignment-mandated) hash functions
def H_original(self, i):
    return (i ** 2 + 2 * i + 17) % self.num_slots


def PHI_original(self, i, j):
    return (i ** 2 + j ** 2 + 2 * j + 25) % self.num_slots


# Alternative: simple multiplicative hash (Knuth-style), much more
# uniform across the ring than the quadratic function above.
def H_alt(self, i):
    return (i * 2654435761) % self.num_slots


def PHI_alt(self, i, j):
    return ((i * 2654435761) ^ (j * 40503)) % self.num_slots


def main():
    orig = run_experiment(H_original, PHI_original, "original")
    alt = run_experiment(H_alt, PHI_alt, "alternative")

    servers = sorted(set(orig) | set(alt))
    orig_vals = [orig.get(s, 0) for s in servers]
    alt_vals = [alt.get(s, 0) for s in servers]

    x = range(len(servers))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], orig_vals, width, label="Original H/PHI")
    plt.bar([i + width / 2 for i in x], alt_vals, width, label="Alternative H/PHI")
    plt.xticks(list(x), servers)
    plt.ylabel("Requests handled (of 10,000)")
    plt.title("A-4: Original vs. alternative hash functions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("analysis/a4_hash_function_comparison.png")
    print("Saved chart to analysis/a4_hash_function_comparison.png")


if __name__ == "__main__":
    main()
