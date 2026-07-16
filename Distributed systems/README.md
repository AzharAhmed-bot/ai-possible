# Customizable Load Balancer — ICS 4104 Distributed Systems, Assignment 1

## 1. Overview

This project implements a **customizable load balancer** that distributes
asynchronous client requests across a set of replicated web server
containers using **consistent hashing**, as specified in the assignment
brief. It has three moving parts:

| Component | Description |
|---|---|
| `server/` | A minimal Flask web server exposing `/home` and `/heartbeat`. Each replica is identified by a `SERVER_ID` env var. |
| `load_balancer/` | A Flask service that (a) maintains a consistent-hash ring of server replicas, (b) exposes `/rep`, `/add`, `/rm` for cluster management, (c) routes arbitrary `GET /<path>` requests to a replica via the ring, and (d) runs a background heartbeat monitor that detects failed replicas and spawns replacements so `N` replicas are always available. |
| `analysis/` | Scripts that reproduce the Task 4 experiments (load distribution, scalability, failure recovery, alternate hash functions) and save the resulting charts. |

Both containers run inside a single Docker bridge network (`net1`); only
the load balancer's port `5000` is published to the host, matching Fig. 1
of the assignment.

## 2. Design choices & assumptions

- **Hash functions.** Implements exactly the functions given in the
  spec: `H(i) = i² + 2i + 17 (mod M)` for request routing and
  `Φ(i, j) = i² + j² + 2j + 25 (mod M)` for virtual-server placement,
  with `M = 512` slots and `K = 9` virtual servers per replica (`log₂512`).
  Collisions are resolved with **linear probing** (scan clockwise for the
  next empty/occupied slot), as suggested in the assignment hint.
- **Numeric server IDs.** The hash functions need an integer server ID
  `i`. Since hostnames can be arbitrary strings (user-supplied via
  `/add`), we derive a deterministic integer from the hostname
  (`sum(ord(c) for c in hostname) % 9973`) so both preset and randomly
  generated hostnames can be placed on the ring reproducibly.
- **Hostnames use underscores, not spaces** (e.g. `Server_1` rather than
  `Server 1` from the spec's illustrative examples), because Docker
  container/hostnames cannot contain spaces.
- **Request IDs** are generated as random 6-digit integers per incoming
  request, per the assignment's stated assumption in Appendix B.
- **Container orchestration.** The load balancer is a privileged
  container with the host's Docker socket mounted in, and uses the
  **Docker SDK for Python** to spawn/stop replica containers directly
  (rather than shelling out to the `docker` CLI), per the assignment's
  implementation hint.
- **Failure detection.** A background thread polls every replica's
  `/heartbeat` every `HEARTBEAT_INTERVAL` seconds (default 5s, 2s
  timeout). A non-200/timeout response is treated as failure: the
  replica is deregistered from the ring, its container is stopped and
  removed, and a new replica with a randomly generated hostname is
  spawned and registered — keeping the replica count at exactly `N`.
- **Thread safety.** All ring mutations (`add_server`/`remove_server`)
  and reads are guarded by a single `RLock`, since Flask serves requests
  concurrently and the heartbeat thread runs in the background.

## 3. Repository structure

```
.
├── server/                 # Task 1: minimal replica web server
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── load_balancer/          # Task 2 + 3: consistent hashing + load balancer
│   ├── app.py
│   ├── consistent_hash.py
│   ├── Dockerfile
│   └── requirements.txt
├── analysis/                # Task 4: experiments & generated charts
│   ├── a1_load_distribution.py
│   ├── a2_scalability.py
│   ├── a3_endpoint_and_failure_test.py
│   ├── a4_alternate_hash_functions.py
│   └── run_all.py
├── tests/                   # Unit tests (no Docker required)
│   └── test_consistent_hash.py
├── docker-compose.yml
├── Makefile
├── requirements.txt          # for running tests/analysis from the host
└── README.md
```

## 4. Installation / dependencies

**Requirements:**
- Ubuntu 20.04+ (or any Docker-capable Linux host)
- Docker Engine ≥ 20.10.23
- Docker Compose ≥ v2.15.1
- Python 3.11+ (for running tests/analysis from the host)

Install Docker (see `Makefile`/assignment Appendix A for the exact apt
commands), then, from the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Deployment instructions

```bash
# Build both images and start the load balancer (which spawns N=3 replicas)
make up

# Check status
curl http://localhost:5000/rep

# Tear everything down (including any spawned replica containers)
make down
```

Configuration (`N`, `NUM_SLOTS`, `NUM_VIRTUAL_SERVERS`, heartbeat timing)
is set via environment variables in `docker-compose.yml` and can be
overridden there or with `docker-compose run -e N=5 ...`.

## 6. Usage / API

| Endpoint | Method | Purpose |
|---|---|---|
| `/rep` | GET | List current replica count and hostnames |
| `/add` | POST | Add `n` new replicas, optionally with preferred `hostnames` |
| `/rm` | DELETE | Remove `n` replicas, optionally specifying `hostnames` |
| `/<path>` | GET | Routed to a replica via consistent hashing (e.g. `/home`) |

Example:

```bash
curl http://localhost:5000/home
curl -X POST http://localhost:5000/add -H "Content-Type: application/json" \
     -d '{"n": 2, "hostnames": ["S5", "S4"]}'
curl -X DELETE http://localhost:5000/rm -H "Content-Type: application/json" \
     -d '{"n": 1, "hostnames": ["S5"]}'
```

## 7. Testing

Unit tests cover the consistent-hash map's core correctness properties
(virtual-slot placement, collision-free insertion, clean removal,
deterministic routing, minimal remapping on scale-out) and do **not**
require Docker.

```bash
make test
# or directly:
python3 -m pytest tests/ -v
```

Integration/behavioral testing of the live system (all endpoints +
failure recovery) is done by `analysis/a3_endpoint_and_failure_test.py`
against a running stack (see below).

## 8. Task 4: Analysis

With the stack running (`make up`), reproduce all four experiments:

```bash
make analysis
# or individually, e.g.:
python3 analysis/a1_load_distribution.py
```

- **A-1** (`a1_load_distribution.py`): fires 10,000 async requests at
  `N=3` and bar-charts the per-server load.
- **A-2** (`a2_scalability.py`): scales `N` from 2→6 via `/add`/`/rm`,
  firing 10,000 requests at each step and line-charting average load.
- **A-3** (`a3_endpoint_and_failure_test.py`): exercises every endpoint,
  then `docker kill`s a replica directly and confirms the heartbeat
  monitor detects and replaces it within the heartbeat window.
- **A-4** (`a4_alternate_hash_functions.py`, runs fully offline): compares
  the assignment's quadratic `H`/`Φ` against a multiplicative alternative.

**Observed result (A-1/A-4, N=3, 10,000 requests):** the assignment's
mandated hash functions produce a *highly skewed* distribution
(≈8,400 / 1,100 / 480 in one run) because `i² + 2i + 17 mod 512` is far
from uniform over 6-digit inputs — it clusters into a small number of
residues. Swapping in a multiplicative hash (`i * 2654435761 mod 512`)
gives a much more balanced split (≈2,600 / 4,000 / 3,300), confirming
that the *quality of the hash function*, not the consistent-hashing
structure itself, is the main driver of load imbalance here. Charts
generated by these scripts are saved as PNGs in `analysis/` for
inclusion in the submission.

## 9. Video recording

Per the submission requirements, record a short screen capture showing:
`make up` → `/rep` → a routed request → `/add`/`/rm` → killing a
container and watching the load balancer recover it → one of the
analysis charts being generated.

## 10. Grading component mapping

- Task 1 (Server): `server/`
- Task 2 (Consistent Hashing): `load_balancer/consistent_hash.py`, `tests/`
- Task 3 (Load Balancer): `load_balancer/app.py`, `docker-compose.yml`, `Makefile`
- Task 4 (Analysis): `analysis/`
