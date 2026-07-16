"""
app.py (Load Balancer)
-----------------------
Implements the load balancer container described in Task 3 of the
assignment. Responsibilities:

  1. Maintain N server replicas registered in a ConsistentHashMap.
  2. Expose management endpoints: /rep, /add, /rm
  3. Route arbitrary GET requests (/<path>) to the correct replica using
     consistent hashing, and proxy the response back to the client.
  4. Run a background heartbeat monitor that checks each replica's
     /heartbeat endpoint every HEARTBEAT_INTERVAL seconds. If a replica
     stops responding, it is removed from the hash map and a brand-new
     replica (random hostname) is spawned in its place, so that exactly
     N replicas are always available.

Server spawning uses the Docker SDK for Python, talking to the Docker
daemon via the socket mounted into this container
(/var/run/docker.sock), per the assignment's implementation hint.
"""

import os
import random
import string
import threading
import time
import logging

import requests
from flask import Flask, request, jsonify

from consistent_hash import ConsistentHashMap

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:  # pragma: no cover - docker sdk not installed locally
    DOCKER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LB] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------
# Configuration (matches Task 2 defaults; overridable via env vars)
# ---------------------------------------------------------------------
NUM_SLOTS = int(os.environ.get("NUM_SLOTS", 512))
NUM_VIRTUAL_SERVERS = int(os.environ.get("NUM_VIRTUAL_SERVERS", 9))
TARGET_N = int(os.environ.get("N", 3))
DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK", "net1")
SERVER_IMAGE = os.environ.get("SERVER_IMAGE", "lb-server:latest")
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", 5))
HEARTBEAT_TIMEOUT = float(os.environ.get("HEARTBEAT_TIMEOUT", 2))

lock = threading.RLock()
chm = ConsistentHashMap(NUM_SLOTS, NUM_VIRTUAL_SERVERS)

# hostname -> numeric server id (also stored in chm.server_ids, kept here
# for convenience / logging)
next_server_numeric_id = 1

docker_client = docker.from_env() if DOCKER_AVAILABLE else None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _random_hostname() -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"Server_{suffix}"


def _hostname_to_numeric_id(hostname: str) -> int:
    """Deterministically derive a numeric server id `i` (input to PHI)
    from a hostname string, so both preset and randomly generated
    hostnames can be placed on the ring."""
    return sum(ord(c) for c in hostname) % 9973  # arbitrary large prime bound


def _spawn_container(hostname: str) -> bool:
    """Start a new server container with the given hostname via Docker."""
    if not DOCKER_AVAILABLE:
        log.warning("Docker SDK unavailable; skipping actual container spawn for %s", hostname)
        return False
    try:
        docker_client.containers.run(
            SERVER_IMAGE,
            name=hostname,
            hostname=hostname,
            network=DOCKER_NETWORK,
            environment={"SERVER_ID": hostname},
            detach=True,
        )
        log.info("Spawned container %s", hostname)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to spawn container %s: %s", hostname, exc)
        return False


def _stop_container(hostname: str) -> bool:
    if not DOCKER_AVAILABLE:
        log.warning("Docker SDK unavailable; skipping actual container removal for %s", hostname)
        return False
    try:
        c = docker_client.containers.get(hostname)
        c.stop(timeout=3)
        c.remove()
        log.info("Removed container %s", hostname)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to remove container %s: %s", hostname, exc)
        return False


def _register_server(hostname: str, spawn: bool = True) -> None:
    with lock:
        server_id = _hostname_to_numeric_id(hostname)
        chm.add_server(hostname, server_id)
    if spawn:
        _spawn_container(hostname)


def _deregister_server(hostname: str, stop: bool = True) -> None:
    with lock:
        chm.remove_server(hostname)
    if stop:
        _stop_container(hostname)


def _bootstrap_initial_servers() -> None:
    with lock:
        current = len(chm.servers())
    for i in range(current, TARGET_N):
        _register_server(f"Server_{i + 1}", spawn=True)


# ---------------------------------------------------------------------
# Heartbeat / failure-recovery background thread
# ---------------------------------------------------------------------
def _heartbeat_loop():
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        with lock:
            hosts = list(chm.servers())
        for host in hosts:
            try:
                r = requests.get(f"http://{host}:5000/heartbeat", timeout=HEARTBEAT_TIMEOUT)
                alive = r.status_code == 200
            except requests.RequestException:
                alive = False

            if not alive:
                log.warning("Server %s failed heartbeat check; replacing it", host)
                try:
                    _deregister_server(host, stop=True)
                except ValueError:
                    pass  # already removed by a concurrent check
                new_host = _random_hostname()
                _register_server(new_host, spawn=True)


# ---------------------------------------------------------------------
# Management endpoints
# ---------------------------------------------------------------------
@app.route("/rep", methods=["GET"])
def rep():
    with lock:
        replicas = chm.servers()
    return jsonify({
        "message": {
            "N": len(replicas),
            "replicas": replicas,
        },
        "status": "successful",
    }), 200


@app.route("/add", methods=["POST"])
def add():
    payload = request.get_json(silent=True) or {}
    n = payload.get("n")
    hostnames = payload.get("hostnames", [])

    if not isinstance(n, int) or n <= 0:
        return jsonify({"message": "<Error> 'n' must be a positive integer", "status": "failure"}), 400

    if len(hostnames) > n:
        return jsonify({
            "message": "<Error> Length of hostname list is more than newly added instances",
            "status": "failure",
        }), 400

    new_hosts = list(hostnames)
    while len(new_hosts) < n:
        candidate = _random_hostname()
        with lock:
            existing = set(chm.servers())
        if candidate not in existing and candidate not in new_hosts:
            new_hosts.append(candidate)

    for host in new_hosts:
        _register_server(host, spawn=True)

    with lock:
        replicas = chm.servers()

    return jsonify({
        "message": {"N": len(replicas), "replicas": replicas},
        "status": "successful",
    }), 200


@app.route("/rm", methods=["DELETE"])
def rm():
    payload = request.get_json(silent=True) or {}
    n = payload.get("n")
    hostnames = payload.get("hostnames", [])

    if not isinstance(n, int) or n <= 0:
        return jsonify({"message": "<Error> 'n' must be a positive integer", "status": "failure"}), 400

    if len(hostnames) > n:
        return jsonify({
            "message": "<Error> Length of hostname list is more than removable instances",
            "status": "failure",
        }), 400

    with lock:
        current = chm.servers()

    to_remove = list(hostnames)
    remaining_pool = [h for h in current if h not in to_remove]
    while len(to_remove) < n and remaining_pool:
        pick = random.choice(remaining_pool)
        remaining_pool.remove(pick)
        to_remove.append(pick)

    for host in to_remove:
        try:
            _deregister_server(host, stop=True)
        except ValueError:
            continue

    with lock:
        replicas = chm.servers()

    return jsonify({
        "message": {"N": len(replicas), "replicas": replicas},
        "status": "successful",
    }), 200


# ---------------------------------------------------------------------
# Request routing endpoint
# ---------------------------------------------------------------------
@app.route("/<path:endpoint>", methods=["GET"])
def route_request(endpoint):
    request_id = random.randint(100000, 999999)

    with lock:
        if not chm.servers():
            return jsonify({"message": "<Error> No servers available", "status": "failure"}), 503
        target_host = chm.get_server(request_id)

    try:
        resp = requests.get(f"http://{target_host}:5000/{endpoint}", timeout=5)
        return (resp.content, resp.status_code, {"Content-Type": resp.headers.get("Content-Type", "application/json")})
    except requests.RequestException:
        return jsonify({
            "message": f"<Error> '/{endpoint}' endpoint does not exist in server replicas",
            "status": "failure",
        }), 400


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Load balancer is running", "status": "successful"}), 200


if __name__ == "__main__":
    _bootstrap_initial_servers()
    t = threading.Thread(target=_heartbeat_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
