"""
A-3: Exercise every load balancer endpoint, then kill a server container
directly (bypassing the LB) to prove the heartbeat mechanism detects the
failure and spawns a replacement within HEARTBEAT_INTERVAL seconds.

Usage:
    python3 analysis/a3_endpoint_and_failure_test.py
"""
import time
import subprocess
import requests

LB_URL = "http://localhost:5000"


def test_rep():
    r = requests.get(f"{LB_URL}/rep")
    print("/rep ->", r.status_code, r.json())
    return r.json()["message"]["replicas"]


def test_route_home():
    r = requests.get(f"{LB_URL}/home")
    print("/home ->", r.status_code, r.json())


def test_route_invalid():
    r = requests.get(f"{LB_URL}/other")
    print("/other (invalid) ->", r.status_code, r.json())


def test_add():
    r = requests.post(f"{LB_URL}/add", json={"n": 2, "hostnames": ["S5", "S6"]})
    print("/add ->", r.status_code, r.json())


def test_rm():
    r = requests.delete(f"{LB_URL}/rm", json={"n": 1, "hostnames": ["S5"]})
    print("/rm ->", r.status_code, r.json())


def test_failure_recovery():
    replicas_before = test_rep()
    victim = replicas_before[0]
    print(f"Killing container '{victim}' directly via docker to simulate failure...")
    subprocess.run(["docker", "kill", victim], check=False)

    print("Waiting for heartbeat monitor to detect and recover...")
    for elapsed in range(0, 30, 5):
        time.sleep(5)
        replicas_now = test_rep()
        if victim not in replicas_now and len(replicas_now) == len(replicas_before):
            print(f"Recovered after ~{elapsed + 5}s. New replica set: {replicas_now}")
            return
    print("Recovery not observed within 30s window; check HEARTBEAT_INTERVAL setting.")


if __name__ == "__main__":
    print("=== Testing management endpoints ===")
    test_rep()
    test_add()
    test_rm()

    print("\n=== Testing routing endpoints ===")
    test_route_home()
    test_route_invalid()

    print("\n=== Testing failure recovery ===")
    test_failure_recovery()
