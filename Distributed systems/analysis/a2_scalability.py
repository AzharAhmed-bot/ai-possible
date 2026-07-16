"""
A-2: Increment N from 2 to 6. At each step, launch 10,000 requests and
record the average load per server. Plot average load vs N.

Requires the load balancer's /add and /rm endpoints to reshape the
cluster between runs. Run with the stack already up (`make up`).

Usage:
    python3 analysis/a2_scalability.py
"""
import asyncio
import requests
import aiohttp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LB_URL = "http://localhost:5000"
NUM_REQUESTS = 10000
N_VALUES = [2, 3, 4, 5, 6]


def set_replica_count(target_n: int):
    """Scale the cluster up/down to exactly target_n replicas."""
    current = requests.get(f"{LB_URL}/rep", timeout=5).json()["message"]["N"]
    if target_n > current:
        requests.post(f"{LB_URL}/add", json={"n": target_n - current}, timeout=10)
    elif target_n < current:
        requests.delete(f"{LB_URL}/rm", json={"n": current - target_n}, timeout=10)


async def fire_requests(n_requests):
    counts = {}

    async def one(session):
        try:
            async with session.get(f"{LB_URL}/home") as resp:
                data = await resp.json()
                server = data["message"].split("Hello from Server: ")[-1]
                counts[server] = counts.get(server, 0) + 1
        except Exception:  # noqa: BLE001
            pass

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[one(session) for _ in range(n_requests)])
    return counts


def main():
    avg_loads = []
    for n in N_VALUES:
        set_replica_count(n)
        counts = asyncio.run(fire_requests(NUM_REQUESTS))
        total = sum(counts.values())
        avg = total / max(len(counts), 1)
        avg_loads.append(avg)
        print(f"N={n}: per-server counts={counts}, average={avg:.1f}")

    plt.figure(figsize=(8, 5))
    plt.plot(N_VALUES, avg_loads, marker="o", color="#DD8452")
    plt.xlabel("Number of server replicas (N)")
    plt.ylabel("Average requests handled per server")
    plt.title(f"A-2: Scalability ({NUM_REQUESTS} requests per run)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("analysis/a2_scalability.png")
    print("Saved chart to analysis/a2_scalability.png")


if __name__ == "__main__":
    main()
