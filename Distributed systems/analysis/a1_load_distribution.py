"""
A-1: Launch 10,000 async requests against a running load balancer (N=3)
and plot the number of requests each server handled.

Usage (with the stack already running via `make up`):
    python3 analysis/a1_load_distribution.py
"""
import asyncio
import sys
from collections import Counter

import aiohttp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LB_URL = "http://localhost:5000"
NUM_REQUESTS = 10000


async def fire_request(session, counter):
    try:
        async with session.get(f"{LB_URL}/home") as resp:
            data = await resp.json()
            server = data["message"].split("Hello from Server: ")[-1]
            counter[server] += 1
    except Exception as exc:  # noqa: BLE001
        counter["__error__"] += 1


async def main():
    counter = Counter()
    async with aiohttp.ClientSession() as session:
        tasks = [fire_request(session, counter) for _ in range(NUM_REQUESTS)]
        await asyncio.gather(*tasks)

    print("Result counts:", dict(counter))

    servers = [k for k in counter if k != "__error__"]
    counts = [counter[k] for k in servers]

    plt.figure(figsize=(8, 5))
    plt.bar(servers, counts, color="#4C72B0")
    plt.xlabel("Server")
    plt.ylabel("Requests handled")
    plt.title(f"A-1: Load distribution across N=3 servers ({NUM_REQUESTS} requests)")
    plt.tight_layout()
    plt.savefig("analysis/a1_load_distribution.png")
    print("Saved chart to analysis/a1_load_distribution.png")


if __name__ == "__main__":
    asyncio.run(main())
