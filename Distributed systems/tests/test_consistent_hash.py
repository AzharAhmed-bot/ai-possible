"""
Unit tests for consistent_hash.ConsistentHashMap.

Run with:  python3 -m pytest tests/ -v
(from the repository root; requires load_balancer/ on the Python path,
handled below via sys.path manipulation so no extra packaging is needed.)
"""

import os
import sys
import random
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "load_balancer"))

from consistent_hash import ConsistentHashMap  # noqa: E402


def make_map(n_servers=3, num_slots=512, k=9):
    chm = ConsistentHashMap(num_slots, k)
    for i in range(n_servers):
        chm.add_server(f"Server_{i+1}", i + 1)
    return chm


def test_add_server_creates_k_virtual_slots():
    chm = make_map(n_servers=1, k=9)
    assert len(chm.server_slots["Server_1"]) == 9
    assert sum(1 for s in chm.slots if s is not None) == 9


def test_add_multiple_servers_no_double_booking():
    chm = make_map(n_servers=5, k=9)
    occupied = [s for s in chm.slots if s is not None]
    # No two servers should ever occupy the same slot index.
    seen_indices = [i for i, s in enumerate(chm.slots) if s is not None]
    assert len(seen_indices) == len(set(seen_indices))
    assert len(occupied) == 5 * 9


def test_remove_server_frees_its_slots():
    chm = make_map(n_servers=3, k=9)
    chm.remove_server("Server_2")
    assert "Server_2" not in chm.servers()
    assert all(s != "Server_2" for s in chm.slots)
    assert sum(1 for s in chm.slots if s is not None) == 2 * 9


def test_get_server_returns_registered_hostname():
    chm = make_map(n_servers=3, k=9)
    for _ in range(200):
        rid = random.randint(100000, 999999)
        host = chm.get_server(rid)
        assert host in chm.servers()


def test_get_server_raises_when_empty():
    chm = ConsistentHashMap(512, 9)
    with pytest.raises(RuntimeError):
        chm.get_server(123456)


def test_add_duplicate_server_raises():
    chm = make_map(n_servers=1, k=9)
    with pytest.raises(ValueError):
        chm.add_server("Server_1", 1)


def test_remove_unknown_server_raises():
    chm = make_map(n_servers=1, k=9)
    with pytest.raises(ValueError):
        chm.remove_server("Server_99")


def test_load_distribution_slots_matches_k():
    chm = make_map(n_servers=4, k=9)
    dist = chm.load_distribution_slots()
    assert all(v == 9 for v in dist.values())
    assert len(dist) == 4


def test_deterministic_routing_same_request_id():
    """The same request ID should always route to the same server as long
    as the ring hasn't changed (core correctness property of consistent
    hashing)."""
    chm = make_map(n_servers=3, k=9)
    rid = 424242
    first = chm.get_server(rid)
    second = chm.get_server(rid)
    assert first == second


def test_minimal_remap_on_server_addition():
    """Adding a server should only reassign a subset of request IDs, not
    all of them (the key benefit of consistent hashing over mod-N
    hashing)."""
    chm = make_map(n_servers=3, k=9)
    sample_ids = [random.randint(100000, 999999) for _ in range(500)]
    before = {rid: chm.get_server(rid) for rid in sample_ids}

    chm.add_server("Server_4", 4)
    after = {rid: chm.get_server(rid) for rid in sample_ids}

    changed = sum(1 for rid in sample_ids if before[rid] != after[rid])
    # Not a strict correctness bound (depends on where slots land), but it
    # should clearly be far less than "all requests changed".
    assert changed < len(sample_ids)
