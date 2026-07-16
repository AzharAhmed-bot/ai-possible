"""
consistent_hash.py
-------------------
Implements the circular consistent-hash map used by the load balancer to
route client requests to server replicas, as specified in Appendix B of
the assignment.

Hash functions (as given in the assignment spec):
    Request mapping:      H(i)    = i^2 + 2*i + 17      (mod M)
    Virtual server mapping: PHI(i, j) = i^2 + j^2 + 2*j + 25  (mod M)

Where:
    i = request ID (for H) or physical server ID (for PHI)
    j = virtual server replica index (0..K-1)
    M = total number of slots in the hash map (default 512)
    K = number of virtual servers per physical server (default log2(M) = 9)

Collisions are resolved with linear probing: if the target slot is already
occupied, we scan clockwise (increasing index, wrapping around) until an
empty slot is found.
"""

from typing import Optional, Dict, List


class ConsistentHashMap:
    def __init__(self, num_slots: int = 512, num_virtual_servers: int = 9):
        self.num_slots = num_slots
        self.num_virtual_servers = num_virtual_servers

        # slots[k] = hostname occupying slot k, or None if empty
        self.slots: List[Optional[str]] = [None] * num_slots

        # Track which slots belong to which physical server so we can
        # remove them cleanly on server failure/removal.
        # hostname -> list of occupied slot indices
        self.server_slots: Dict[str, List[int]] = {}

        # hostname -> numeric server id used as input `i` to PHI
        self.server_ids: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Hash functions
    # ------------------------------------------------------------------
    def H(self, request_id: int) -> int:
        """Request-mapping hash function: H(i) = i^2 + 2i + 17 (mod M)."""
        return (request_id ** 2 + 2 * request_id + 17) % self.num_slots

    def PHI(self, server_id: int, replica_id: int) -> int:
        """Virtual-server hash function: PHI(i,j) = i^2 + j^2 + 2j + 25 (mod M)."""
        return (server_id ** 2 + replica_id ** 2 + 2 * replica_id + 25) % self.num_slots

    # ------------------------------------------------------------------
    # Probing helper
    # ------------------------------------------------------------------
    def _find_empty_slot(self, start: int) -> int:
        """Linear probing: scan clockwise from `start` for an empty slot.

        Raises RuntimeError if the map is completely full (should not
        normally happen since num_slots >> N * K for reasonable N).
        """
        idx = start % self.num_slots
        for _ in range(self.num_slots):
            if self.slots[idx] is None:
                return idx
            idx = (idx + 1) % self.num_slots
        raise RuntimeError("Consistent hash map is full; cannot place server")

    def _find_occupied_slot_clockwise(self, start: int) -> int:
        """Scan clockwise from `start` for the nearest occupied slot.

        Used to route a request: if the request's own slot has no server,
        it is served by the next server found going clockwise.
        """
        idx = start % self.num_slots
        for _ in range(self.num_slots):
            if self.slots[idx] is not None:
                return idx
            idx = (idx + 1) % self.num_slots
        raise RuntimeError("No servers registered in the consistent hash map")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_server(self, hostname: str, server_id: int) -> None:
        """Place K virtual replicas of `hostname` (numeric id `server_id`)
        into the ring."""
        if hostname in self.server_slots:
            raise ValueError(f"Server {hostname} already present in hash map")

        occupied = []
        for j in range(self.num_virtual_servers):
            target = self.PHI(server_id, j)
            slot = self._find_empty_slot(target)
            self.slots[slot] = hostname
            occupied.append(slot)

        self.server_slots[hostname] = occupied
        self.server_ids[hostname] = server_id

    def remove_server(self, hostname: str) -> None:
        """Remove all virtual replicas belonging to `hostname`."""
        if hostname not in self.server_slots:
            raise ValueError(f"Server {hostname} not present in hash map")

        for slot in self.server_slots[hostname]:
            self.slots[slot] = None

        del self.server_slots[hostname]
        del self.server_ids[hostname]

    def get_server(self, request_id: int) -> str:
        """Return the hostname that should serve `request_id`."""
        start = self.H(request_id)
        slot = self._find_occupied_slot_clockwise(start)
        return self.slots[slot]

    def servers(self) -> List[str]:
        return list(self.server_slots.keys())

    def load_distribution_slots(self) -> Dict[str, int]:
        """Number of ring slots currently owned by each server (diagnostic)."""
        return {h: len(slots) for h, slots in self.server_slots.items()}
