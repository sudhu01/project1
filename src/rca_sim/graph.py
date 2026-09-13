"""Dependency graph construction and queries.

Edges point from a caller to the dependency it calls. Public service IDs are
row and column indices in ``adjacency``, where ``adjacency[u, v]`` means that
service ``u`` calls service ``v``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DependencyGraph:
    """A rooted DAG whose edges point from callers to dependencies.

    Arrays are read-only so cached distances and degrees cannot drift away
    from the adjacency matrix.
    """

    adjacency: BoolArray
    entry_service: int
    distances: IntArray
    in_degree: IntArray
    out_degree: IntArray

    @classmethod
    def from_edges(
        cls,
        n_services: int,
        edges: Iterable[Edge],
        *,
        entry_service: int,
    ) -> DependencyGraph:
        """Build and validate a rooted dependency DAG from public IDs."""
        if isinstance(n_services, bool) or not isinstance(n_services, Integral):
            raise TypeError("n_services must be an integer")
        n_services = int(n_services)
        if n_services < 1:
            raise ValueError("n_services must be at least 1")
        if isinstance(entry_service, bool) or not isinstance(entry_service, Integral):
            raise TypeError("entry_service must be an integer")
        entry_service = int(entry_service)
        if not 0 <= entry_service < n_services:
            raise ValueError("entry_service must be a valid service ID")

        adjacency = np.zeros((n_services, n_services), dtype=np.bool_)
        for edge in edges:
            try:
                caller, dependency = edge
            except (TypeError, ValueError) as error:
                raise ValueError("each edge must contain exactly two service IDs") from error
            if isinstance(caller, bool) or not isinstance(caller, Integral):
                raise TypeError("edge service IDs must be integers")
            if isinstance(dependency, bool) or not isinstance(dependency, Integral):
                raise TypeError("edge service IDs must be integers")
            caller = int(caller)
            dependency = int(dependency)
            if not 0 <= caller < n_services or not 0 <= dependency < n_services:
                raise ValueError("edge contains an invalid service ID")
            if caller == dependency:
                raise ValueError("self-edges are not allowed")
            adjacency[caller, dependency] = True

        if not _is_acyclic(adjacency):
            raise ValueError("dependency graph must be acyclic")

        distances = _all_pairs_shortest_path_distances(adjacency)
        if np.any(distances[entry_service] < 0):
            raise ValueError("every service must be reachable from the entry service")

        in_degree = adjacency.sum(axis=0, dtype=np.int64)
        out_degree = adjacency.sum(axis=1, dtype=np.int64)
        for array in (adjacency, distances, in_degree, out_degree):
            array.flags.writeable = False

        return cls(
            adjacency=adjacency,
            entry_service=entry_service,
            distances=distances,
            in_degree=in_degree,
            out_degree=out_degree,
        )

    @property
    def n_services(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Return edges in stable public-ID order."""
        callers, dependencies = np.nonzero(self.adjacency)
        return tuple(zip(callers.tolist(), dependencies.tolist(), strict=True))

    def shortest_path_distance(self, caller: int, dependency: int) -> int | None:
        """Return directed hop distance, or ``None`` when no path exists."""
        self._validate_service_id(caller)
        self._validate_service_id(dependency)
        distance = int(self.distances[caller, dependency])
        return None if distance < 0 else distance

    def affected_callers(self, failed_service: int) -> tuple[int, ...]:
        """Return services that can carry symptoms from ``failed_service``.

        The failed service itself is excluded. A caller is affected exactly
        when it has a directed path to the failed dependency.
        """
        self._validate_service_id(failed_service)
        callers = np.flatnonzero(self.distances[:, failed_service] > 0)
        return tuple(int(caller) for caller in callers)

    def _validate_service_id(self, service: int) -> None:
        if isinstance(service, bool) or not isinstance(service, Integral):
            raise TypeError("service ID must be an integer")
        service = int(service)
        if not 0 <= service < self.n_services:
            raise ValueError("service ID is out of range")


def generate_dependency_graph(
    n_services: int,
    rng: np.random.Generator,
    *,
    extra_edge_probability: float = 0.15,
) -> DependencyGraph:
    """Generate a reachable DAG and randomly relabel its public service IDs."""
    if isinstance(n_services, bool) or not isinstance(n_services, Integral):
        raise TypeError("n_services must be an integer")
    n_services = int(n_services)
    if n_services < 1:
        raise ValueError("n_services must be at least 1")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if isinstance(extra_edge_probability, bool) or not isinstance(
        extra_edge_probability, Real
    ):
        raise TypeError("extra_edge_probability must be a real number")
    extra_edge_probability = float(extra_edge_probability)
    if not np.isfinite(extra_edge_probability) or not 0 <= extra_edge_probability <= 1:
        raise ValueError("extra_edge_probability must be between 0 and 1")

    hidden_adjacency = np.zeros((n_services, n_services), dtype=np.bool_)

    # Each non-entry service gets one uniformly sampled earlier caller. These
    # edges alone make every service reachable from hidden service 0.
    for dependency in range(1, n_services):
        parent = int(rng.integers(0, dependency))
        hidden_adjacency[parent, dependency] = True

    # All other forward pairs receive independent Bernoulli trials.
    for caller in range(n_services):
        for dependency in range(caller + 1, n_services):
            if not hidden_adjacency[caller, dependency]:
                hidden_adjacency[caller, dependency] = (
                    rng.random() < extra_edge_probability
                )

    # permutation[hidden_id] is its public ID. Applying it to both axes keeps
    # every edge and the entry identity aligned while erasing topological order.
    permutation = rng.permutation(n_services)
    public_adjacency = np.zeros_like(hidden_adjacency)
    hidden_callers, hidden_dependencies = np.nonzero(hidden_adjacency)
    public_adjacency[
        permutation[hidden_callers], permutation[hidden_dependencies]
    ] = True

    public_callers, public_dependencies = np.nonzero(public_adjacency)
    public_edges = zip(
        public_callers.tolist(),
        public_dependencies.tolist(),
        strict=True,
    )
    return DependencyGraph.from_edges(
        n_services,
        public_edges,
        entry_service=int(permutation[0]),
    )


def _all_pairs_shortest_path_distances(adjacency: BoolArray) -> IntArray:
    """Compute directed unweighted distances with BFS from every service."""
    n_services = adjacency.shape[0]
    distances = np.full((n_services, n_services), -1, dtype=np.int64)

    for source in range(n_services):
        distances[source, source] = 0
        queue = [source]
        next_index = 0
        while next_index < len(queue):
            caller = queue[next_index]
            next_index += 1
            for dependency in np.flatnonzero(adjacency[caller]):
                dependency_id = int(dependency)
                if distances[source, dependency_id] < 0:
                    distances[source, dependency_id] = (
                        distances[source, caller] + 1
                    )
                    queue.append(dependency_id)

    return distances


def _is_acyclic(adjacency: BoolArray) -> bool:
    """Check acyclicity with Kahn's algorithm."""
    in_degree = adjacency.sum(axis=0, dtype=np.int64)
    queue = [int(node) for node in np.flatnonzero(in_degree == 0)]
    visited = 0
    next_index = 0

    while next_index < len(queue):
        caller = queue[next_index]
        next_index += 1
        visited += 1
        for dependency in np.flatnonzero(adjacency[caller]):
            dependency_id = int(dependency)
            in_degree[dependency_id] -= 1
            if in_degree[dependency_id] == 0:
                queue.append(dependency_id)

    return visited == adjacency.shape[0]
