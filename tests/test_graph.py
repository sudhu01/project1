"""Tests for dependency graph behavior."""

import numpy as np
import pytest

from rca_sim.graph import DependencyGraph, generate_dependency_graph


def test_chain_distances_degrees_and_propagation_direction() -> None:
    graph = DependencyGraph.from_edges(
        4,
        [(0, 1), (1, 2), (2, 3)],
        entry_service=0,
    )

    assert graph.edges == ((0, 1), (1, 2), (2, 3))
    assert graph.shortest_path_distance(0, 3) == 3
    assert graph.shortest_path_distance(3, 0) is None
    assert graph.in_degree.tolist() == [0, 1, 1, 1]
    assert graph.out_degree.tolist() == [1, 1, 1, 0]
    assert graph.affected_callers(3) == (0, 1, 2)
    assert graph.affected_callers(1) == (0,)
    assert graph.affected_callers(0) == ()


def test_branching_tree_distances() -> None:
    graph = DependencyGraph.from_edges(
        5,
        [(0, 1), (0, 2), (1, 3), (1, 4)],
        entry_service=0,
    )

    assert graph.distances.tolist() == [
        [0, 1, 1, 2, 2],
        [-1, 0, -1, 1, 1],
        [-1, -1, 0, -1, -1],
        [-1, -1, -1, 0, -1],
        [-1, -1, -1, -1, 0],
    ]
    assert graph.in_degree.tolist() == [0, 1, 1, 1, 1]
    assert graph.out_degree.tolist() == [2, 2, 0, 0, 0]


def test_shared_dependency_uses_shortest_directed_path() -> None:
    graph = DependencyGraph.from_edges(
        5,
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (0, 4)],
        entry_service=0,
    )

    assert graph.shortest_path_distance(0, 3) == 2
    assert graph.shortest_path_distance(0, 4) == 1
    assert graph.affected_callers(3) == (0, 1, 2)
    assert graph.in_degree.tolist() == [0, 1, 1, 2, 2]


@pytest.mark.parametrize("seed", range(20))
def test_generator_returns_a_reachable_relabelled_dag(seed: int) -> None:
    n_services = 8
    graph = generate_dependency_graph(n_services, np.random.default_rng(seed))

    assert graph.n_services == n_services
    assert graph.adjacency.dtype == np.bool_
    assert not np.any(np.diag(graph.adjacency))
    assert np.all(graph.distances[graph.entry_service] >= 0)
    assert len(graph.edges) >= n_services - 1

    remaining_in_degree = graph.in_degree.copy()
    queue = list(np.flatnonzero(remaining_in_degree == 0))
    visited = 0
    while queue:
        caller = int(queue.pop())
        visited += 1
        for dependency in np.flatnonzero(graph.adjacency[caller]):
            remaining_in_degree[dependency] -= 1
            if remaining_in_degree[dependency] == 0:
                queue.append(int(dependency))
    assert visited == n_services


def test_generator_is_seed_reproducible() -> None:
    first = generate_dependency_graph(8, np.random.default_rng(1234))
    second = generate_dependency_graph(8, np.random.default_rng(1234))

    np.testing.assert_array_equal(first.adjacency, second.adjacency)
    np.testing.assert_array_equal(first.distances, second.distances)
    assert first.entry_service == second.entry_service


def test_graph_queries_accept_numpy_integer_ids() -> None:
    graph = DependencyGraph.from_edges(
        np.int64(2),
        [(np.int64(0), np.int64(1))],
        entry_service=np.int64(0),
    )

    assert graph.shortest_path_distance(np.int64(0), np.int64(1)) == 1


def test_permutation_hides_entry_service_index() -> None:
    entry_ids = {
        generate_dependency_graph(8, np.random.default_rng(seed)).entry_service
        for seed in range(100)
    }

    assert entry_ids == set(range(8))


def test_extra_edge_probability_extremes() -> None:
    sparse = generate_dependency_graph(
        8,
        np.random.default_rng(7),
        extra_edge_probability=0.0,
    )
    dense = generate_dependency_graph(
        8,
        np.random.default_rng(7),
        extra_edge_probability=1.0,
    )

    assert len(sparse.edges) == 7
    assert len(dense.edges) == 28


@pytest.mark.parametrize(
    ("n_services", "probability"),
    [
        (0, 0.15),
        (4, -0.01),
        (4, 1.01),
        (4, float("nan")),
    ],
)
def test_generator_rejects_invalid_parameters(
    n_services: int,
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        generate_dependency_graph(
            n_services,
            np.random.default_rng(0),
            extra_edge_probability=probability,
        )


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([(0, 1), (1, 0)], "acyclic"),
        ([(0, 1), (2, 3)], "reachable"),
        ([(0, 0)], "self-edges"),
        ([(0, 4)], "invalid service ID"),
    ],
)
def test_from_edges_rejects_invalid_graphs(
    edges: list[tuple[int, int]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DependencyGraph.from_edges(4, edges, entry_service=0)


def test_graph_arrays_are_read_only() -> None:
    graph = DependencyGraph.from_edges(2, [(0, 1)], entry_service=0)

    with pytest.raises(ValueError):
        graph.adjacency[0, 1] = False
    with pytest.raises(ValueError):
        graph.distances[0, 1] = 9
