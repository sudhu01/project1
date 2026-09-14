"""Tests for policy-visible observation construction."""

import numpy as np
import pytest

from rca_sim.belief import ExactBeliefEstimator
from rca_sim.contracts import LogCategory, LogEvidenceRecord, MetricCategory, MetricEvidenceRecord
from rca_sim.graph import DependencyGraph
from rca_sim.observation import (
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    OBSERVATION_SCHEMA_VERSION,
    build_observation,
    make_observation_space,
)
from rca_sim.tools import ProbeTool, probe_action_index


def _graph() -> DependencyGraph:
    return DependencyGraph.from_edges(3, [(0, 1), (1, 2)], entry_service=0)


def _build(**overrides: object) -> dict[str, np.ndarray]:
    graph = overrides.pop("graph", _graph())
    belief = overrides.pop("belief", ExactBeliefEstimator(graph))
    values = {
        "graph": graph,
        "belief": belief,
        "evidence_records": (),
        "executed_probe_actions": (),
        "initial_budget": 8,
        "remaining_credits": 8,
        "probes_taken": 0,
        "max_probes": 6,
        "lambda_cost": 0.05,
    }
    values.update(overrides)
    return build_observation(**values)  # type: ignore[arg-type,return-value]


def test_schema_declares_fixed_names_shapes_dtypes_and_bounds() -> None:
    observation = _build()
    space = make_observation_space()

    assert OBSERVATION_SCHEMA_VERSION == "sim_v0"
    assert len(NODE_FEATURE_NAMES) == 42
    assert len(GLOBAL_FEATURE_NAMES) == 10
    assert observation.keys() == {
        "node_features", "node_mask", "adjacency", "global_features", "action_mask"
    }
    assert observation["node_features"].shape == (10, 42)
    assert observation["node_features"].dtype == np.float32
    assert observation["node_mask"].shape == (10,)
    assert observation["node_mask"].dtype == np.bool_
    assert observation["adjacency"].shape == (10, 10)
    assert observation["adjacency"].dtype == np.float32
    assert observation["global_features"].shape == (10,)
    assert observation["global_features"].dtype == np.float32
    assert observation["action_mask"].shape == (41,)
    assert observation["action_mask"].dtype == np.bool_
    assert space["node_mask"].dtype == np.bool_
    assert space["action_mask"].dtype == np.bool_
    assert space.contains(observation)


def test_initial_node_and_global_features_follow_schema_order() -> None:
    observation = _build()
    nodes = observation["node_features"]

    np.testing.assert_allclose(nodes[0, :8], [1, 0, 1 / 9, 0, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(nodes[1, :8], [0, 1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(nodes[2, :8], [0, 1 / 9, 0, 2 / 9, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    assert np.all(nodes[:3, 8:35] == 0)
    assert np.all(nodes[:3, 35:37] == 1)
    assert np.all(nodes[:3, 37:] == 0)
    np.testing.assert_allclose(
        observation["global_features"],
        [0.5, 0.5, 1, 1, 1, 1 / 3, 0, 0.5, 0.3, 0.25],
    )


def test_acquired_values_presence_execution_and_spend_are_distinct() -> None:
    metric = MetricEvidenceRecord(1, MetricCategory.CPU_HIGH, 0, False)
    log = LogEvidenceRecord(1, 2, LogCategory.TIMEOUT)
    action = probe_action_index(1, "metrics.detailed")
    detailed_metrics = tuple(
        MetricEvidenceRecord(
            1,
            category,
            reading_index,
            metric.value
            if category is MetricCategory.CPU_HIGH and reading_index == 0
            else True,
        )
        for category in MetricCategory
        for reading_index in range(2)
    )

    observation = _build(
        evidence_records=(*detailed_metrics, log),
        executed_probe_actions=(action,),
        probes_taken=1,
        remaining_credits=6,
    )
    row = observation["node_features"][1]

    assert row[8] == 0
    assert row[14] == 1
    assert row[20 + 2 * 4 + LogCategory.TIMEOUT] == 1
    assert row[32 + 2] == 1
    assert row[37:41].tolist() == [0, 1, 0, 0]
    assert row[41] == pytest.approx(0.25)
    assert not observation["action_mask"][probe_action_index(1, "metrics.quick")]


def test_tool_availability_and_terminated_mask_are_visible() -> None:
    available = {0: frozenset({ProbeTool.METRICS}), 2: frozenset({ProbeTool.LOGS})}
    observation = _build(available_tools=available, terminated=True)

    assert observation["node_features"][:3, 35:37].tolist() == [
        [1, 0], [0, 0], [0, 1]
    ]
    assert not observation["action_mask"].any()


def test_padded_nodes_and_adjacency_are_zero_and_fresh() -> None:
    graph = _graph()
    first = _build(graph=graph)
    second = _build(graph=graph)

    assert first["node_mask"].tolist() == [True, True, True] + [False] * 7
    assert not first["node_features"][3:].any()
    assert not first["adjacency"][3:, :].any()
    assert not first["adjacency"][:, 3:].any()
    np.testing.assert_array_equal(first["adjacency"][:3, :3], graph.adjacency)
    first["node_features"][0, 0] = 0
    first["adjacency"][0, 1] = 0
    assert second["node_features"][0, 0] == 1
    assert second["adjacency"][0, 1] == 1
    assert graph.adjacency[0, 1]


def test_observation_contains_no_hidden_incident_fields() -> None:
    assert set(_build()) == {
        "node_features", "node_mask", "adjacency", "global_features", "action_mask"
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"initial_budget": 17}, "schema"),
        ({"lambda_cost": 0.21}, "lambda_cost"),
        ({"probes_taken": 1}, "executed probe action count"),
    ],
)
def test_out_of_schema_or_inconsistent_episode_values_are_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(**overrides)
