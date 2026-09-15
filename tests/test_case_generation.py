from rca_sim.config import load_simulator_config
from rca_sim.environment import InvestigationEnv
from rca_sim.make_cases import PARTITION_IDS, incident_seed, topology_hash
from rca_sim.tools import probe_action_index
from rca_sim.world import generate_incident_world, permute_incident_world


def test_resolved_simulator_config_loads() -> None:
    config = load_simulator_config("configs/sim_v0.yaml")
    assert config.services == 8
    assert config.environment_kwargs()["lambda_cost"] == 0.05
    assert config.tool_costs["logs.detailed"] == 4


def test_partition_seed_namespaces_are_stable_and_separate() -> None:
    first = incident_seed(20260913, "debug", 0)
    assert first == incident_seed(20260913, "debug", 0)
    assert first != incident_seed(20260913, "validation", 0)
    assert first != incident_seed(20260913, "debug", 1)
    assert len(set(PARTITION_IDS.values())) == len(PARTITION_IDS)


def test_topology_hash_uses_public_graph() -> None:
    world = generate_incident_world(n_services=8, incident_seed=42)
    digest = topology_hash(world)
    assert len(digest) == 64
    int(digest, 16)


def test_exact_case_records_do_not_depend_on_action_order() -> None:
    case = {
        "case_id": "order-check",
        "world": {
            "generator_version": "sim_v0",
            "incident_seed": incident_seed(20260913, "debug", 0),
            "n_services": 8,
            "extra_edge_probability": 0.15,
        },
    }
    actions = (
        probe_action_index(0, "metrics.quick"),
        probe_action_index(1, "logs.quick"),
    )

    observed = []
    for order in (actions, tuple(reversed(actions))):
        environment = InvestigationEnv()
        try:
            environment.reset(options={"case": case})
            values = {}
            for action in order:
                _, _, _, _, info = environment.step(action)
                values.update(
                    (record.evidence_id, record.value)
                    for record in environment.evidence_records
                )
            observed.append(values)
        finally:
            environment.close()
    assert observed[0] == observed[1]


def test_service_permutation_preserves_world_semantics_and_replay() -> None:
    world = generate_incident_world(n_services=8, incident_seed=2718)
    permutation = (3, 0, 7, 2, 5, 1, 6, 4)
    permuted = permute_incident_world(world, permutation)
    assert permuted.graph.entry_service == permutation[world.graph.entry_service]
    assert permuted.hypothesis.cause_service == permutation[world.hypothesis.cause_service]
    for old_id, new_id in enumerate(permutation):
        assert (
            permuted.observations.metric_readings[new_id]
            == world.observations.metric_readings[old_id]
        ).all()
    replayed = permuted.replay()
    assert replayed.graph.edges == permuted.graph.edges
    assert replayed.hypothesis == permuted.hypothesis
    assert (replayed.observations.metric_readings == permuted.observations.metric_readings).all()
