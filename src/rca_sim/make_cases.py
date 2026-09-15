"""Generate reproducible incident case partitions and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rca_sim.config import SimulatorConfig, load_simulator_config
from rca_sim.contracts import FaultType, Workload
from rca_sim.world import HiddenIncidentWorld, generate_incident_world


PARTITION_IDS = {
    "fixtures": 11,
    "debug": 101,
    "streaming_train": 202,
    "validation": 303,
    "audit": 404,
    "test": 505,
    "policy_actions": 606,
}
PARTITION_SIZES = {"debug": 64, "validation": 256, "audit": 512, "test": 1000}
FIXTURE_NAMES = (
    "known_answer", "one_perfect_probe", "useless_evidence",
    "duplicate_evidence", "expensive_discriminator", "scope_choice",
    "last_credit_evidence", "no_affordable_probe", "two_probe_complementarity",
)


def incident_seed(master_seed: int, partition: str, index: int) -> int:
    """Derive one stable uint64 seed from a named partition namespace."""
    if partition not in PARTITION_IDS:
        raise ValueError(f"unknown seed partition: {partition}")
    sequence = np.random.SeedSequence([master_seed, PARTITION_IDS[partition], index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def topology_hash(world: HiddenIncidentWorld) -> str:
    """Hash the public graph without evaluator-only incident labels."""
    payload = {
        "entry_service": world.graph.entry_service,
        "edges": world.graph.edges,
        "n_services": world.graph.n_services,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_case_partitions(
    config: SimulatorConfig, *, master_seed: int, output: Path
) -> dict[str, Any]:
    """Write all frozen development and test descriptors."""
    output.mkdir(parents=True, exist_ok=True)
    manifests = output / "manifests"
    manifests.mkdir(exist_ok=True)
    root: dict[str, Any] = {
        "schema_version": config.schema_version,
        "generator_version": config.generator_version,
        "master_seed": master_seed,
        "partition_ids": PARTITION_IDS,
        "partitions": {},
        "streaming_training": {
            "partition_id": PARTITION_IDS["streaming_train"],
            "generated_at_reset": True,
        },
        "policy_action_rng": {"partition_id": PARTITION_IDS["policy_actions"]},
    }

    fixture_manifest = {
        "partition": "fixtures",
        "partition_id": PARTITION_IDS["fixtures"],
        "size": len(FIXTURE_NAMES),
        "fixtures": list(FIXTURE_NAMES),
    }
    _write_json(manifests / "fixtures.json", fixture_manifest)
    root["partitions"]["fixtures"] = "manifests/fixtures.json"

    audit_worlds: list[tuple[Path, HiddenIncidentWorld]] = []
    for partition, size in PARTITION_SIZES.items():
        directory = output / partition
        directory.mkdir(exist_ok=True)
        records: list[dict[str, Any]] = []
        for index in range(size):
            seed = incident_seed(master_seed, partition, index)
            world = generate_incident_world(
                n_services=config.services,
                incident_seed=seed,
                extra_edge_probability=config.extra_edge_probability,
            )
            case_id = f"{partition}_{index:04d}"
            case_path = directory / f"case_{index:04d}.json"
            digest = topology_hash(world)
            descriptor = {
                "case_id": case_id,
                "partition": partition,
                "schema_version": config.schema_version,
                "topology_hash": digest,
                "world": {
                    "generator_version": config.generator_version,
                    "incident_seed": seed,
                    "n_services": config.services,
                    "extra_edge_probability": config.extra_edge_probability,
                },
                "evaluator_only": {
                    "cause_service": world.hypothesis.cause_service,
                    "fault_type": world.hypothesis.fault_type.name.lower(),
                    "workload": world.hypothesis.workload.name.lower(),
                },
            }
            _write_json(case_path, descriptor)
            records.append({
                "case_id": case_id,
                "incident_seed": seed,
                "topology_hash": digest,
                "world_record": case_path.relative_to(output).as_posix(),
            })
            if partition == "audit":
                audit_worlds.append((case_path, world))

        manifest = {
            "schema_version": config.schema_version,
            "generator_version": config.generator_version,
            "master_seed": master_seed,
            "partition": partition,
            "partition_id": PARTITION_IDS[partition],
            "size": size,
            "cases": records,
        }
        manifest_path = manifests / f"{partition}.json"
        _write_json(manifest_path, manifest)
        root["partitions"][partition] = manifest_path.relative_to(output).as_posix()

    selection = _select_manual_audit_cases(audit_worlds, output)
    _write_json(manifests / "manual_audit.json", selection)
    root["manual_audit"] = "manifests/manual_audit.json"
    _write_json(output / "manifest.json", root)
    return root


def _select_manual_audit_cases(
    worlds: list[tuple[Path, HiddenIncidentWorld]], output: Path
) -> dict[str, Any]:
    predicates = {
        "entry_cause": lambda world: world.hypothesis.cause_service == world.graph.entry_service,
        "deep_dependency": lambda world: (
            world.graph.shortest_path_distance(world.graph.entry_service, world.hypothesis.cause_service) or 0
        ) >= 3,
        "shared_dependency": lambda world: bool(np.max(world.graph.in_degree) >= 2),
        "busy_workload": lambda world: world.hypothesis.workload is Workload.BUSY,
        "misleading_metric": _has_misleading_cause_metric,
    }
    used: set[Path] = set()
    selected: list[dict[str, Any]] = []
    for reason, predicate in predicates.items():
        match = next(
            ((path, world) for path, world in worlds if path not in used and predicate(world)),
            None,
        )
        if match is None:
            raise RuntimeError(f"could not find manual audit case for {reason}")
        path, world = match
        used.add(path)
        selected.append({
            "reason": reason,
            "case": path.relative_to(output).as_posix(),
            "cause_service": world.hypothesis.cause_service,
            "entry_service": world.graph.entry_service,
            "fault_type": world.hypothesis.fault_type.name.lower(),
            "workload": world.hypothesis.workload.name.lower(),
        })
    return {"count": len(selected), "cases": selected}


def _has_misleading_cause_metric(world: HiddenIncidentWorld) -> bool:
    primary_index = {
        FaultType.CPU: 0,
        FaultType.MEMORY: 1,
        FaultType.NETWORK_DELAY: 2,
    }[world.hypothesis.fault_type]
    readings = world.observations.metric_readings[world.hypothesis.cause_service, primary_index]
    return not bool(np.any(readings))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.master_seed < 0:
        raise ValueError("master seed must be nonnegative")
    config = load_simulator_config(args.config)
    result = generate_case_partitions(config, master_seed=args.master_seed, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
