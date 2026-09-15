"""Run a readable policy trace against one frozen incident case."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from rca_sim.baselines import ImmediateStopPolicy, OneStepValueOfInformationPolicy, ScriptedInvestigatorPolicy
from rca_sim.config import load_simulator_config
from rca_sim.environment import InvestigationEnv, load_exact_case
from rca_sim.evaluate import evaluate_episode
from rca_sim.tools import STOP_ACTION_INDEX, decode_probe_action


def inspect_case(
    case_path: Path, *, output: Path, policy_name: str, config_path: Path
) -> dict[str, object]:
    config = load_simulator_config(config_path)
    environment = InvestigationEnv(**config.environment_kwargs(), trace_enabled=True)
    if policy_name == "script":
        policy = ScriptedInvestigatorPolicy(confidence_threshold=0.75)
    elif policy_name == "voi1":
        policy = OneStepValueOfInformationPolicy(environment)
    else:
        policy = ImmediateStopPolicy()
    try:
        row = evaluate_episode(environment, case_path, policy, fallback_case_id=case_path.stem)
        events = []
        for event in environment.trace_events:
            copied = dict(event)
            action = copied.get("action")
            if isinstance(action, int) and action != STOP_ACTION_INDEX:
                service, template = decode_probe_action(action)
                copied["semantic_action"] = {"service": service, "template": template.name}
            elif action == STOP_ACTION_INDEX:
                copied["semantic_action"] = "stop"
            events.append(copied)
        world, _ = load_exact_case(
            case_path,
            default_n_services=config.services,
            default_edge_probability=config.extra_edge_probability,
        )
        summary = asdict(row)
        summary["truth"] = {
            "cause_service": world.hypothesis.cause_service,
            "fault_type": world.hypothesis.fault_type.name.lower(),
            "workload": world.hypothesis.workload.name.lower(),
        }
        summary["graph"] = {
            "entry_service": world.graph.entry_service,
            "edges": world.graph.edges,
        }
    finally:
        environment.close()

    output.mkdir(parents=True, exist_ok=True)
    (output / "trace.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--policy", choices=("script", "voi1", "stop"), default="script")
    parser.add_argument("--config", type=Path, default=Path("configs/sim_v0.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = inspect_case(
        args.case, output=args.output, policy_name=args.policy, config_path=args.config
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
