"""Resolved simulator configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


EXPECTED_FAULT_TYPES = ("cpu", "memory", "network_delay")
EXPECTED_TOOL_COSTS = {
    "metrics.quick": 1,
    "metrics.detailed": 2,
    "logs.quick": 2,
    "logs.detailed": 4,
}


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    """Validated values needed to generate and run a simulator version."""

    schema_version: int
    generator_version: str
    services: int
    max_services: int
    extra_edge_probability: float
    fault_types: tuple[str, ...]
    busy_probability: float
    metric_readings_per_category: int
    log_records_per_service: int
    caller_decay: float
    budget_credits: int
    max_probes: int
    lambda_cost: float
    reward_shaping: str
    tool_costs: Mapping[str, int]
    likelihoods: Mapping[str, Any]

    def environment_kwargs(self) -> dict[str, int | float]:
        return {
            "n_services": self.services,
            "initial_budget": self.budget_credits,
            "max_probes": self.max_probes,
            "lambda_cost": self.lambda_cost,
            "extra_edge_probability": self.extra_edge_probability,
        }


def load_simulator_config(path: str | Path) -> SimulatorConfig:
    """Load a complete simulator YAML file and reject unresolved values."""
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not load simulator config {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("simulator config must contain a mapping")

    required = {
        "schema_version", "generator_version", "services", "max_services",
        "extra_edge_probability", "fault_types", "busy_probability",
        "metric_readings_per_category", "log_records_per_service", "caller_decay",
        "budget_credits", "max_probes", "lambda_cost", "reward_shaping",
        "tool_costs", "likelihoods",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"simulator config is missing: {', '.join(missing)}")

    schema_version = _integer(payload["schema_version"], "schema_version", minimum=1)
    generator_version = _string(payload["generator_version"], "generator_version")
    services = _integer(payload["services"], "services", minimum=1, maximum=10)
    max_services = _integer(payload["max_services"], "max_services", minimum=services, maximum=10)
    raw_fault_types = payload["fault_types"]
    if not isinstance(raw_fault_types, list):
        raise TypeError("fault_types must be a list")
    fault_types = tuple(raw_fault_types)
    if fault_types != EXPECTED_FAULT_TYPES:
        raise ValueError(f"fault_types must be {list(EXPECTED_FAULT_TYPES)!r}")
    reward_shaping = _string(payload["reward_shaping"], "reward_shaping")
    if reward_shaping != "none":
        raise ValueError("the v0 simulator supports only reward_shaping: none")
    tool_costs = _integer_mapping(payload["tool_costs"], "tool_costs")
    if tool_costs != EXPECTED_TOOL_COSTS:
        raise ValueError(f"tool_costs must be {EXPECTED_TOOL_COSTS!r}")
    likelihoods = payload["likelihoods"]
    if not isinstance(likelihoods, dict) or not likelihoods:
        raise ValueError("likelihoods must contain resolved probability tables")
    _validate_probability_tree(likelihoods, "likelihoods")

    return SimulatorConfig(
        schema_version=schema_version,
        generator_version=generator_version,
        services=services,
        max_services=max_services,
        extra_edge_probability=_real(payload["extra_edge_probability"], "extra_edge_probability", minimum=0.0, maximum=1.0),
        fault_types=fault_types,
        busy_probability=_real(payload["busy_probability"], "busy_probability", minimum=0.0, maximum=1.0),
        metric_readings_per_category=_integer(payload["metric_readings_per_category"], "metric_readings_per_category", minimum=1),
        log_records_per_service=_integer(payload["log_records_per_service"], "log_records_per_service", minimum=1),
        caller_decay=_real(payload["caller_decay"], "caller_decay", minimum=0.0, maximum=1.0),
        budget_credits=_integer(payload["budget_credits"], "budget_credits", minimum=1, maximum=16),
        max_probes=_integer(payload["max_probes"], "max_probes", minimum=1),
        lambda_cost=_real(payload["lambda_cost"], "lambda_cost", minimum=0.0, maximum=0.2),
        reward_shaping=reward_shaping,
        tool_costs=tool_costs,
        likelihoods=likelihoods,
    )


def _integer(value: object, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or maximum is not None and value > maximum:
        bound = f" between {minimum} and {maximum}" if maximum is not None else f" at least {minimum}"
        raise ValueError(f"{name} must be{bound}")
    return value


def _real(value: object, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _integer_mapping(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        result[key] = _integer(item, f"{name}.{key}", minimum=0)
    return result


def _validate_probability_tree(value: object, name: str) -> None:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{name} cannot be empty")
        for key, item in value.items():
            _validate_probability_tree(item, f"{name}.{key}")
        return
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} cannot be empty")
        for index, item in enumerate(value):
            _validate_probability_tree(item, f"{name}[{index}]")
        return
    _real(value, name, minimum=0.0, maximum=1.0)
