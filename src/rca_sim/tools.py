"""Semantic diagnostic probe templates and evidence collection."""

from __future__ import annotations

from collections.abc import Collection, Mapping as MappingABC
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from rca_sim.contracts import EvidenceRecord, MetricCategory
from rca_sim.world import HiddenIncidentWorld


BoolArray = NDArray[np.bool_]


class ProbeTool(str, Enum):
    """Supported diagnostic tool families."""

    METRICS = "metrics"
    LOGS = "logs"


class ProbePreset(str, Enum):
    """Supported evidence-scope presets."""

    QUICK = "quick"
    DETAILED = "detailed"


class ProbeStatus(str, Enum):
    """Backend-neutral execution status values."""

    OK = "ok"


@dataclass(frozen=True, slots=True)
class ProbeAction:
    """A semantic request that does not depend on the simulator backend."""

    tool: ProbeTool
    target_id: str
    preset: ProbePreset

    def __post_init__(self) -> None:
        if not isinstance(self.tool, ProbeTool):
            raise TypeError("tool must be a ProbeTool")
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("target_id must be a nonempty string")
        if not isinstance(self.preset, ProbePreset):
            raise TypeError("preset must be a ProbePreset")


@dataclass(frozen=True, slots=True)
class ProbeCost:
    """Public resource cost charged for a probe."""

    credits: int

    def __post_init__(self) -> None:
        if isinstance(self.credits, bool) or not isinstance(self.credits, Integral):
            raise TypeError("credits must be an integer")
        if self.credits < 0:
            raise ValueError("credits must be nonnegative")
        object.__setattr__(self, "credits", int(self.credits))


@dataclass(frozen=True, slots=True)
class ProbeCoverage:
    """Whether the backend returned the full requested evidence coverage."""

    complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.complete, (bool, np.bool_)):
            raise TypeError("complete must be boolean")
        object.__setattr__(self, "complete", bool(self.complete))


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Evidence returned by a diagnostic backend, without policy outputs."""

    action: ProbeAction
    status: ProbeStatus
    evidence: tuple[EvidenceRecord, ...]
    cost: ProbeCost
    coverage: ProbeCoverage

    def __post_init__(self) -> None:
        if not isinstance(self.action, ProbeAction):
            raise TypeError("action must be a ProbeAction")
        if not isinstance(self.status, ProbeStatus):
            raise TypeError("status must be a ProbeStatus")
        if not isinstance(self.evidence, tuple) or any(
            not _is_evidence_record(record) for record in self.evidence
        ):
            raise TypeError("evidence must be a tuple of evidence records")
        if not isinstance(self.cost, ProbeCost):
            raise TypeError("cost must be a ProbeCost")
        if not isinstance(self.coverage, ProbeCoverage):
            raise TypeError("coverage must be a ProbeCoverage")

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible representation."""
        return {
            "action": {
                "tool": self.action.tool.value,
                "target_id": self.action.target_id,
                "preset": self.action.preset.value,
            },
            "status": self.status.value,
            "evidence": [_serialize_evidence(record) for record in self.evidence],
            "cost": {"credits": self.cost.credits},
            "coverage": {"complete": self.coverage.complete},
        }


@dataclass(frozen=True, slots=True)
class ProbeTemplate:
    """One semantic tool and preset with fixed coverage and credit cost."""

    tool: ProbeTool
    preset: ProbePreset
    cost_credits: int
    reading_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tool, ProbeTool):
            raise TypeError("tool must be a ProbeTool")
        if not isinstance(self.preset, ProbePreset):
            raise TypeError("preset must be a ProbePreset")
        if isinstance(self.cost_credits, bool) or not isinstance(
            self.cost_credits, Integral
        ):
            raise TypeError("cost_credits must be an integer")
        if self.cost_credits <= 0:
            raise ValueError("cost_credits must be positive")
        if (
            not isinstance(self.reading_indices, tuple)
            or not self.reading_indices
            or any(
                isinstance(index, bool) or not isinstance(index, Integral)
                for index in self.reading_indices
            )
        ):
            raise TypeError("reading_indices must be a nonempty tuple of integers")

        reading_indices = tuple(int(index) for index in self.reading_indices)
        if reading_indices != tuple(range(len(reading_indices))):
            raise ValueError("reading_indices must be consecutive and start at zero")

        expected_indices = (0,) if self.preset is ProbePreset.QUICK else (0, 1)
        if self.tool is ProbeTool.LOGS and self.preset is ProbePreset.DETAILED:
            expected_indices = (0, 1, 2)
        if reading_indices != expected_indices:
            raise ValueError(
                f"{self.name} must cover reading indices {expected_indices}"
            )

        object.__setattr__(self, "cost_credits", int(self.cost_credits))
        object.__setattr__(self, "reading_indices", reading_indices)

    @property
    def name(self) -> str:
        """Return the stable ``tool.preset`` template name."""
        return f"{self.tool.value}.{self.preset.value}"

    def evidence_ids(self, service_id: int) -> tuple[str, ...]:
        """Return this template's complete evidence coverage for one service."""
        service_id = _validate_service_id_value(service_id)
        if self.tool is ProbeTool.METRICS:
            return tuple(
                f"metrics/service-{service_id}/{metric.evidence_name}/{reading_index}"
                for reading_index in self.reading_indices
                for metric in MetricCategory
            )
        return tuple(
            f"logs/service-{service_id}/{reading_index}"
            for reading_index in self.reading_indices
        )


_TEMPLATES = (
    ProbeTemplate(ProbeTool.METRICS, ProbePreset.QUICK, 1, (0,)),
    ProbeTemplate(ProbeTool.METRICS, ProbePreset.DETAILED, 2, (0, 1)),
    ProbeTemplate(ProbeTool.LOGS, ProbePreset.QUICK, 2, (0,)),
    ProbeTemplate(ProbeTool.LOGS, ProbePreset.DETAILED, 4, (0, 1, 2)),
)

PROBE_TEMPLATES: Mapping[str, ProbeTemplate] = MappingProxyType(
    {template.name: template for template in _TEMPLATES}
)
PROBE_TEMPLATE_NAMES = tuple(PROBE_TEMPLATES)
TEMPLATES_PER_SERVICE = len(PROBE_TEMPLATE_NAMES)
MAX_SERVICE_SLOTS = 10
STOP_ACTION_INDEX = TEMPLATES_PER_SERVICE * MAX_SERVICE_SLOTS
ACTION_SPACE_SIZE = STOP_ACTION_INDEX + 1

_TEMPLATE_INDEX_BY_NAME: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(PROBE_TEMPLATE_NAMES)}
)


def get_probe_template(name: str) -> ProbeTemplate:
    """Look up one probe template by its stable name."""
    if not isinstance(name, str):
        raise TypeError("probe template name must be a string")
    try:
        return PROBE_TEMPLATES[name]
    except KeyError as error:
        raise KeyError(f"unknown probe template: {name}") from error


def probe_action_index(
    service_slot: int,
    template: str | ProbeTemplate,
) -> int:
    """Encode one service slot and probe template as a stable action index."""
    service_slot = _validate_service_slot(service_slot)
    template = _resolve_probe_template(template)
    return (
        TEMPLATES_PER_SERVICE * service_slot
        + _TEMPLATE_INDEX_BY_NAME[template.name]
    )


def decode_probe_action(action_index: int) -> tuple[int, ProbeTemplate]:
    """Decode a probe index into its service slot and canonical template."""
    action_index = _validate_action_index(action_index)
    if action_index == STOP_ACTION_INDEX:
        raise ValueError("STOP does not identify a probe action")

    service_slot, template_index = divmod(action_index, TEMPLATES_PER_SERVICE)
    template_name = PROBE_TEMPLATE_NAMES[template_index]
    return service_slot, PROBE_TEMPLATES[template_name]


def is_stop_action(action_index: int) -> bool:
    """Return whether an in-range action index is the fixed STOP action."""
    return _validate_action_index(action_index) == STOP_ACTION_INDEX


def construct_action_mask(
    *,
    n_services: int,
    remaining_credits: int,
    seen_evidence_ids: Collection[str],
    available_tools: Mapping[int, Collection[ProbeTool]] | None = None,
    terminated: bool = False,
) -> BoolArray:
    """Return valid probe and STOP actions for the visible episode state."""
    n_services = _validate_active_service_count(n_services)
    remaining_credits = _validate_remaining_credits(remaining_credits)
    seen_ids = _validate_seen_evidence_ids(seen_evidence_ids)
    tools_by_service = _resolve_available_tools(n_services, available_tools)
    if not isinstance(terminated, (bool, np.bool_)):
        raise TypeError("terminated must be boolean")

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    if bool(terminated):
        return mask

    for service_id in range(n_services):
        service_tools = tools_by_service[service_id]
        for template_name in PROBE_TEMPLATE_NAMES:
            template = PROBE_TEMPLATES[template_name]
            action_index = probe_action_index(service_id, template)
            mask[action_index] = (
                template.tool in service_tools
                and template.cost_credits <= remaining_credits
                and not set(template.evidence_ids(service_id)).issubset(seen_ids)
            )

    mask[STOP_ACTION_INDEX] = True
    return mask


def execute_probe_action(
    world: HiddenIncidentWorld,
    action_index: int,
    *,
    remaining_credits: int,
    seen_evidence_ids: Collection[str],
    available_tools: Mapping[int, Collection[ProbeTool]] | None = None,
) -> ProbeResult:
    """Validate and execute one simulator probe without mutating episode state."""
    if not isinstance(world, HiddenIncidentWorld):
        raise TypeError("world must be a HiddenIncidentWorld")
    action_index = _validate_action_index(action_index)
    mask = construct_action_mask(
        n_services=world.graph.n_services,
        remaining_credits=remaining_credits,
        seen_evidence_ids=seen_evidence_ids,
        available_tools=available_tools,
    )
    if action_index == STOP_ACTION_INDEX:
        raise ValueError("STOP does not execute a probe")
    if not mask[action_index]:
        raise ValueError(f"probe action {action_index} is invalid for the current state")

    service_id, template = decode_probe_action(action_index)
    evidence = collect_probe_evidence(world, template, service_id)
    return ProbeResult(
        action=ProbeAction(
            tool=template.tool,
            target_id=f"service-{service_id}",
            preset=template.preset,
        ),
        status=ProbeStatus.OK,
        evidence=evidence,
        cost=ProbeCost(template.cost_credits),
        coverage=ProbeCoverage(complete=True),
    )


def collect_probe_evidence(
    world: HiddenIncidentWorld,
    template: str | ProbeTemplate,
    service_id: int,
) -> tuple[EvidenceRecord, ...]:
    """Return a probe's frozen records without changing or resampling the world."""
    if not isinstance(world, HiddenIncidentWorld):
        raise TypeError("world must be a HiddenIncidentWorld")
    template = _resolve_probe_template(template)

    service_id = _validate_service_id_value(service_id)
    if service_id >= world.graph.n_services:
        raise ValueError(
            f"service_id {service_id} is out of range for "
            f"{world.graph.n_services} services"
        )
    return tuple(
        world.get_evidence(evidence_id)
        for evidence_id in template.evidence_ids(service_id)
    )


def _validate_service_id_value(service_id: int) -> int:
    if isinstance(service_id, bool) or not isinstance(service_id, Integral):
        raise TypeError("service_id must be an integer")
    service_id = int(service_id)
    if service_id < 0:
        raise ValueError("service_id must be nonnegative")
    return service_id


def _validate_active_service_count(n_services: int) -> int:
    if isinstance(n_services, bool) or not isinstance(n_services, Integral):
        raise TypeError("n_services must be an integer")
    n_services = int(n_services)
    if not 1 <= n_services <= MAX_SERVICE_SLOTS:
        raise ValueError(
            f"n_services must be between 1 and {MAX_SERVICE_SLOTS}"
        )
    return n_services


def _validate_remaining_credits(remaining_credits: int) -> int:
    if isinstance(remaining_credits, bool) or not isinstance(
        remaining_credits, Integral
    ):
        raise TypeError("remaining_credits must be an integer")
    remaining_credits = int(remaining_credits)
    if remaining_credits < 0:
        raise ValueError("remaining_credits must be nonnegative")
    return remaining_credits


def _validate_seen_evidence_ids(
    evidence_ids: Collection[str],
) -> frozenset[str]:
    if isinstance(evidence_ids, (str, bytes)) or not isinstance(
        evidence_ids, Collection
    ):
        raise TypeError("seen_evidence_ids must be a collection of strings")
    if any(not isinstance(evidence_id, str) for evidence_id in evidence_ids):
        raise TypeError("seen_evidence_ids must contain only strings")
    return frozenset(evidence_ids)


def _resolve_available_tools(
    n_services: int,
    available_tools: Mapping[int, Collection[ProbeTool]] | None,
) -> dict[int, frozenset[ProbeTool]]:
    if available_tools is None:
        all_tools = frozenset(ProbeTool)
        return {service_id: all_tools for service_id in range(n_services)}
    if not isinstance(available_tools, MappingABC):
        raise TypeError("available_tools must be a mapping")

    resolved = {service_id: frozenset() for service_id in range(n_services)}
    for service_id, tools in available_tools.items():
        service_id = _validate_service_id_value(service_id)
        if service_id >= n_services:
            raise ValueError("available_tools contains an inactive service")
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Collection):
            raise TypeError("available tool values must be collections")
        if any(not isinstance(tool, ProbeTool) for tool in tools):
            raise TypeError("available tools must contain only ProbeTool values")
        resolved[service_id] = frozenset(tools)
    return resolved


def _serialize_evidence(record: EvidenceRecord) -> dict[str, Any]:
    value = record.value
    if isinstance(value, (bool, np.bool_)):
        serialized_value: int = int(value)
    elif isinstance(value, Integral):
        serialized_value = int(value)
    else:
        raise TypeError("evidence value is not JSON-compatible")
    return {
        "id": record.evidence_id,
        "kind": record.kind,
        "value": serialized_value,
    }


def _is_evidence_record(record: object) -> bool:
    from rca_sim.contracts import LogEvidenceRecord, MetricEvidenceRecord

    return isinstance(record, (MetricEvidenceRecord, LogEvidenceRecord))


def _resolve_probe_template(template: str | ProbeTemplate) -> ProbeTemplate:
    if isinstance(template, str):
        return get_probe_template(template)
    if not isinstance(template, ProbeTemplate):
        raise TypeError("template must be a name or ProbeTemplate")
    return get_probe_template(template.name)


def _validate_service_slot(service_slot: int) -> int:
    if isinstance(service_slot, bool) or not isinstance(service_slot, Integral):
        raise TypeError("service_slot must be an integer")
    service_slot = int(service_slot)
    if not 0 <= service_slot < MAX_SERVICE_SLOTS:
        raise ValueError(
            f"service_slot must be between 0 and {MAX_SERVICE_SLOTS - 1}"
        )
    return service_slot


def _validate_action_index(action_index: int) -> int:
    if isinstance(action_index, bool) or not isinstance(action_index, Integral):
        raise TypeError("action_index must be an integer")
    action_index = int(action_index)
    if not 0 <= action_index < ACTION_SPACE_SIZE:
        raise ValueError(
            f"action_index must be between 0 and {ACTION_SPACE_SIZE - 1}"
        )
    return action_index
