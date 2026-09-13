"""Semantic diagnostic probe templates and evidence collection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

from rca_sim.contracts import EvidenceRecord, MetricCategory
from rca_sim.world import HiddenIncidentWorld


class ProbeTool(str, Enum):
    """Supported diagnostic tool families."""

    METRICS = "metrics"
    LOGS = "logs"


class ProbePreset(str, Enum):
    """Supported evidence-scope presets."""

    QUICK = "quick"
    DETAILED = "detailed"


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
