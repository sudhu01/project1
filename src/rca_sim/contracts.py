"""Shared simulator and policy data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral


METRIC_READINGS_PER_CATEGORY = 2
LOG_READINGS_PER_SERVICE = 3


class FaultType(IntEnum):
    """Supported root-cause fault families in model encoding order."""

    CPU = 0
    MEMORY = 1
    NETWORK_DELAY = 2


class Workload(IntEnum):
    """Hidden background workload condition."""

    NORMAL = 0
    BUSY = 1


class RelationshipKind(IntEnum):
    """How an observed service relates to a candidate cause service."""

    CAUSE = 0
    AFFECTED_CALLER = 1
    UNRELATED = 2


class MetricCategory(IntEnum):
    """Binary diagnostic metric categories in observation encoding order."""

    CPU_HIGH = 0
    MEMORY_HIGH = 1
    LATENCY_HIGH = 2


class LogCategory(IntEnum):
    """Synthetic log categories in observation encoding order."""

    NO_RELEVANT_ERROR = 0
    RESOURCE_PRESSURE = 1
    TIMEOUT = 2
    OTHER_ERROR = 3


@dataclass(frozen=True, slots=True)
class ServiceRelationship:
    """A relationship category and its directed distance to the cause."""

    kind: RelationshipKind
    distance: int | None

    def __post_init__(self) -> None:
        if isinstance(self.kind, bool) or not isinstance(self.kind, Integral):
            raise TypeError("kind must be an integer encoding")
        try:
            kind = RelationshipKind(int(self.kind))
        except (TypeError, ValueError) as error:
            raise ValueError("relationship kind is not supported") from error
        object.__setattr__(self, "kind", kind)

        if self.distance is not None:
            if isinstance(self.distance, bool) or not isinstance(
                self.distance, Integral
            ):
                raise TypeError("distance must be an integer or None")
            object.__setattr__(self, "distance", int(self.distance))

        if kind is RelationshipKind.CAUSE and self.distance != 0:
            raise ValueError("cause relationship must have distance 0")
        if kind is RelationshipKind.AFFECTED_CALLER and (
            self.distance is None or self.distance < 1
        ):
            raise ValueError("affected caller relationship needs a positive distance")
        if kind is RelationshipKind.UNRELATED and self.distance is not None:
            raise ValueError("unrelated relationship must have no distance")
