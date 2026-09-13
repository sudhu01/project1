"""Shared simulator and policy data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral

import numpy as np


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

    @property
    def evidence_name(self) -> str:
        """Return the stable metric segment used in evidence IDs."""
        return self.name.removesuffix("_HIGH").lower()


class LogCategory(IntEnum):
    """Synthetic log categories in observation encoding order."""

    NO_RELEVANT_ERROR = 0
    RESOURCE_PRESSURE = 1
    TIMEOUT = 2
    OTHER_ERROR = 3


@dataclass(frozen=True, slots=True)
class MetricEvidenceRecord:
    """One immutable binary metric reading."""

    service_id: int
    metric: MetricCategory
    reading_index: int
    value: bool

    def __post_init__(self) -> None:
        service_id = _nonnegative_integer(self.service_id, "service_id")
        reading_index = _bounded_index(
            self.reading_index,
            METRIC_READINGS_PER_CATEGORY,
            "metric reading_index",
        )
        if isinstance(self.metric, bool) or not isinstance(self.metric, Integral):
            raise TypeError("metric must be an integer category encoding")
        try:
            metric = MetricCategory(int(self.metric))
        except (TypeError, ValueError) as error:
            raise ValueError("metric category is not supported") from error
        if not isinstance(self.value, (bool, np.bool_)):
            raise TypeError("metric value must be boolean")

        object.__setattr__(self, "service_id", service_id)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "reading_index", reading_index)
        object.__setattr__(self, "value", bool(self.value))

    @property
    def evidence_id(self) -> str:
        return (
            f"metrics/service-{self.service_id}/"
            f"{self.metric.evidence_name}/{self.reading_index}"
        )

    @property
    def kind(self) -> str:
        return self.metric.name.lower()


@dataclass(frozen=True, slots=True)
class LogEvidenceRecord:
    """One immutable categorical log reading."""

    service_id: int
    reading_index: int
    value: LogCategory

    def __post_init__(self) -> None:
        service_id = _nonnegative_integer(self.service_id, "service_id")
        reading_index = _bounded_index(
            self.reading_index,
            LOG_READINGS_PER_SERVICE,
            "log reading_index",
        )
        if isinstance(self.value, bool) or not isinstance(self.value, Integral):
            raise TypeError("log value must be an integer category encoding")
        try:
            value = LogCategory(int(self.value))
        except (TypeError, ValueError) as error:
            raise ValueError("log category is not supported") from error

        object.__setattr__(self, "service_id", service_id)
        object.__setattr__(self, "reading_index", reading_index)
        object.__setattr__(self, "value", value)

    @property
    def evidence_id(self) -> str:
        return f"logs/service-{self.service_id}/{self.reading_index}"

    @property
    def kind(self) -> str:
        return "log_category"


EvidenceRecord = MetricEvidenceRecord | LogEvidenceRecord


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


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _bounded_index(value: int, size: int, name: str) -> int:
    value = _nonnegative_integer(value, name)
    if value >= size:
        raise ValueError(f"{name} must be less than {size}")
    return value
