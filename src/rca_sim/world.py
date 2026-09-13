"""Hidden incident state and generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral

import numpy as np

from rca_sim.graph import DependencyGraph


class FaultType(IntEnum):
    """Supported root-cause fault families in their model encoding order."""

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


@dataclass(frozen=True, slots=True)
class HiddenHypothesis:
    """The private service, fault, and workload state for one incident."""

    cause_service: int
    fault_type: FaultType
    workload: Workload

    def __post_init__(self) -> None:
        if isinstance(self.cause_service, bool) or not isinstance(
            self.cause_service, Integral
        ):
            raise TypeError("cause_service must be an integer")
        if self.cause_service < 0:
            raise ValueError("cause_service must be nonnegative")
        if isinstance(self.fault_type, bool) or not isinstance(
            self.fault_type, Integral
        ):
            raise TypeError("fault_type must be an integer encoding")
        if isinstance(self.workload, bool) or not isinstance(self.workload, Integral):
            raise TypeError("workload must be an integer encoding")

        object.__setattr__(self, "cause_service", int(self.cause_service))
        try:
            object.__setattr__(self, "fault_type", FaultType(int(self.fault_type)))
        except (TypeError, ValueError) as error:
            raise ValueError("fault_type is not supported") from error
        try:
            object.__setattr__(self, "workload", Workload(int(self.workload)))
        except (TypeError, ValueError) as error:
            raise ValueError("workload is not supported") from error


def enumerate_hidden_hypotheses(n_services: int) -> tuple[HiddenHypothesis, ...]:
    """Return the complete uniform-prior hypothesis space in stable order."""
    if isinstance(n_services, bool) or not isinstance(n_services, Integral):
        raise TypeError("n_services must be an integer")
    n_services = int(n_services)
    if n_services < 1:
        raise ValueError("n_services must be at least 1")

    return tuple(
        HiddenHypothesis(cause_service, fault_type, workload)
        for cause_service in range(n_services)
        for fault_type in FaultType
        for workload in Workload
    )


def sample_hidden_hypothesis(
    graph: DependencyGraph,
    rng: np.random.Generator,
) -> HiddenHypothesis:
    """Sample service, fault, and workload independently from uniform priors."""
    if not isinstance(graph, DependencyGraph):
        raise TypeError("graph must be a DependencyGraph")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    return HiddenHypothesis(
        cause_service=int(rng.integers(graph.n_services)),
        fault_type=FaultType(int(rng.integers(len(FaultType)))),
        workload=Workload(int(rng.integers(len(Workload)))),
    )


def classify_service_relationship(
    graph: DependencyGraph,
    *,
    observed_service: int,
    cause_service: int,
) -> ServiceRelationship:
    """Classify one service using caller-to-dependency path direction.

    A service is an affected caller only when it can reach the candidate cause
    by following call edges. A service downstream of the cause is unrelated in
    this first propagation model.
    """
    if not isinstance(graph, DependencyGraph):
        raise TypeError("graph must be a DependencyGraph")

    distance = graph.shortest_path_distance(observed_service, cause_service)
    if distance == 0:
        return ServiceRelationship(RelationshipKind.CAUSE, distance=0)
    if distance is not None:
        return ServiceRelationship(RelationshipKind.AFFECTED_CALLER, distance)
    return ServiceRelationship(RelationshipKind.UNRELATED, distance=None)
