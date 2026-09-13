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
