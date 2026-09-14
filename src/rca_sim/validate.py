"""Reproducible validation checks for the simulator probability and belief models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rca_sim.belief import ExactBeliefEstimator, recompute_posterior
from rca_sim.contracts import (
    FaultType,
    LogCategory,
    LogEvidenceRecord,
    MetricCategory,
    MetricEvidenceRecord,
    RelationshipKind,
    ServiceRelationship,
    Workload,
)
from rca_sim.evidence import EvidenceLedger
from rca_sim.graph import DependencyGraph, classify_service_relationship
from rca_sim.likelihoods import (
    log_category_probabilities,
    log_category_probability,
    metric_high_probabilities,
    metric_high_probability,
)
from rca_sim.tools import collect_probe_evidence, execute_probe_action, probe_action_index
from rca_sim.world import (
    HiddenHypothesis,
    generate_incident_world,
    generate_synthetic_observations,
    sample_hidden_hypothesis,
)


MIN_OBSERVATION_SAMPLES = 10_000
DEFAULT_PRIOR_SAMPLES = 60_000
DEFAULT_SEED = 20260914
NUMERICAL_MARGIN = 0.001


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One named validation result with JSON-compatible measurements."""

    name: str
    passed: bool
    measurements: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Results for execution-plan steps 8.1 and 8.2."""

    seed: int
    observation_samples_per_hypothesis: int
    prior_samples: int
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = list(payload["checks"])
        payload["passed"] = self.passed
        return payload


def run_validation(
    *,
    seed: int = DEFAULT_SEED,
    observation_samples: int = MIN_OBSERVATION_SAMPLES,
    prior_samples: int = DEFAULT_PRIOR_SAMPLES,
) -> ValidationReport:
    """Run the complete validation required by plan steps 8.1 and 8.2."""
    if observation_samples < MIN_OBSERVATION_SAMPLES:
        raise ValueError(
            f"observation_samples must be at least {MIN_OBSERVATION_SAMPLES}"
        )
    if prior_samples < MIN_OBSERVATION_SAMPLES:
        raise ValueError(f"prior_samples must be at least {MIN_OBSERVATION_SAMPLES}")

    checks = (
        _validate_probability_tables(),
        _validate_distance_decay(),
        _validate_observation_frequencies(seed, observation_samples),
        _validate_prior_frequencies(seed + 1, prior_samples),
        _validate_posterior_invariants(seed + 2),
        _validate_quick_detailed_and_duplicates(seed + 3),
        _validate_shared_workload_marginalization(),
    )
    return ValidationReport(
        seed=seed,
        observation_samples_per_hypothesis=observation_samples,
        prior_samples=prior_samples,
        checks=checks,
    )


def _validate_probability_tables() -> ValidationCheck:
    minimum = 1.0
    maximum = 0.0
    largest_row_error = 0.0
    relationships = (
        ServiceRelationship(RelationshipKind.CAUSE, 0),
        *(
            ServiceRelationship(RelationshipKind.AFFECTED_CALLER, distance)
            for distance in range(1, 11)
        ),
        ServiceRelationship(RelationshipKind.UNRELATED, None),
    )
    passed = True
    for relationship in relationships:
        for fault in FaultType:
            for workload in Workload:
                metric = metric_high_probabilities(relationship, fault, workload)
                logs = log_category_probabilities(relationship, fault, workload)
                minimum = min(minimum, float(metric.min()), float(logs.min()))
                maximum = max(maximum, float(metric.max()), float(logs.max()))
                row_error = abs(float(logs.sum()) - 1.0)
                largest_row_error = max(largest_row_error, row_error)
                passed &= bool(np.all((0.0 <= metric) & (metric <= 1.0)))
                passed &= bool(np.all((0.0 <= logs) & (logs <= 1.0)))
                passed &= row_error <= 1e-12
    return ValidationCheck(
        "8.1 probability bounds and categorical normalization",
        passed,
        {
            "minimum_probability": minimum,
            "maximum_probability": maximum,
            "largest_log_row_sum_error": largest_row_error,
        },
    )


def _validate_distance_decay() -> ValidationCheck:
    passed = True
    largest_increase = 0.0
    unrelated = ServiceRelationship(RelationshipKind.UNRELATED, None)
    for fault in FaultType:
        for workload in Workload:
            metric_background = metric_high_probabilities(unrelated, fault, workload)
            log_background = log_category_probabilities(unrelated, fault, workload)
            prior_metric_distance = np.inf
            prior_log_distance = np.inf
            for distance in range(1, 11):
                relationship = ServiceRelationship(
                    RelationshipKind.AFFECTED_CALLER,
                    distance,
                )
                metric_distance = float(
                    np.abs(
                        metric_high_probabilities(relationship, fault, workload)
                        - metric_background
                    ).sum()
                )
                log_distance = float(
                    np.abs(
                        log_category_probabilities(relationship, fault, workload)
                        - log_background
                    ).sum()
                )
                largest_increase = max(
                    largest_increase,
                    metric_distance - prior_metric_distance,
                    log_distance - prior_log_distance,
                )
                passed &= metric_distance < prior_metric_distance
                passed &= log_distance < prior_log_distance
                prior_metric_distance = metric_distance
                prior_log_distance = log_distance
    return ValidationCheck(
        "8.1 caller distance approaches unrelated behavior",
        passed,
        {"largest_distance_increase": largest_increase},
    )


def _validate_observation_frequencies(
    seed: int,
    sample_count: int,
) -> ValidationCheck:
    graph = DependencyGraph.from_edges(
        4,
        [(0, 1), (1, 2), (2, 3)],
        entry_service=0,
    )
    hypotheses = (
        HiddenHypothesis(3, FaultType.CPU, Workload.NORMAL),
        HiddenHypothesis(3, FaultType.MEMORY, Workload.BUSY),
        HiddenHypothesis(3, FaultType.NETWORK_DELAY, Workload.NORMAL),
    )
    rng = np.random.default_rng(seed)
    largest_standardized_error = 0.0
    largest_absolute_error = 0.0
    passed = True

    for hypothesis in hypotheses:
        metric_counts = np.zeros(
            (graph.n_services, len(MetricCategory)),
            dtype=np.int64,
        )
        log_counts = np.zeros(
            (graph.n_services, len(LogCategory)),
            dtype=np.int64,
        )
        for _ in range(sample_count):
            observations = generate_synthetic_observations(graph, hypothesis, rng)
            metric_counts += observations.metric_readings.sum(axis=2)
            for service_id in range(graph.n_services):
                log_counts[service_id] += np.bincount(
                    observations.log_readings[service_id],
                    minlength=len(LogCategory),
                )

        for service_id in range(graph.n_services):
            relationship = classify_service_relationship(
                graph,
                observed_service=service_id,
                cause_service=hypothesis.cause_service,
            )
            expected_metrics = metric_high_probabilities(
                relationship,
                hypothesis.fault_type,
                hypothesis.workload,
            )
            expected_logs = log_category_probabilities(
                relationship,
                hypothesis.fault_type,
                hypothesis.workload,
            )
            distributions = (
                (
                    metric_counts[service_id] / (2 * sample_count),
                    expected_metrics,
                    2 * sample_count,
                ),
                (
                    log_counts[service_id] / (3 * sample_count),
                    expected_logs,
                    3 * sample_count,
                ),
            )
            for observed, expected, trials in distributions:
                standard_errors = np.sqrt(expected * (1.0 - expected) / trials)
                errors = np.abs(observed - expected)
                tolerances = 5.0 * standard_errors + NUMERICAL_MARGIN
                passed &= bool(np.all(errors <= tolerances))
                largest_absolute_error = max(largest_absolute_error, float(errors.max()))
                standardized = errors / np.maximum(
                    standard_errors,
                    np.finfo(float).eps,
                )
                largest_standardized_error = max(
                    largest_standardized_error,
                    float(standardized.max()),
                )

    return ValidationCheck(
        "8.1 fixed-hypothesis empirical observation frequencies",
        passed,
        {
            "hypotheses": len(hypotheses),
            "samples_per_hypothesis": sample_count,
            "largest_absolute_error": largest_absolute_error,
            "largest_standardized_error": largest_standardized_error,
            "tolerance": "five standard errors plus 0.001",
        },
    )


def _validate_prior_frequencies(seed: int, sample_count: int) -> ValidationCheck:
    graph = DependencyGraph.from_edges(
        8,
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (5, 7)],
        entry_service=0,
    )
    rng = np.random.default_rng(seed)
    counts = {
        "cause": np.zeros(graph.n_services, dtype=np.int64),
        "fault": np.zeros(len(FaultType), dtype=np.int64),
        "workload": np.zeros(len(Workload), dtype=np.int64),
    }
    for _ in range(sample_count):
        hypothesis = sample_hidden_hypothesis(graph, rng)
        counts["cause"][hypothesis.cause_service] += 1
        counts["fault"][hypothesis.fault_type] += 1
        counts["workload"][hypothesis.workload] += 1

    largest_standardized_error = 0.0
    passed = True
    for category_counts in counts.values():
        expected = 1.0 / len(category_counts)
        standard_error = np.sqrt(expected * (1.0 - expected) / sample_count)
        errors = np.abs(category_counts / sample_count - expected)
        passed &= bool(np.all(errors <= 5.0 * standard_error + NUMERICAL_MARGIN))
        largest_standardized_error = max(
            largest_standardized_error,
            float((errors / standard_error).max()),
        )
    return ValidationCheck(
        "8.1 independent cause, fault, and workload sampling frequencies",
        passed,
        {
            "samples": sample_count,
            "largest_standardized_error": largest_standardized_error,
            "counts": {name: values.tolist() for name, values in counts.items()},
            "tolerance": "five standard errors plus 0.001",
        },
    )


def _validate_posterior_invariants(seed: int) -> ValidationCheck:
    world = generate_incident_world(n_services=8, incident_seed=seed)
    records = tuple(world.evidence_by_id.values())
    orderings = (
        records,
        tuple(reversed(records)),
        tuple(
            records[index]
            for index in np.random.default_rng(seed).permutation(len(records))
        ),
    )
    posteriors = []
    largest_reference_error = 0.0
    passed = True
    for ordering in orderings:
        estimator = ExactBeliefEstimator(world.graph)
        for record in ordering:
            estimator.update((record,))
        posterior = estimator.posterior
        reference = recompute_posterior(world.graph, ordering)
        posteriors.append(posterior)
        error = float(np.max(np.abs(posterior - reference)))
        largest_reference_error = max(largest_reference_error, error)
        passed &= abs(float(posterior.sum()) - 1.0) <= 1e-12
        passed &= bool(np.all(posterior >= 0.0))
        passed &= error <= 1e-12

    largest_order_error = max(
        float(np.max(np.abs(posteriors[0] - posterior)))
        for posterior in posteriors[1:]
    )
    passed &= largest_order_error <= 1e-12
    return ValidationCheck(
        "8.2 posterior normalization, reference agreement, and order invariance",
        passed,
        {
            "evidence_records": len(records),
            "orderings": len(orderings),
            "largest_reference_error": largest_reference_error,
            "largest_order_error": largest_order_error,
        },
    )


def _validate_quick_detailed_and_duplicates(seed: int) -> ValidationCheck:
    world = generate_incident_world(n_services=4, incident_seed=seed)
    quick_action = probe_action_index(2, "metrics.quick")
    detailed_action = probe_action_index(2, "metrics.detailed")

    combined_ledger = EvidenceLedger()
    combined_estimator = ExactBeliefEstimator(world.graph)
    quick = execute_probe_action(
        world,
        quick_action,
        remaining_credits=8,
        seen_evidence_ids=combined_ledger.evidence_ids,
    )
    combined_estimator.update(combined_ledger.add(quick.evidence))
    detailed = execute_probe_action(
        world,
        detailed_action,
        remaining_credits=8 - quick.cost.credits,
        seen_evidence_ids=combined_ledger.evidence_ids,
    )
    combined_estimator.update(combined_ledger.add(detailed.evidence))

    direct_ledger = EvidenceLedger()
    direct_estimator = ExactBeliefEstimator(world.graph)
    direct = execute_probe_action(
        world,
        detailed_action,
        remaining_credits=8,
        seen_evidence_ids=direct_ledger.evidence_ids,
    )
    direct_estimator.update(direct_ledger.add(direct.evidence))

    before_duplicate = combined_estimator.posterior
    repeated_backend_records = collect_probe_evidence(world, "metrics.quick", 2)
    ledger_added = combined_ledger.add(repeated_backend_records)
    estimator_added = combined_estimator.update(repeated_backend_records)
    duplicate_error = float(
        np.max(np.abs(before_duplicate - combined_estimator.posterior))
    )
    posterior_error = float(
        np.max(np.abs(combined_estimator.posterior - direct_estimator.posterior))
    )
    combined_cost = quick.cost.credits + detailed.cost.credits
    direct_cost = direct.cost.credits
    passed = (
        posterior_error <= 1e-12
        and combined_cost > direct_cost
        and ledger_added == ()
        and estimator_added == ()
        and duplicate_error == 0.0
    )
    return ValidationCheck(
        "8.2 quick-to-detailed coverage and backend duplicate protection",
        passed,
        {
            "posterior_error": posterior_error,
            "quick_then_detailed_cost": combined_cost,
            "detailed_only_cost": direct_cost,
            "duplicate_posterior_error": duplicate_error,
            "unique_evidence_records": len(combined_ledger),
        },
    )


def _validate_shared_workload_marginalization() -> ValidationCheck:
    graph = DependencyGraph.from_edges(1, [], entry_service=0)
    records = (
        MetricEvidenceRecord(0, MetricCategory.CPU_HIGH, 0, True),
        LogEvidenceRecord(0, 0, LogCategory.RESOURCE_PRESSURE),
    )
    estimator = ExactBeliefEstimator(graph)
    estimator.update(records)
    actual = estimator.belief_service_fault[0]

    correct = np.zeros(len(FaultType), dtype=np.float64)
    incorrect = np.zeros(len(FaultType), dtype=np.float64)
    relationship = ServiceRelationship(RelationshipKind.CAUSE, 0)
    for fault in FaultType:
        per_workload = []
        per_record_likelihoods = []
        for workload in Workload:
            likelihoods = (
                metric_high_probability(
                    relationship,
                    fault,
                    workload,
                    records[0].metric,
                ),
                log_category_probability(
                    relationship,
                    fault,
                    workload,
                    records[1].value,
                ),
            )
            per_workload.append(float(np.prod(likelihoods)))
            per_record_likelihoods.append(likelihoods)
        correct[fault] = float(np.mean(per_workload))
        incorrect[fault] = float(
            np.prod(np.mean(per_record_likelihoods, axis=0))
        )
    correct /= correct.sum()
    incorrect /= incorrect.sum()

    estimator_error = float(np.max(np.abs(actual - correct)))
    independence_error = float(np.max(np.abs(actual - incorrect)))
    passed = estimator_error <= 1e-12 and independence_error > 1e-5
    return ValidationCheck(
        "8.2 shared-workload two-record marginalization",
        passed,
        {
            "estimator_error": estimator_error,
            "incorrect_independence_error": independence_error,
            "estimator_fault_posterior": actual.tolist(),
            "hand_enumerated_fault_posterior": correct.tolist(),
            "incorrect_fault_posterior": incorrect.tolist(),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--observation-samples",
        type=int,
        default=MIN_OBSERVATION_SAMPLES,
    )
    parser.add_argument("--prior-samples", type=int, default=DEFAULT_PRIOR_SAMPLES)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    report = run_validation(
        seed=arguments.seed,
        observation_samples=arguments.observation_samples,
        prior_samples=arguments.prior_samples,
    )
    rendered = json.dumps(report.as_dict(), indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
