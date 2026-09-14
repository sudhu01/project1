"""Tests for the acquired evidence ledger."""

import pytest

from rca_sim.contracts import MetricCategory, MetricEvidenceRecord
from rca_sim.evidence import EvidenceConflictError, EvidenceLedger


def metric_record(
    *,
    service_id: int = 0,
    reading_index: int = 0,
    value: bool = True,
) -> MetricEvidenceRecord:
    return MetricEvidenceRecord(
        service_id=service_id,
        metric=MetricCategory.CPU_HIGH,
        reading_index=reading_index,
        value=value,
    )


def test_ledger_adds_unseen_records_and_deduplicates_exact_repeats() -> None:
    ledger = EvidenceLedger()
    first = metric_record()
    second = metric_record(reading_index=1, value=False)

    assert ledger.add((first, second)) == (first, second)
    assert ledger.add((first,)) == ()
    assert len(ledger) == 2
    assert ledger.evidence_ids == (first.evidence_id, second.evidence_id)
    assert ledger.records == (first, second)
    assert ledger.get(first.evidence_id) is first


def test_ledger_rejects_conflicting_value_without_partial_add() -> None:
    ledger = EvidenceLedger()
    original = metric_record(value=True)
    unseen = metric_record(service_id=1, value=False)
    conflict = metric_record(value=False)
    ledger.add((original,))

    with pytest.raises(EvidenceConflictError, match="conflicting values"):
        ledger.add((unseen, conflict))

    assert ledger.records == (original,)


def test_ledger_rejects_conflicts_within_one_batch_atomically() -> None:
    ledger = EvidenceLedger()
    first = metric_record(value=True)
    conflict = metric_record(value=False)

    with pytest.raises(EvidenceConflictError):
        ledger.add((first, conflict))

    assert len(ledger) == 0


def test_ledger_mapping_view_is_read_only() -> None:
    ledger = EvidenceLedger()
    record = metric_record()
    ledger.add((record,))

    with pytest.raises(TypeError):
        ledger.by_id[record.evidence_id] = record  # type: ignore[index]
