"""Deduplicated storage for evidence acquired during an investigation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import MappingProxyType
from typing import Mapping

from rca_sim.contracts import EvidenceRecord


class EvidenceConflictError(ValueError):
    """An evidence ID was reused for a different record."""


class EvidenceLedger:
    """Store one immutable record per evidence ID in acquisition order."""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[EvidenceRecord]:
        return iter(self._records.values())

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._records

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return acquired IDs in first-seen order."""
        return tuple(self._records)

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        """Return acquired records in first-seen order."""
        return tuple(self._records.values())

    @property
    def by_id(self) -> Mapping[str, EvidenceRecord]:
        """Return a live, read-only view of records keyed by evidence ID."""
        return MappingProxyType(self._records)

    def get(self, evidence_id: str) -> EvidenceRecord:
        """Return an acquired record by ID."""
        if not isinstance(evidence_id, str):
            raise TypeError("evidence_id must be a string")
        try:
            return self._records[evidence_id]
        except KeyError as error:
            raise KeyError(f"unknown acquired evidence ID: {evidence_id}") from error

    def add(self, records: Iterable[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
        """Atomically add records and return only records that were unseen.

        Exact repeats are ignored. Reusing an ID for a different record rejects
        the full batch before the ledger changes.
        """
        try:
            batch = tuple(records)
        except TypeError as error:
            raise TypeError("records must be an iterable of evidence records") from error

        pending: dict[str, EvidenceRecord] = {}
        for record in batch:
            if not _is_evidence_record(record):
                raise TypeError("records must contain only evidence records")
            evidence_id = record.evidence_id
            existing = pending.get(evidence_id, self._records.get(evidence_id))
            if existing is not None and existing != record:
                raise EvidenceConflictError(
                    f"evidence ID {evidence_id!r} has conflicting values"
                )
            if existing is None:
                pending[evidence_id] = record

        self._records.update(pending)
        return tuple(pending.values())


def _is_evidence_record(record: object) -> bool:
    from rca_sim.contracts import LogEvidenceRecord, MetricEvidenceRecord

    return isinstance(record, (MetricEvidenceRecord, LogEvidenceRecord))
