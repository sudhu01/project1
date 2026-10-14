"""Immutable public packets and separately loaded private records."""
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import math
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", value):
        raise ValueError("invalid identifier")


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer")


def real(value, minimum=0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum or maximum is not None and value > maximum:
        raise ValueError("invalid real number")


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp requires a timezone")
    return result


def unique(values):
    if type(values) is not tuple or len(set(values)) != len(values):
        raise ValueError("expected a tuple of unique identifiers")
    for value in values:
        identifier(value)


class Availability(StrEnum):
    OK = "OK"
    NO_DATA = "NO_DATA"
    UNSUPPORTED = "UNSUPPORTED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class JsonRecord:
    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return canonical(self.to_dict())

    @classmethod
    def from_dict(cls, payload):
        if type(payload) is not dict or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError(f"invalid {cls.__name__} fields")
        converted = dict(payload)
        for field in fields(cls):
            if field.type is tuple:
                if type(converted[field.name]) is not list:
                    raise ValueError("serialized tuples must be lists")
                converted[field.name] = tuple(tuple(v) if isinstance(v, list) else v for v in converted[field.name])
        return cls(**converted)

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class TopologySnapshot(JsonRecord):
    version: str
    resources: tuple
    categories: tuple
    calls: tuple

    def __post_init__(self):
        identifier(self.version)
        unique(self.resources)
        if tuple(sorted(self.resources)) != self.resources:
            raise ValueError("resources must be sorted")
        if type(self.categories) is not tuple or len(self.categories) != len(self.resources) or any(v not in {"Service", "Database", "Queue"} for v in self.categories):
            raise ValueError("invalid resource categories")
        if type(self.calls) is not tuple or len(set(self.calls)) != len(self.calls):
            raise ValueError("duplicate edges")
        for edge in self.calls:
            if type(edge) is not tuple or len(edge) != 2 or any(v not in self.resources for v in edge) or edge[0] == edge[1]:
                raise ValueError("invalid CALLS edge")

    @property
    def hash(self):
        return digest(self.to_dict())


@dataclass(frozen=True)
class ToolSpec(JsonRecord):
    action_id: str
    resource_id: str
    template: str
    credits: int
    responses: tuple
    capabilities: tuple
    parameters_json: str = "{}"

    def __post_init__(self):
        identifier(self.action_id)
        identifier(self.resource_id)
        if self.template not in {"status", "cpu", "logs"} or self.action_id != f"{self.template}/{self.resource_id}/v1":
            raise ValueError("invalid tool template or action ID")
        integer(self.credits, 1)
        unique(self.responses)
        unique(self.capabilities)
        params = json.loads(self.parameters_json)
        allowed = {"status": {}, "cpu": {"window_seconds": 60}, "logs": {"max_lines": 20}}[self.template]
        if params != allowed or canonical(params) != self.parameters_json:
            raise ValueError("unsupported parameters")

    def validate_resource(self, topology):
        if self.resource_id not in topology.resources:
            raise ValueError("unknown resource")


@dataclass(frozen=True)
class IncidentPublic(JsonRecord):
    incident_id: str
    topology_hash: str
    catalogue_hash: str
    entry_resource: str
    entry_protocol: str
    window_start: str
    window_end: str
    budget: int
    max_probes: int
    lambda_cost: float

    def __post_init__(self):
        for v in (self.incident_id, self.topology_hash, self.catalogue_hash, self.entry_resource):
            identifier(v)
        if self.entry_protocol != "manual_frontend":
            raise ValueError("unsupported entry protocol")
        if timestamp(self.window_start) >= timestamp(self.window_end):
            raise ValueError("invalid observation window")
        integer(self.budget, 1)
        integer(self.max_probes, 1)
        real(self.lambda_cost)

    def validate_context(self, topology, catalogue):
        if self.topology_hash != topology.hash or self.catalogue_hash != digest([s.to_dict() for s in catalogue]):
            raise ValueError("incident context hash mismatch")
        if self.entry_resource not in topology.resources:
            raise ValueError("unknown entry resource")


@dataclass(frozen=True)
class ToolResult(JsonRecord):
    action_id: str
    resource_id: str
    availability: str
    response_code: str
    window_start: str
    window_end: str
    sample_time: str
    source: str
    duration_ms: int
    raw_json: str = "{}"

    def __post_init__(self):
        for v in (self.action_id, self.resource_id, self.response_code, self.source):
            identifier(v)
        Availability(self.availability)
        start, end, sample = map(timestamp, (self.window_start, self.window_end, self.sample_time))
        if start >= end or not start <= sample <= end:
            raise ValueError("invalid packet timestamps")
        integer(self.duration_ms)
        raw = json.loads(self.raw_json)
        if type(raw) is not dict or len(self.raw_json.encode()) > 32768 or canonical(raw) != self.raw_json:
            raise ValueError("raw packet must be bounded canonical JSON")
        if self.availability != "OK" and self.response_code != "UNAVAILABLE":
            raise ValueError("failed tool cannot report health")

    @property
    def evidence_id(self):
        return digest(self.to_dict())

    @property
    def symbol(self):
        return f"{self.availability}:{self.response_code}"

    def validate_context(self, spec, incident):
        version = getattr(self, 'schema_version', 1)
        if version != getattr(spec, 'schema_version', 1) or version != getattr(incident, 'schema_version', 1):
            raise ValueError('packet, tool and incident schema versions differ')
        if (self.action_id, self.resource_id) != (spec.action_id, spec.resource_id) or self.response_code not in spec.responses:
            raise ValueError("packet does not match requested tool")
        if (self.window_start, self.window_end) != (incident.window_start, incident.window_end):
            raise ValueError("packet window mismatch")


@dataclass(frozen=True)
class IncidentPrivate(JsonRecord):
    incident_id: str
    resources: tuple
    mechanism: str
    injector_receipt_json: str
    verification_json: str
    recovery_json: str

    def __post_init__(self):
        identifier(self.incident_id)
        unique(self.resources)
        if self.mechanism not in {"container_stopped", "healthy", "cpu_contention"}:
            raise ValueError("unsupported fault mechanism")
        for field in (self.injector_receipt_json, self.verification_json, self.recovery_json):
            if type(json.loads(field)) is not dict:
                raise ValueError("receipts must be objects")


@dataclass(frozen=True)
class Decision(JsonRecord):
    prediction: str
    confidence: float
    evidence_ids: tuple
    policy_hash: str
    model_hash: str
    stop_reason: str

    def __post_init__(self):
        for v in (self.prediction, self.policy_hash, self.model_hash, self.stop_reason):
            identifier(v)
        real(self.confidence, 0, 1)
        unique(self.evidence_ids)


@dataclass(frozen=True)
class RunManifest(JsonRecord):
    source_revisions: tuple
    lock_hash: str
    resolved_config_json: str
    split_hashes: tuple
    seeds: tuple
    checkpoint: str
    backend_version: str

    def __post_init__(self):
        if any(type(v) is not tuple for v in (self.source_revisions, self.split_hashes, self.seeds)):
            raise ValueError("manifest collections must be immutable tuples")
        for pairs in (self.source_revisions, self.split_hashes):
            for pair in pairs:
                if type(pair) is not tuple or len(pair) != 2:
                    raise ValueError("manifest entries must be pairs")
                for value in pair:
                    identifier(value)
        for seed in self.seeds:
            integer(seed)
        for value in (self.lock_hash, self.checkpoint, self.backend_version):
            identifier(value)
        if type(json.loads(self.resolved_config_json)) is not dict:
            raise ValueError("invalid resolved settings")
