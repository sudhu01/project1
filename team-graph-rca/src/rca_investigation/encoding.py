"""Training-fitted encoder artifacts and immutable derived packet bundles."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from .contracts import canonical, digest
from .contracts_v2 import ToolResultV2, VOCABULARIES, UNITS, object_json
from .metrics.common import percentile
from .dataset import atomic_json, write_bundle
from .capture_store import read_raw


@dataclass(frozen=True)
class Encoder:
    canonical_json: str

    @property
    def values(self):
        return json.loads(self.canonical_json)

    @property
    def hash(self):
        return digest(self.values)

    @classmethod
    def fit(cls, records, split_manifest):
        if split_manifest['group'] != 'capture_session':
            raise ValueError('encoder requires independent session splits')
        training = set(split_manifest['splits']['train'])
        ids = [record['incident_id'] for record in records]
        if not ids or len(set(ids)) != len(ids) or set(ids) != training:
            raise ValueError('encoder fitting requires exactly frozen training incidents')
        # Each sample must come from a declared healthy training control.
        healthy = [record for record in records if record['private_cause'] == 'NO_FAULT']
        if not healthy:
            raise ValueError('healthy training controls required')
        values = {}
        per_resource = {}
        for family in ('cpu', 'request_errors', 'transport_errors', 'latency', 'memory', 'memory_growth'):
            samples = [number for record in healthy for number in record['fitting_values'].get(family, [])]
            if samples:
                from .contracts import real
                for number in samples:
                    real(number)
                values[family] = percentile(samples, .95)
            resources = {resource for record in healthy for resource in record.get('fitting_values_by_resource', {}).get(family, {})}
            for resource in sorted(resources):
                measured = [number for record in healthy for number in record.get('fitting_values_by_resource', {}).get(family, {}).get(resource, [])]
                if measured:
                    from .contracts import real
                    for number in measured:
                        real(number)
                    per_resource.setdefault(family, {})[resource] = percentile(measured, .95)
        return cls(canonical({'schema_version': 2, 'source_version': 'adaptive_raw_v2',
                              'training_incidents': sorted(ids), 'split_hash': split_manifest['hash'],
                              'thresholds': values, 'thresholds_by_resource': per_resource, 'vocabularies': VOCABULARIES, 'units': UNITS,
                              'minimum_coverage': .8, 'sample_interval_seconds': 2,
                              'minimum_requests': 20, 'log_mapping_version': 'validated_training_peers_v2',
                              'threshold_rules': {'percentile': .95, 'interpolation': 'linear_index_n_minus_one',
                                                  'latency': 'healthy_window_successful_checkout_p95'},
                              'sources': {'cpu': 'docker_cli_quota_normalized', 'memory': 'docker_cli_stats_linux_v2',
                                          'requests': 'frontend_workload_checkout', 'latency': 'successful_frontend_checkout',
                                          'logs': 'verified_loki_historical_error_query'},
                              'validated_log_peers': {resource: sorted({peer for record in records
                                  for peer in record.get('observed_log_peers', {}).get(resource, [])})
                                  for resource in sorted({resource for record in records for resource in record.get('observed_log_peers', {})})}}))

    def save(self, destination):
        from pathlib import Path
        if Path(destination).exists():
            existing = type(self).load(destination)
            if existing.hash != self.hash:
                raise ValueError('fitted encoder is immutable; declare a new artifact path')
            return
        atomic_json(destination, {'encoder': self.values, 'hash': self.hash})

    @classmethod
    def load(cls, source):
        from pathlib import Path
        payload = json.loads(Path(source).read_text(encoding='utf-8'))
        encoder = cls(canonical(payload['encoder']))
        if encoder.hash != payload['hash']:
            raise ValueError('encoder checksum mismatch')
        values = encoder.values
        expected = {'schema_version', 'source_version', 'training_incidents', 'split_hash', 'thresholds',
                    'thresholds_by_resource', 'vocabularies', 'units', 'minimum_coverage', 'sample_interval_seconds',
                    'minimum_requests', 'log_mapping_version', 'threshold_rules', 'sources', 'validated_log_peers'}
        if set(values) != expected or values['schema_version'] != 2 or type(values['schema_version']) is not int:
            raise ValueError('unsupported fitted encoder schema')
        if values['source_version'] != 'adaptive_raw_v2' or values['vocabularies'] != {k: list(v) for k, v in VOCABULARIES.items()} or values['units'] != UNITS:
            raise ValueError('encoder source or vocabulary mismatch')
        if (values['minimum_requests'], values['minimum_coverage'], values['sample_interval_seconds'], values['log_mapping_version']) != (20, .8, 2, 'validated_training_peers_v2'):
            raise ValueError('encoder coverage or mapping rule changed')
        if values['threshold_rules'] != {'percentile': .95, 'interpolation': 'linear_index_n_minus_one',
                                        'latency': 'healthy_window_successful_checkout_p95'}:
            raise ValueError('encoder threshold rule changed')
        from .contracts import identifier, real, unique
        from .contracts_v2 import sha256
        sha256(values['split_hash'])
        unique(tuple(values['training_incidents']))
        if not values['training_incidents']:
            raise ValueError('encoder requires training provenance')
        families = {'cpu', 'request_errors', 'transport_errors', 'latency', 'memory', 'memory_growth'}
        if set(values['thresholds']) - families or set(values['thresholds_by_resource']) - families:
            raise ValueError('unknown encoder threshold family')
        for number in values['thresholds'].values():
            real(number)
        for mapping in values['thresholds_by_resource'].values():
            for resource, number in mapping.items():
                identifier(resource)
                real(number)
        from .topology import sockshop_topology
        for caller, peers in values['validated_log_peers'].items():
            unique(tuple(peers))
            if any((caller, peer) not in sockshop_topology().calls for peer in peers):
                raise ValueError('encoder contains an unregistered observed peer')
        return encoder


def encode_packet(spec, incident, summary, source, collected_at=None, duration_ms=0, sample_time=None):
    availability, code, evidence, reason = summary
    result = ToolResultV2(spec.action_id, spec.resource_id, availability, code,
                          incident.window_start, incident.window_end, sample_time or incident.window_end, source, duration_ms,
                          collected_at=collected_at or datetime.now(timezone.utc).isoformat(),
                          encoding_hash=spec.encoding_hash, evidence_json=canonical(evidence), reason=reason)
    result.validate_context(spec, incident)
    return result


def derive_bundle(destination, raw_path, public, topology, specs, encoder, project):
    payloads, raw_hash = read_raw(raw_path, public_only=False)
    if public.encoding_hash != encoder.hash or any(s.encoding_hash != encoder.hash for s in specs):
        raise ValueError('derived encoding mismatch')
    raw_public = payloads['public.json']
    if any(raw_public[key] != getattr(public, key) for key in ('window_start', 'window_end')):
        raise ValueError('raw window mismatch')
    if raw_public.get('incident_id') != public.incident_id:
        raise ValueError('raw incident mismatch')
    packets = tuple(project(spec, payloads['measurements.json'], public, encoder) for spec in specs)
    from .contracts_v2 import decode_record
    private = decode_record(payloads['private_label.json'], 'private')
    receipt = {**payloads['capture_receipt.json'], 'raw_hash': raw_hash, 'encoder_hash': encoder.hash,
               'catalogue_hash': public.catalogue_hash}
    return write_bundle(destination, public, topology, packets, private, receipt)
