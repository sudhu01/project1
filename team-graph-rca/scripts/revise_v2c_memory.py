"""Declare and fit a separate memory boundary from frozen training windows only."""
from pathlib import Path
import hashlib
import json
from statistics import median
import yaml
from rca_investigation.contracts import canonical, digest
from rca_investigation.contracts_v2 import decode_record
from rca_investigation.dataset import atomic_json
from rca_investigation.encoding import Encoder
from rca_investigation.metrics.common import covered
from rca_investigation.raw_dataset import load_raw_split
from rca_investigation.train import runtime_hash


def fit_boundary(records, training_ids):
    if len(records) != len(training_ids) or {r['incident_id'] for r in records} != set(training_ids):
        raise ValueError('exact frozen training cohort required')
    healthy, fault = [], []
    healthy_sessions, fault_sessions = set(), set()
    for row in records:
        value = row.get('fraction')
        if value is None:
            continue
        if not 0 <= value <= 1:
            raise ValueError('invalid window memory fraction')
        if row['cause'] == 'NO_FAULT':
            healthy.append(value)
            healthy_sessions.add(row['session'])
        elif row['cause'] == 'resource/carts':
            if not row['effect_confirmed'] or not row['recovery_confirmed']:
                raise ValueError('verified memory effect and recovery required')
            fault.append(value)
            fault_sessions.add(row['session'])
    if min(len(healthy_sessions), len(fault_sessions)) < 5:
        raise ValueError('at least five independent training sessions per group required')
    if max(healthy) >= min(fault):
        raise ValueError('training memory groups overlap; do not invent a boundary')
    return {'healthy_upper': max(healthy), 'fault_lower': min(fault),
            'threshold': (max(healthy) + min(fault)) / 2,
            'healthy_windows': len(healthy), 'fault_windows': len(fault),
            'healthy_sessions': len(healthy_sessions), 'fault_sessions': len(fault_sessions)}


def main():
    root = Path.cwd()
    run = root / 'runs/adaptive_v2c_memory_separation_r1'
    config_path = root / 'configs/investigation_v2c_memory_separation_r1.yaml'
    raw = root / 'datasets/raw_v2c'
    frozen = json.loads((raw / 'splits.json').read_text())
    original = Encoder.load(root / 'experiments/v2c/encoder.json')
    # Record the algorithm before fitting or reading validation measurements.
    declaration = dict(schema_version=1, revision='v2c_memory_separation_r1',
        rule='midpoint_of_max_healthy_and_min_verified_pressure_training_window_medians',
        resource='carts', negative_class='NO_FAULT', positive_class='resource/carts',
        minimum_independent_sessions=5, overlap_behavior='fail_closed',
        other_encoder_values='retain_original', raw_split_hash=frozen['hash'],
        original_encoder_hash=original.hash, runtime_hash=runtime_hash(),
        fitter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_gate_accuracy=0.9, fitting_partition='train',
        validation_used_for_fitting=False, reserved_test_used_for_fitting=False,
        original_failed_run='runs/adaptive_v2c')
    declaration['hash'] = digest(declaration)
    target = run / 'revision-declaration.json'
    if target.exists() and json.loads(target.read_text()) != declaration:
        raise ValueError('revision declaration changed; declare a new identity')
    atomic_json(target, declaration)
    config = yaml.safe_load((root / 'configs/investigation_v2c.yaml').read_text())
    # Keep the registered physical task; paths and encoder hash identify revision.
    config['paths'].update(dataset='datasets/adaptive_v2c_memory_separation_r1',
        encoder='experiments/v2c_memory_separation_r1/encoder.json',
        model='runs/adaptive_v2c_memory_separation_r1/model.json',
        runs='runs/adaptive_v2c_memory_separation_r1')
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
        raise ValueError('revision config changed')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    manifest, rows = load_raw_split(raw, 'train')
    records = []
    for incident, payload in rows:
        public, receipt = payload['public.json'], payload['capture_receipt.json']
        samples = [s for s in payload['measurements.json']['stats']
                   if s['resource_id'] == 'carts' and s['availability'] == 'OK']
        _, sufficient = covered(samples, public['window_start'], public['window_end'])
        fractions = [s['memory_bytes'] / s['memory_limit_bytes'] for s in samples]
        records.append(dict(incident_id=incident,
            cause=decode_record(payload['private_label.json'], 'private').cause_id,
            session=receipt['capture_session'], effect_confirmed=receipt['effect_confirmed'],
            recovery_confirmed=receipt['recovery_confirmed'],
            fraction=median(fractions) if sufficient else None))
    fitted = fit_boundary(records, manifest['splits']['train'])
    values = original.values
    values['thresholds_by_resource']['memory']['carts'] = fitted['threshold']
    # The legacy rules specify healthy percentile fitting for other families.
    # This revision's memory override and its provenance are frozen in the sidecar.
    revised = Encoder(canonical(values))
    revised.save(root / config['paths']['encoder'])
    report = dict(passed=True, **fitted, encoder_hash=revised.hash,
                  declaration_hash=declaration['hash'], training_incidents=sorted(manifest['splits']['train']),
                  validation_opened=False, reserved_test_opened=False)
    atomic_json(run / 'memory-boundary-fit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'training_incidents'}, indent=2))


if __name__ == '__main__':
    main()
