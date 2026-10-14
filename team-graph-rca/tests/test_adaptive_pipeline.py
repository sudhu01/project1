"""Synthetic source fixtures verify the frozen pipeline, never physical claims."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import yaml
import pytest
from rca_investigation.contracts import canonical, digest
from rca_investigation.contracts_v2 import IncidentPrivateV2
from rca_investigation.capture_store import write_raw
from rca_investigation.raw_dataset import split_raw, fit_raw_encoder
from rca_investigation.config import load_config
from rca_investigation.catalogue import build_catalogue
from rca_investigation.topology import sockshop_topology
from rca_investigation.derived_dataset import encode_dataset, full_evidence_reference
from rca_investigation.experiment import fit_dataset
from rca_investigation.ablations import declare_ablations


def test_raw_encoder_derived_model_reference_and_ablations(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / 'configs/investigation_v2a.yaml').read_text())
    monkeypatch.chdir(tmp_path)
    topology = sockshop_topology()
    resources = value['resources']
    registry = {resource: {'container_id': f'{index+1:064x}', 'cpu_quota_cores': 2.,
                          'memory_limit_bytes': 512 * 1024**2} for index, resource in enumerate(resources)}
    Path('configs').mkdir()
    Path('configs/registry.yaml').write_text(yaml.safe_dump({'resources': registry}))
    value['paths']['resource_registry'] = 'configs/registry.yaml'
    Path('configs/task.yaml').write_text(yaml.safe_dump(value))
    accepted, roles = {}, {}
    raw_root = Path(value['paths']['raw_dataset'])
    for partition in ('train', 'validation', 'test'):
        roles[partition] = partition
        for index, cause in enumerate(value['diagnoses']):
            incident = partition + '-' + str(index)
            start = datetime(2026, 10, 7, tzinfo=timezone.utc) + timedelta(minutes=3 * index)
            end = start + timedelta(seconds=120)
            affected = (cause.split('/')[1],) if cause != 'NO_FAULT' else ()
            mechanism = 'cpu_contention' if cause == 'resource/payment' else 'container_stopped' if affected else 'healthy'
            private = IncidentPrivateV2(2, incident, cause, 'resource' if affected else 'healthy', mechanism,
                                         affected, '{}', '{}', '{}')
            status = {resource: {'container_id': item['container_id'], 'state': 'exited' if resource in affected and mechanism == 'container_stopped' else 'running',
                                'sample_time': (start + timedelta(seconds=1)).isoformat()} for resource, item in registry.items()}
            stats = []
            for resource, item in registry.items():
                for number in range(60):
                    stats.append({'resource_id': resource, 'container_id': item['container_id'], 'availability': 'OK',
                        'sample_time': (start + timedelta(seconds=number*2)).isoformat(), 'units': 'docker_cpu_percent',
                        'quota_cores': 2., 'cpu_percent': 190. if cause == 'resource/payment' and resource == 'payment' else 10.,
                        'memory_bytes': 100 * 1024**2, 'memory_limit_bytes': item['memory_limit_bytes'],
                        'memory_source': 'docker_cli_stats_linux_v2', 'memory_usage_original': '100MiB / 512MiB'})
            requests = [{'resource_id': 'front-end', 'operation': 'checkout', 'outcome': 'completed',
                'http_status': 500 if mechanism == 'container_stopped' else 201, 'duration_ms': 50,
                'sample_time': (start + timedelta(seconds=number*2)).isoformat()} for number in range(40)]
            measurements = {'status': status, 'stats': stats, 'requests': requests, 'registry': registry,
                'logs': {resource: {'availability': 'OK', 'lines': []} for resource in resources},
                'collected_at': (end + timedelta(seconds=10)).isoformat()}
            public = {'incident_id': incident, 'task_id': value['task_id'], 'topology_hash': topology.hash,
                      'registry_hash': digest(registry), 'window_start': start.isoformat(), 'window_end': end.isoformat()}
            receipt = {'status': 'ACCEPTED', 'recovery_confirmed': True, 'capture_session': partition,
                       'physical_episode': incident, 'synthetic_source_fixture': True}
            write_raw(raw_root, incident, public, measurements, private, receipt)
            accepted[incident] = incident
    state = {'complete': True, 'roles': roles, 'accepted': accepted, 'schedule_hash': digest('synthetic-schedule')}
    Path('state.json').write_text(canonical(state))
    manifest = split_raw(raw_root, 'state.json')
    encoder = fit_raw_encoder(raw_root, value['paths']['encoder'])
    assert encoder.values['training_incidents'] == sorted(manifest['splits']['train'])
    assert encoder.values['thresholds']['memory_growth'] == 0
    config = load_config('configs/task.yaml', topology)
    specs = build_catalogue(config, topology)
    assert all('/' not in symbol for spec in specs if spec.template == 'logs' for symbol in spec.responses)
    result = encode_dataset(config, topology, specs)
    assert result['derived_incidents'] == 12
    from rca_investigation.dataset import load_public_bundle
    public, _, packets = load_public_bundle(Path(value['paths']['dataset']) / 'train-0', specs)
    assert all(packet.sample_time != public.window_end for packet in packets if packet.action_id.startswith('status/'))
    fit_dataset(value['paths']['dataset'], config, specs, value['paths']['model'])
    fit_dataset(value['paths']['dataset'], config, specs, value['paths']['model'])
    changed = json.loads(config.canonical_json)
    changed['model']['smoothing'] = .2
    Path('configs/changed.yaml').write_text(yaml.safe_dump(changed))
    changed_config = load_config('configs/changed.yaml', topology)
    with pytest.raises(ValueError, match='immutable'):
        fit_dataset(value['paths']['dataset'], changed_config, specs, value['paths']['model'])
    reference = full_evidence_reference(config, specs)
    assert reference['accuracy'] == 1 and reference['training_promotion_passed']
    assert reference['limited_budget_policy'] is False
    with pytest.raises(ValueError, match='already exists'):
        encode_dataset(config, topology, specs)
    ablations = declare_ablations(config, topology, specs, 'configs/ablations')
    status = next(variant for variant in ablations['variants'] if variant['name'] == 'status')
    variant = load_config(status['config'], topology)
    subset = build_catalogue(variant, topology)
    encode_dataset(variant, topology, subset)
    fit_dataset(variant.values['paths']['dataset'], variant, subset, variant.values['paths']['model'])
    reference = full_evidence_reference(variant, subset)
    assert reference['accuracy'] == .75 and not reference['training_promotion_passed']
    assert variant.hash != config.hash and variant.values['diagnoses'] == config.values['diagnoses']


def test_paired_mixture_resamples_sessions_and_preserves_priors():
    from rca_investigation.adaptive_comparison import paired_mixture_sweep
    first, second = [], []
    for session in range(5):
        for hard in (False, True):
            row = {'incident_id': f'{session}-{hard}', 'capture_session': str(session),
                   'truth': 'resource/payment' if hard else 'NO_FAULT', 'return_value': .1 if hard else -.1}
            first.append(row)
            second.append({**row, 'return_value': 0.})
    result = paired_mixture_sweep(first, second, {'resource/payment'}, [0., .5, 1.], samples=50)
    assert [row['paired_return_difference'] for row in result] == pytest.approx([-.1, 0., .1])
    assert all(row['independent_sessions'] == 5 and not row['priors_changed'] for row in result)


def test_validation_reference_rejects_a_different_model_and_keeps_failed_gate(tmp_path):
    from types import SimpleNamespace
    from rca_investigation.comparison import validation_reference
    config = SimpleNamespace(hash='task', values={'paths': {'runs': str(tmp_path)}})
    model = SimpleNamespace(hash='model')
    with pytest.raises(ValueError, match='reference required'):
        validation_reference(config, model, 'split')
    report = {'config_hash': 'task', 'model_hash': 'model', 'split_hash': 'split',
              'training_promotion_passed': False}
    path = tmp_path / 'full-evidence-reference.json'
    path.write_text(json.dumps(report))
    assert not validation_reference(config, model, 'split')['training_promotion_passed']
    report['model_hash'] = 'another-model'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='different experiment'):
        validation_reference(config, model, 'split')


def test_fixed_search_separates_classes_without_spending_its_prefix_on_duplicates():
    from types import SimpleNamespace
    from rca_investigation.adaptive_comparison import fixed_candidates
    from rca_investigation.belief import PrototypeModel
    ids = ('status/a', 'status/b', 'status/duplicate', 'latency', 'cpu', 'memory')
    rows = tuple((str(index == 0), str(index == 1), str(index == 0),
                  'missing' if index < 2 else 'slow' if index < 4 else 'normal',
                  str(index == 2), str(index == 4)) for index in range(6))
    labels = tuple('ABCDEF')
    model = PrototypeModel(labels, ids, tuple(tuple(sorted({row[i] for row in rows})) for i in range(6)),
                           labels, rows, (1/6,) * 6, .1)
    specs = tuple(SimpleNamespace(action_id=action, credits=1 if action.startswith('status') else 2) for action in ids)
    candidates = fixed_candidates(specs, model=model)
    assert len(candidates) <= 32 and len(set(candidates)) == len(candidates)
    useful = {'latency', 'status/a', 'cpu', 'memory'}
    orders = [order for order in candidates if set(order[:4]) == useful]
    assert orders and {order[0] for order in orders} == useful
    for label, row in zip(labels, rows):
        history = tuple((action, row[ids.index(action)]) for action in orders[0][:4])
        assert max(model.diagnosis(history), key=model.diagnosis(history).get) == label
