import json
from pathlib import Path
import pytest
import yaml
from rca_investigation import raw_schedule
from rca_investigation.contracts import digest
from rca_investigation.capture_store import read_raw


@pytest.fixture
def capture(tmp_path, monkeypatch):
    resources = ['front-end', 'catalogue', 'catalogue-db', 'orders', 'payment']
    registry = {r: {'container_id': f'{i+1:064x}', 'cpu_quota_cores': .25, 'memory_limit_bytes': 512*1024**2}
                for i, r in enumerate(resources)}
    path = tmp_path / 'runs/adaptive_v2'
    path.mkdir(parents=True)
    (path / 'capabilities.json').write_text(json.dumps({'passed': True, 'resources': registry, 'hash': digest(registry)}))
    (path / 'cpu-pilot-tail-confirmation.json').write_text('{"passed":true}')
    config = {'task_id': 'sockshop_adaptive_v2a', 'resources': resources,
              'paths': {'runs': 'runs/adaptive_v2a', 'raw_dataset': 'datasets/raw_v2a'}}
    cfg, schedule = tmp_path / 'config.yaml', tmp_path / 'schedule.yaml'
    cfg.write_text(yaml.safe_dump(config))
    schedule.write_text(yaml.safe_dump({'task_id': config['task_id'], 'lab_batch': 'unit-only',
                        'sessions': [{'id': 'session1', 'role': 'pilot', 'scenarios': ['payment_cpu']}]}))
    class Driver:
        fail_injection = False
        fail_recovery = False
        calls = []
        def __init__(self, topology, root):
            self.topology, self.ids = topology, {r: item['container_id'] for r, item in registry.items()}
        def resolve(self): return self.ids
        def states(self): return {r: 'running' for r in resources}
        def inject_helper(self, *args, **kwargs):
            self.calls.append('inject')
            if self.fail_injection:
                raise KeyboardInterrupt()
            return {'verified_dispatch': True}
        def stop_helper(self, resource):
            self.calls.append('recover')
            if self.fail_recovery:
                raise RuntimeError('cleanup_failed')
            return {'terminated': True}
    class Workload:
        def prepare(self): pass
        def checkout(self): return {'http_status': 201, 'duration_ms': 10, 'outcome': 'completed'}
        def reset(self): pass
        def close(self): pass
    monkeypatch.setattr(raw_schedule, 'AdaptiveLabDriver', Driver)
    monkeypatch.setattr(raw_schedule, 'CheckoutWorkload', Workload)
    monkeypatch.setattr(raw_schedule, 'ContainerStatsCollector', lambda registry: object())
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    monkeypatch.setattr(raw_schedule, 'settle_states', lambda *args: None)
    from datetime import datetime, timedelta, timezone
    start = datetime(2026, 10, 7, 10, tzinfo=timezone.utc)
    raw = {'stats': [{'resource_id': 'payment', 'container_id': registry['payment']['container_id'],
                     'cpu_percent': 25, 'quota_cores': .25, 'availability': 'OK', 'units': 'docker_cpu_percent',
                     'sample_time': (start + timedelta(seconds=2 * i)).isoformat()} for i in range(60)],
           'requests': [{'http_status': 500, 'duration_ms': 200}]}
    def collect(*args, **kwargs):
        data = {'requests': [Workload().checkout() for _ in range(20)]} if kwargs.get('ingestion_seconds') == 0 else raw
        return {'window_start': '2026-10-07T10:00:00Z', 'window_end': '2026-10-07T10:02:00Z'}, data
    monkeypatch.setattr(raw_schedule, 'collect_window', collect)
    return tmp_path, cfg, schedule, Driver


def test_raw_capture_accepted_pilot_is_separate_and_resumes(capture):
    root, config, schedule, driver = capture
    state = raw_schedule.capture_raw(root, config, schedule)
    assert state['complete'] and len(state['accepted']) == 1
    assert driver.calls == ['inject', 'recover']
    identifier = next(iter(state['accepted'].values()))
    read_raw(root / 'datasets/raw_v2a_pilot' / identifier, public_only=False)
    assert not (root / 'datasets/raw_v2a').exists()
    assert raw_schedule.capture_raw(root, config, schedule) == state
    assert driver.calls == ['inject', 'recover']


def test_interruption_recovers_and_leaves_no_unverified_label(capture):
    root, config, schedule, driver = capture
    driver.fail_injection = True
    with pytest.raises(KeyboardInterrupt):
        raw_schedule.capture_raw(root, config, schedule)
    assert driver.calls == ['inject', 'recover']
    assert not (root / 'runs/adaptive_v2/lab/blocked-lab.json').exists()
    assert not (root / 'datasets/raw_v2a_pilot').exists()


def test_failed_baseline_records_requests_before_any_injection(capture, monkeypatch):
    root, config, schedule, driver = capture
    original = raw_schedule.collect_window
    def degraded_baseline(*args, **kwargs):
        window, data = original(*args, **kwargs)
        if kwargs.get('ingestion_seconds') == 0:
            data['requests'][0]['http_status'] = 503
        return window, data
    monkeypatch.setattr(raw_schedule, 'collect_window', degraded_baseline)
    with pytest.raises(RuntimeError, match='healthy capture baseline failed'):
        raw_schedule.capture_raw(root, config, schedule)
    audit = root / 'runs/adaptive_v2a/capture_pilot'
    receipt = json.loads(next((audit / 'attempts').glob('*.json')).read_text())
    assert receipt['baseline'][0]['http_status'] == 503
    assert receipt['baseline_window']['window_start'] == '2026-10-07T10:00:00Z'
    assert receipt['failure_message'] == 'healthy capture baseline failed'
    assert receipt['recovery_confirmed'] and receipt['status'] == 'REJECTED'
    assert driver.calls == []
    assert not (root / 'datasets/raw_v2a_pilot').exists()


def test_failed_recovery_blocks_next_injection(capture):
    root, config, schedule, driver = capture
    driver.fail_recovery = True
    with pytest.raises(RuntimeError, match='cleanup_failed'):
        raw_schedule.capture_raw(root, config, schedule)
    audit = root / 'runs/adaptive_v2a/capture_pilot'
    receipt = json.loads(next((audit / 'attempts').glob('*.json')).read_text())
    observation = json.loads((audit / 'rejected-observations' / (receipt['incident_id'] + '.json')).read_text())
    assert receipt['recovery_failure'] == {'type': 'RuntimeError', 'message': 'cleanup_failed'}
    assert receipt['observation_audit_hash'] == digest(observation)
    assert observation['measurements']['stats']
    assert not (root / 'datasets/raw_v2a_pilot').exists()
    assert (root / 'runs/adaptive_v2/lab/blocked-lab.json').exists()
    before = list(driver.calls)
    with pytest.raises(RuntimeError, match='recovery'):
        raw_schedule.capture_raw(root, config, schedule)
    assert driver.calls == before


@pytest.mark.parametrize('oversized', [False, True])
def test_rejected_effect_retains_observations_without_publishing(capture, monkeypatch, oversized):
    root, config, schedule, driver = capture
    original = raw_schedule.collect_window
    def without_contention(*args, **kwargs):
        window, data = original(*args, **kwargs)
        if kwargs.get('ingestion_seconds') != 0:
            for row in data['stats']:
                row['cpu_percent'] = 0
            if oversized:
                data['extra'] = 'x' * (8 * 1024 * 1024)
        return window, data
    monkeypatch.setattr(raw_schedule, 'collect_window', without_contention)
    with pytest.raises(RuntimeError, match='contention'):
        raw_schedule.capture_raw(root, config, schedule)
    audit = root / 'runs/adaptive_v2a/capture_pilot'
    receipt = json.loads(next((audit / 'attempts').glob('*.json')).read_text())
    observation = json.loads((audit / 'rejected-observations' / (receipt['incident_id'] + '.json')).read_text())
    assert receipt['status'] == 'REJECTED' and receipt['recovery_confirmed']
    assert 'contention' in receipt['failure_message']
    assert receipt['observation_audit_hash'] == digest(observation)
    if oversized:
        assert observation['omitted_oversized_measurements']
        assert observation['observation_bytes'] > 8 * 1024 * 1024
        assert 'measurements' not in observation
    else:
        assert all(row['cpu_percent'] == 0 for row in observation['measurements']['stats'])
    assert driver.calls == ['inject', 'recover']
    assert not (root / 'datasets/raw_v2a_pilot').exists()
    state = json.loads((audit / 'schedule.json').read_text())
    assert not state['accepted'] and not state['complete']
