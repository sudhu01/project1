import importlib.util
from pathlib import Path
import pytest
from rca_investigation import capture_store


@pytest.mark.parametrize('profile,clean,lease,expected', [
    ('v2c', True, 'owned', 0), ('v2c', False, 'owned', 1),
    ('v2b', True, 'owned', 1), ('v2c', True, '', 1)])
def test_memory_effect_retry_requires_owned_v2c_and_confirmed_cleanup(tmp_path, profile, clean, lease, expected):
    import json
    _, module = legacy_driver()
    run = tmp_path / 'runs' / ('adaptive_' + profile)
    attempts = run / 'capture/attempts'
    attempts.mkdir(parents=True)
    (run / 'capture/schedule.json').write_text('{"accepted":{"s/0":"original"}}')
    lab = tmp_path / 'runs/adaptive_v2/lab'
    lab.mkdir(parents=True)
    calls = []
    def capture(output):
        calls.append(1)
        if len(calls) == 1:
            receipt = dict(incident_id='rejected_memory', status='REJECTED',
                recovery_confirmed=clean, failure_message='retained memory pressure absent',
                injector=dict(mode='memory', resource='carts', lease=lease))
            (attempts / 'rejected.json').write_text(json.dumps(receipt))
            output.write_text('{"passed":false,"error":"retained memory pressure absent"}')
            return 1
        output.write_text('{"passed":true}')
        return 0
    def recover():
        (lab / 'manual-recovery.json').write_text('{"passed":true}')
    assert module.capture_with_recovery(tmp_path, capture, recover, profile) == expected
    assert len(calls) == (2 if expected == 0 else 1)
    assert json.loads((run / 'capture/schedule.json').read_text())['accepted'] == {'s/0':'original'}


def test_publication_retries_sharing_lock_without_rewriting(tmp_path, monkeypatch):
    staged, target = tmp_path / 'raw.pending', tmp_path / 'raw'
    staged.mkdir()
    (staged / 'proof').write_bytes(b'immutable observation')
    rename = Path.rename
    attempts = []
    def locked(path, destination):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError('sharing lock')
        return rename(path, destination)
    monkeypatch.setattr(Path, 'rename', locked)
    monkeypatch.setattr(capture_store, 'sleep', lambda seconds: None)
    capture_store.publish_staged(staged, target)
    assert len(attempts) == 3
    assert (target / 'proof').read_bytes() == b'immutable observation'


def test_permanent_publication_failure_preserves_stage(tmp_path, monkeypatch):
    staged, target = tmp_path / 'raw.pending', tmp_path / 'raw'
    staged.mkdir()
    (staged / 'proof').write_bytes(b'keep')
    attempts = []
    def locked(*args):
        attempts.append(1)
        raise PermissionError('persistent lock')
    monkeypatch.setattr(Path, 'rename', locked)
    monkeypatch.setattr(capture_store, 'sleep', lambda seconds: None)
    with pytest.raises(PermissionError):
        capture_store.publish_staged(staged, target)
    assert len(attempts) == 11 and not target.exists()
    assert (staged / 'proof').read_bytes() == b'keep'


def legacy_driver():
    spec = importlib.util.spec_from_file_location('legacy_resume', Path(__file__).resolve().parents[1] / 'scripts/resume_v2a_legacy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = object.__new__(module.LegacyV2aDriver)
    value.root = Path(__file__).resolve().parents[1]
    value.ids, value.helper_leases = {'payment': 'a' * 64}, {}
    value.state = lambda resource: {'state': 'running'}
    value.calls = []
    def command(argv):
        value.calls.append(argv)
        return '', {'argv': argv}
    value.command = command
    return value, module


def test_legacy_resume_preserves_original_pin_and_owns_cleanup():
    value, module = legacy_driver()
    with pytest.raises(ValueError, match='own'):
        value.stop_helper('payment')
    for args in [('carts', 'memory', 210, 8, 0), ('payment', 'cpu', 211, 8, 0)]:
        with pytest.raises(ValueError):
            value.inject_helper(*args)
    assert not value.calls
    receipt = value.inject_helper('payment', 'cpu', seconds=210, workers=8)
    assert receipt['helper_sha256'] == module.ORIGINAL_HELPER_SHA256
    assert '-lease' not in value.calls[-1]
    value.stop_helper('payment')
    assert 'readlink /proc/$p/exe' in value.calls[-1][-1]
    with pytest.raises(ValueError, match='own'):
        value.stop_helper('payment')


def test_recovery_retry_keeps_failed_incident_outside_accepted_schedule(tmp_path):
    _, module = legacy_driver()
    state = tmp_path / 'runs/adaptive_v2a/capture/schedule.json'
    state.parent.mkdir(parents=True)
    state.write_text('{"accepted":{"session/0":"original"}}')
    marker = tmp_path / 'runs/adaptive_v2/lab/blocked-lab.json'
    marker.parent.mkdir(parents=True)
    calls = []
    def capture(output):
        calls.append(1)
        if len(calls) == 1:
            marker.write_text('{"incident_id":"failed1","recovery_failed":true}')
            output.write_text('{"passed":false,"error":"capture checkout did not recover"}')
            return 1
        output.write_text('{"passed":true}')
        return 0
    def recover():
        marker.unlink()
        (marker.parent / 'manual-recovery.json').write_text('{"passed":true}')
    assert module.capture_with_recovery(tmp_path, capture, recover) == 0
    assert len(calls) == 2
    archive = tmp_path / 'runs/adaptive_v2a/recovery-retries/failed1'
    assert (archive / 'capture-failure.json').exists()
    assert state.read_text() == '{"accepted":{"session/0":"original"}}'


def test_repeated_recovery_failure_is_bounded_and_propagates_exit_code(tmp_path):
    _, module = legacy_driver()
    state = tmp_path / 'runs/adaptive_v2a/capture/schedule.json'
    state.parent.mkdir(parents=True)
    state.write_text('{"accepted":{}}')
    marker = tmp_path / 'runs/adaptive_v2/lab/blocked-lab.json'
    marker.parent.mkdir(parents=True)
    calls = []
    def capture(output):
        calls.append(1)
        marker.write_text('{"incident_id":"failed' + str(len(calls)) + '","recovery_failed":true}')
        output.write_text('{"passed":false,"error":"capture checkout did not recover"}')
        return 1
    def recover():
        marker.unlink()
        (marker.parent / 'manual-recovery.json').write_text('{"passed":true}')
    assert module.capture_with_recovery(tmp_path, capture, recover) == 1
    assert len(calls) == 3 and marker.exists()


@pytest.mark.parametrize('recovery_confirmed,expected_calls', [(True, 2), (False, 1)])
@pytest.mark.parametrize('message', ['CPU capture lacks verified contention and checkout effect', 'healthy capture baseline failed', 'healthy control degraded'])
def test_cpu_effect_retry_requires_rejected_attempt_and_confirmed_cleanup(tmp_path, recovery_confirmed, expected_calls, message):
    import json
    _, module = legacy_driver()
    capture_root = tmp_path / 'runs/adaptive_v2a/capture'
    attempts = capture_root / 'attempts'
    attempts.mkdir(parents=True)
    (capture_root / 'schedule.json').write_text('{"accepted":{"s/0":"original"}}')
    lab = tmp_path / 'runs/adaptive_v2/lab'
    lab.mkdir(parents=True)
    calls = []
    def capture(output):
        calls.append(1)
        if len(calls) == 1:
            receipt = {'incident_id': 'rejected_cpu', 'status': 'REJECTED',
                'recovery_confirmed': recovery_confirmed, 'failure_message': message}
            (attempts / 'rejected_cpu.json').write_text(json.dumps(receipt))
            output.write_text(json.dumps({'passed': False, 'error': message}))
            return 1
        output.write_text('{"passed":true}')
        return 0
    def recover():
        (lab / 'manual-recovery.json').write_text('{"passed":true}')
    assert module.capture_with_recovery(tmp_path, capture, recover) == (0 if recovery_confirmed else 1)
    assert len(calls) == expected_calls
    assert (capture_root / 'schedule.json').read_text() == '{"accepted":{"s/0":"original"}}'


@pytest.mark.parametrize('profile', ['v2b', 'v2c'])
@pytest.mark.parametrize('healthy', [True, False])
def test_leased_profile_retry_requires_verified_recovery_and_preserves_cohort(tmp_path, profile, healthy):
    import json
    _, module = legacy_driver()
    run = tmp_path / 'runs' / ('adaptive_' + profile)
    state = run / 'capture/schedule.json'
    state.parent.mkdir(parents=True)
    original = '{"accepted":{"session/0":"saved"}}'
    state.write_text(original)
    marker = tmp_path / 'runs/adaptive_v2/lab/blocked-lab.json'
    marker.parent.mkdir(parents=True)
    calls = []
    def capture(output):
        calls.append(output)
        if len(calls) == 1:
            marker.write_text('{"incident_id":"rejected","recovery_failed":true}')
            output.write_text('{"passed":false,"error":"capture checkout did not recover"}')
            return 1
        output.write_text('{"passed":true}')
        return 0
    def recover():
        if healthy:
            marker.unlink()
        (marker.parent / 'manual-recovery.json').write_text(json.dumps({'passed': healthy}))
    assert module.capture_with_recovery(tmp_path, capture, recover, profile) == (0 if healthy else 1)
    assert len(calls) == (2 if healthy else 1)
    assert state.read_text() == original
    assert (run / 'recovery-retries/rejected/capture-failure.json').exists()
    assert not (tmp_path / 'runs/adaptive_v2a').exists()
