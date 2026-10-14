"""Adopt the completed training capture left by the October 7 sharing lock."""
from pathlib import Path
import json
import shutil
import yaml
from rca_investigation.capture_store import read_raw
from rca_investigation.contracts import digest
from rca_investigation.dataset import atomic_json


def main():
    root = Path.cwd()
    incident = 'ba33e0a8-3d66-4d8f-8ae3-44d04523ec6b'
    key = 'v2a-train-19/1'
    expected_hash = 'ea98d267fe59d78a9b8cfd4a793427a83aecbc3ad6f918ea144c2597f7c12676'
    state_path = root / 'runs/adaptive_v2a/capture/schedule.json'
    state = json.loads(state_path.read_text())
    schedule = yaml.safe_load((root / 'configs/capture_v2a_evaluation.yaml').read_text())
    assert state['schedule_hash'] == digest(schedule)
    assert state['roles'] == {session['id']: session['role'] for session in schedule['sessions']}
    session = next(session for session in schedule['sessions'] if session['id'] == 'v2a-train-19')
    assert session['role'] == 'train' and session['scenarios'][1] == 'healthy'
    payloads, raw_hash = read_raw(root / 'datasets/raw_v2a' / incident, public_only=False)
    receipt, private = payloads['capture_receipt.json'], payloads['private_label.json']
    assert raw_hash == expected_hash
    assert receipt['incident_id'] == private['incident_id'] == incident
    assert receipt['schedule_hash'] == state['schedule_hash']
    assert receipt['role'] == 'train' and receipt['capture_session'] == session['id']
    assert receipt['status'] == 'ACCEPTED' and receipt['effect_confirmed'] and receipt['recovery_confirmed']
    assert private['cause_id'] == 'NO_FAULT' and private['mechanism'] == 'healthy'
    assert set(receipt['physical_states'].values()) == {'running'}
    assert state['accepted'].get(key) in {None, incident}
    # This recovery opens accepted training receipts only, never reserved data.
    for existing_key, existing_incident in state['accepted'].items():
        if existing_key == key:
            continue
        assert existing_incident != incident
        if state['roles'][existing_key.rsplit('/', 1)[0]] == 'train':
            existing, _ = read_raw(root / 'datasets/raw_v2a' / existing_incident, public_only=False)
            assert existing['capture_receipt.json']['physical_episode'] != receipt['physical_episode']
    archive = root / 'runs/adaptive_v2a/publication-recovery'
    archive.mkdir(parents=True, exist_ok=True)
    for source in (state_path, root / 'runs/adaptive_v2a/capture-evaluation.json', root / 'runs/adaptive_v2/phase-sequence.json'):
        target = archive / source.name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    state['accepted'][key] = incident
    atomic_json(state_path, state)
    atomic_json(archive / 'recovered.json', {'passed': True, 'slot': key, 'incident_id': incident,
        'raw_hash': raw_hash, 'accepted_slots': len(state['accepted']), 'recovery_confirmed': True,
        'reason': 'Windows sharing lock after completed capture; rename retried without changing bytes'})
    failure = root / 'runs/adaptive_v2a/capture-evaluation.json'
    if failure.exists():
        assert not json.loads(failure.read_text()).get('passed')
        assert (archive / failure.name).read_bytes() == failure.read_bytes()
        failure.unlink()
    print(json.dumps({'recovered': key, 'accepted': len(state['accepted']), 'raw_hash': raw_hash}))


if __name__ == '__main__':
    main()
