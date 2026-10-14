"""Audit accepted public captures without opening private labels or test responses."""
import json
from pathlib import Path
from collections import Counter
from rca_investigation.train import runtime_hash
from rca_investigation.dataset import atomic_json
from rca_investigation.contracts import digest

root = Path.cwd()
run = root / 'runs/adaptive_v2c'
state = json.loads((run / 'capture/schedule.json').read_text())
assert state['complete'] and len(state['accepted']) == 240
physical, incidents, roles = set(), set(), Counter()
for incident in state['accepted'].values():
    if isinstance(incident, dict):
        incident = incident['incident_id']
    bundle = root / 'datasets/raw_v2c' / incident
    checksums = json.loads((bundle / 'checksums.json').read_text())
    for name, expected in checksums.items():
        if name == 'private_label.json':
            continue
        assert digest(json.loads((bundle / name).read_text())) == expected
    receipt = json.loads((bundle / 'capture_receipt.json').read_text())
    assert receipt['status'] == 'ACCEPTED' and receipt['recovery_confirmed']
    assert receipt['effect_confirmed']
    assert receipt['incident_id'] not in incidents
    assert receipt['physical_episode'] not in physical
    incidents.add(receipt['incident_id'])
    physical.add(receipt['physical_episode'])
    roles[receipt['role']] += 1
assert dict(roles) == {'train': 120, 'validation': 60, 'test': 60}
assert runtime_hash() == '54322988ac5c35261153217f092d07d4a261dfe36d76098870652f824ad9ef46'
assert not (root / 'runs/adaptive_v2/lab/blocked-lab.json').exists()
assert not (root / 'runs/adaptive_v2/lab/capture.lock').exists()
audit = dict(passed=True, accepted=240, unique_physical_episodes=240,
             public_checksums_verified=True, private_labels_opened=False,
             roles=dict(roles), runtime_hash=runtime_hash(), cleanup_confirmed=True)
atomic_json(run / 'capture-completion-verification.json', audit)
print(json.dumps(audit, indent=2))
