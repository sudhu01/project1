"""Finish the failed synthetic report without repeating fitting or reserved tests."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from rca_investigation.dataset import atomic_json
from rca_investigation.train import runtime_hash

root = Path.cwd()
run = root / 'runs/sockshop_adaptive_v2a/costs/ad9a355bea83'
parent_path = root / 'runs/adaptive_v2a/phase-progress.json'
parent = json.loads(parent_path.read_text())
phase_path = run / 'phase-progress.json'
phase = json.loads(phase_path.read_text())
assert parent['source_hash'] == phase['source_hash'] == runtime_hash()
assert phase['error'] == 'challenge failed; its report contains the reason'
assert json.loads((run / 'compare-test.json').read_text())['passed']
assert json.loads((run / 'challenge.json').read_text())['error'] == '[Errno 28] No space left on device'
archive = run / 'challenge-recovery'
archive.mkdir(exist_ok=False)
for path in (run / 'challenge.json', phase_path, parent_path):
    shutil.copy2(path, archive / ('parent-phase-progress.json' if path == parent_path else path.name))
subprocess.run([sys.executable, '-m', 'rca_investigation.cli', 'challenge', '--config',
               'configs/sockshop_adaptive_v2a-costs-6d1f2caff398/ad9a355bea83.yaml'], check=True)
challenge = json.loads((run / 'challenge.json').read_text())
assert challenge['passed'] and challenge['synthetic_perturbations_only']
assert challenge['additional_physical_incidents'] == 0
assert runtime_hash() == parent['source_hash']
phase.pop('error')
phase['completed'].append('challenge')
phase.update(passed=True, phase='complete', physical_policy_results=True,
             training_promotion_eligible=phase['full_evidence_reference_passed'])
atomic_json(phase_path, phase)
for result in parent['sweep_results']:
    import yaml
    config = yaml.safe_load((root / result['config']).read_text())
    saved = json.loads((root / config['paths']['runs'] / 'phase-progress.json').read_text())
    assert saved['passed'] and saved['phase'] == 'complete', result['config']
    result['passed'] = True
parent.pop('error')
parent.update(passed=True, phase='complete', physical_policy_results=True,
              training_promotion_eligible=parent['full_evidence_reference_passed'])
atomic_json(parent_path, parent)
print('V2a declared sweeps and synthetic challenges complete.', flush=True)
