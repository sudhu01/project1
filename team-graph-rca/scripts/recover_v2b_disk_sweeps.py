"""Recover disk-failed offline variants after the original sweep runner exits."""
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time
import ctypes
from ctypes import wintypes
import yaml
from rca_investigation.dataset import atomic_json
from rca_investigation.train import runtime_hash

root = Path.cwd()
parser = argparse.ArgumentParser()
parser.add_argument('--after-pause', action='store_true', help='Resume after verifying all prior controllers have stopped.')
args = parser.parse_args()
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
# Retain the original process handle so PID reuse cannot start recovery early.
process_handle = None if args.after_pause else kernel32.OpenProcess(0x00100000, False, 8572)
if process_handle:
    try:
        while True:
            status = kernel32.WaitForSingleObject(process_handle, 20000)
            if status == 0:
                break
            if status != 258:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(process_handle)
elif not args.after_pause and ctypes.get_last_error() != 87:
    raise ctypes.WinError(ctypes.get_last_error())
parent_path = root / 'runs/adaptive_v2b/phase-progress.json'
parent = json.loads(parent_path.read_text())
assert runtime_hash() == parent['source_hash']
assert json.loads((root / 'runs/adaptive_v2b/capture/schedule.json').read_text())['complete']
assert not (root / 'runs/adaptive_v2/lab/capture.lock').exists()
archive = root / 'runs/adaptive_v2b/disk-sweep-recovery'
archive.mkdir(exist_ok=True)
archive = archive / datetime.now(timezone.utc).strftime('attempt-%Y%m%dT%H%M%S%fZ')
archive.mkdir(exist_ok=False)
shutil.copy2(parent_path, archive / 'parent-phase-progress.json')
results = []
for family in ('ablations', 'costs'):
    manifest = json.loads((root / f'configs/sockshop_adaptive_v2b-{family}-fcf9c90b8f89/manifest.json').read_text())
    for variant in manifest['variants']:
        config = yaml.safe_load((root / variant['config']).read_text())
        run = root / config['paths']['runs']
        phase_path = run / 'phase-progress.json'
        phase = json.loads(phase_path.read_text()) if phase_path.exists() else {}
        if not (phase.get('passed') and phase.get('phase') == 'complete'):
            print('Recovering ' + variant['config'], flush=True)
            if phase_path.exists():
                shutil.copy2(phase_path, archive / (run.name + '-phase-progress.json'))
            subprocess.run([sys.executable, 'scripts/run_adaptive_experiment.py', '--config',
                variant['config'], '--wait-for-capture', '--allow-ambiguous-evidence', '--reserved-test'], check=True)
        saved = json.loads(phase_path.read_text())
        assert saved['passed'] and saved['phase'] == 'complete'
        assert saved['source_hash'] == parent['source_hash']
        results.append({'config': variant['config'], 'passed': True})
assert runtime_hash() == parent['source_hash']
parent.pop('error', None)
parent.update(sweep_results=results, passed=True, phase='complete',
              physical_policy_results=True, training_promotion_eligible=parent['full_evidence_reference_passed'])
atomic_json(parent_path, parent)
print('V2b disk-failed sweeps recovered. Ready for v2c.', flush=True)
