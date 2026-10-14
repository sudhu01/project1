"""Verify and recover existing lab identities before a new physical cohort."""
import hashlib
import json
from pathlib import Path
import subprocess
from rca_investigation.dataset import atomic_json
from rca_investigation.train import runtime_hash

root=Path.cwd()
state=json.loads((root/'runs/adaptive_v2/follow-up-state.json').read_text())
assert state['pre_v2c_report_delivered']
assert not (root/'runs/adaptive_v2/lab/capture.lock').exists()
for profile in ('v2a','v2b'):
    phase=json.loads((root/f'runs/adaptive_{profile}/phase-progress.json').read_text())
    assert phase['passed'] and phase['phase']=='complete' and phase['source_hash']==runtime_hash()
cap=json.loads((root/'runs/adaptive_v2/capabilities.json').read_text())
pin=json.loads((root/'configs/lab_adaptive_v2_lease.json').read_text())
assert hashlib.sha256((root/'lab/adaptive-helper/rca-helper-lease').read_bytes()).hexdigest()==pin['helper_sha256']
def docker(*args):
    return subprocess.run(['docker',*args],check=True,capture_output=True,text=True,timeout=30).stdout
ids=docker('ps','-aq','--filter','label=com.docker.compose.project=rca-lab').split()
assert ids
rows=json.loads(docker('inspect',*ids))
by_service={row['Config']['Labels']['com.docker.compose.service']:row for row in rows}
for resource,saved in cap['resources'].items():
    row=by_service[saved['compose_service']]
    assert row['Id']==saved['container_id'], resource+' identity changed'
    if resource in pin['helper_images']:
        assert row['Image']==pin['helper_images'][resource]
assert by_service['payment-path-proxy']['Id']=='b8952cca89a63efc1768f30c5a37a46a91a80ca1ffffec55646984c52102234e'
assert by_service['payment-path-proxy']['Image']=='sha256:ffccefc9317a154dc4431a25a5f13970d4d55617ae735bb2679e7668904c119e'
stopped=[row['Id'] for row in rows if not row['State']['Running']]
if stopped:
    docker('start',*stopped)
atomic_json(root/'runs/adaptive_v2c/preflight-identities.json',dict(passed=True,containers={name:row['Id'] for name,row in by_service.items()},started_existing=stopped,helper_sha256=pin['helper_sha256'],runtime_hash=runtime_hash()))
print('Verified lab identities and helper pins; started existing containers:',len(stopped),flush=True)
