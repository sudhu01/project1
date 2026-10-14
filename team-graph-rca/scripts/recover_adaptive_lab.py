"""Recover an owned lab fault after a dead controller, preserving its audit."""
from pathlib import Path
import json
import subprocess
from time import monotonic, sleep
from rca_investigation.faults_v2 import AdaptiveLabDriver
from rca_investigation.topology import sockshop_topology
from rca_investigation.workload_v2 import CheckoutWorkload
from rca_investigation.dataset import atomic_json

root = Path.cwd()
lease_root = root / 'runs/adaptive_v2/lab'
lock = lease_root / 'capture.lock'
if lock.exists():
    owner = json.loads(lock.read_text())['pid']
    if type(owner) is not int or owner <= 0:
        raise RuntimeError('invalid recorded capture owner')
    # Read process existence. Do not use os.kill(pid, 0) on Windows.
    result = subprocess.run(['powershell', '-NoProfile', '-Command',
        f'if (Get-Process -Id {owner} -ErrorAction SilentlyContinue) {{ exit 1 }} else {{ exit 0 }}'],
        timeout=10, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError('a recorded capture owner is still alive; refuse concurrent recovery')
marker_path = lease_root / 'blocked-lab.json'
marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
driver = AdaptiveLabDriver(sockshop_topology(), root)
driver.resolve()
receipt = {'passed': False, 'original_block': marker, 'commands': []}
resource = marker.get('resource')
cause = marker.get('cause', '')
if resource is None and cause.startswith('resource/'):
    resource = cause.removeprefix('resource/')
if resource:
    if resource not in {'catalogue-db', 'orders', 'payment', 'carts'}:
        raise RuntimeError('recovery target is outside registered fault ownership')
    expected = marker.get('container_id') or json.loads((root / 'runs/adaptive_v2/capabilities.json').read_text())['resources'][resource]['container_id']
    if driver.ids[resource] != expected:
        raise RuntimeError('recorded recovery container was replaced; refuse to touch another identity')
    if resource in {'catalogue-db', 'orders'}:
        receipt['commands'].append(driver.recover(resource))
    else:
        injected = marker.get('injector_receipt', {})
        if injected.get('lease') and driver.state(resource)['state'] == 'running':
            driver.helper_leases[resource] = injected['lease']
            receipt['commands'].append(driver.stop_helper(resource))
        else:
            # A registered helper target may be restarted when its dispatch
            # receipt was interrupted or the helper killed the application.
            _, restarted = driver.command(['restart', '--time', '3', driver.ids[resource]])
            receipt['commands'].append(restarted)
if cause.startswith('edge/'):
    receipt['commands'].append(driver.proxy_control(0))
workload = CheckoutWorkload()
try:
    deadline, successful = monotonic() + 90, 0
    while monotonic() < deadline and successful < 5:
        try:
            if not workload.authenticated:
                workload.prepare()
            row = workload.checkout()
            successful = successful + 1 if row['http_status'] == 201 and row['outcome'] == 'completed' else 0
        except Exception:
            successful = 0
        if successful < 5:
            sleep(2)
    receipt['passed'] = successful == 5 and all(state == 'running' for state in driver.states().values())
    if not receipt['passed']:
        raise RuntimeError('healthy operations are still unconfirmed; retain lab block')
    workload.reset()
    marker_path.unlink(missing_ok=True)
    lock.unlink(missing_ok=True)
finally:
    workload.close()
    atomic_json(lease_root / 'manual-recovery.json', receipt)
print(json.dumps(receipt, indent=2))
