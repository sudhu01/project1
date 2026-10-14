"""Resume the started v2a cohort with its original pinned CPU helper."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import sleep
from rca_investigation.dataset import atomic_json
from rca_investigation.faults_v2 import AdaptiveLabDriver

ORIGINAL_HELPER_SHA256 = '6c42d03620b1596d1a186ed1678c2b944bd27738839402c4eba4fd84c4661503'


class LegacyV2aDriver(AdaptiveLabDriver):
    def inject_helper(self, resource, mode, seconds=120, workers=4, allocation_bytes=0):
        if (resource, mode, seconds, workers, allocation_bytes) != ('payment', 'cpu', 210, 8, 0):
            raise ValueError('legacy resume permits only the original v2a CPU dispatch')
        if any(type(value) is not int for value in (seconds, workers, allocation_bytes)):
            raise ValueError('integer helper bounds required')
        if self.state(resource)['state'] != 'running':
            raise RuntimeError('helper target not running')
        binary = Path(self.root) / 'lab/adaptive-helper/rca-helper'
        pin = json.loads((Path(self.root) / 'configs/lab_adaptive_v2.json').read_text())
        if pin['helper_sha256'] != ORIGINAL_HELPER_SHA256 or hashlib.sha256(binary.read_bytes()).hexdigest() != ORIGINAL_HELPER_SHA256:
            raise ValueError('original v2a helper hash mismatch')
        self.command(['cp', str(binary), self.ids[resource] + ':/tmp/rca-helper'])
        self.helper_leases[resource] = 'owned-original-v2a-dispatch'
        _, receipt = self.command(['exec', '-d', self.ids[resource], '/tmp/rca-helper',
                                  '-mode', mode, '-seconds', str(seconds), '-workers', str(workers),
                                  '-bytes', str(allocation_bytes)])
        return {**receipt, 'resource': resource, 'mode': mode, 'helper_sha256': ORIGINAL_HELPER_SHA256,
                'container_id': self.ids[resource]}

    def stop_helper(self, resource):
        if resource != 'payment' or resource not in self.helper_leases:
            raise ValueError('legacy cleanup requires its own payment dispatch')
        self.state(resource)
        command = 'if [ -f /tmp/rca-helper.pid ]; then p=$(cat /tmp/rca-helper.pid); case "$p" in *[!0-9]*|"") exit 2;; esac; [ "$(readlink /proc/$p/exe)" = /tmp/rca-helper ] || exit 3; kill -TERM "$p"; for i in 1 2 3 4 5 6 7 8 9 10; do [ -f /tmp/rca-helper.pid ] || exit 0; sleep 0.2; done; exit 5; fi'
        _, receipt = self.command(['exec', self.ids[resource], 'sh', '-c', command])
        self.helper_leases.pop(resource)
        return receipt


def capture_with_recovery(root, run_capture, recover, profile='v2a'):
    """Retry known capture failures after saving their audit and restoring health."""
    if profile not in {'v2a', 'v2b', 'v2c'}:
        raise ValueError('unknown adaptive capture profile')
    profile_root = root / 'runs' / ('adaptive_' + profile)
    output = profile_root / 'capture-evaluation.json'
    attempt_output = profile_root / 'capture-resume-attempt.json'
    state_path = profile_root / 'capture/schedule.json'
    previous_count, failures = -1, 0
    while True:
        result = run_capture(attempt_output)
        report = json.loads(attempt_output.read_text())
        if result == 0:
            atomic_json(output, report)
            return 0
        count = len(json.loads(state_path.read_text())['accepted'])
        failures = failures + 1 if count == previous_count else 1
        previous_count = count
        marker_path = root / 'runs/adaptive_v2/lab/blocked-lab.json'
        if failures >= 3:
            atomic_json(output, report)
            return result
        attempt_receipt = None
        if report.get('error') == 'capture checkout did not recover' and marker_path.exists():
            marker = json.loads(marker_path.read_text())
            eligible = marker.get('recovery_failed') and marker.get('incident_id')
        elif report.get('error') in {'CPU capture lacks verified contention and checkout effect', 'healthy capture baseline failed', 'healthy control degraded', 'retained memory pressure absent'} and not marker_path.exists():
            attempts = list((profile_root / 'capture/attempts').glob('*.json'))
            attempt_receipt = json.loads(max(attempts, key=lambda path: path.stat().st_mtime_ns).read_text()) if attempts else {}
            accepted = json.loads(state_path.read_text())['accepted']
            eligible = (attempt_receipt.get('status') == 'REJECTED' and attempt_receipt.get('recovery_confirmed')
                and attempt_receipt.get('failure_message') == report['error']
                and attempt_receipt.get('incident_id') not in accepted.values())
            if report['error'] in {'healthy capture baseline failed', 'healthy control degraded'}:
                eligible = (eligible and not attempt_receipt.get('injector_receipt')
                            and not attempt_receipt.get('injection') and not attempt_receipt.get('injector')
                            and not attempt_receipt.get('control'))
            if report['error'] == 'retained memory pressure absent':
                injector = attempt_receipt.get('injector', {})
                eligible = (eligible and profile == 'v2c' and injector.get('mode') == 'memory'
                            and injector.get('resource') == 'carts' and bool(injector.get('lease')))
            marker = {'incident_id': attempt_receipt.get('incident_id'), 'effect_rejected': True}
        else:
            eligible, marker = False, {}
        if not eligible or not marker.get('incident_id'):
            atomic_json(output, report)
            return result
        archive = profile_root / 'recovery-retries' / marker['incident_id']
        archive.mkdir(parents=True, exist_ok=False)
        atomic_json(archive / 'capture-failure.json', report)
        atomic_json(archive / 'blocked-lab.json', marker)
        atomic_json(archive / 'schedule.json', json.loads(state_path.read_text()))
        if attempt_receipt is not None:
            atomic_json(archive / 'rejected-attempt.json', attempt_receipt)
        print('Capture failed; preserving the rejected attempt and checking lab health before retry.', flush=True)
        try:
            recover()
            recovery_report = json.loads((root / 'runs/adaptive_v2/lab/manual-recovery.json').read_text())
            if not recovery_report.get('passed') or marker_path.exists():
                raise RuntimeError('manual lab recovery remains unconfirmed')
            atomic_json(archive / 'manual-recovery.json', recovery_report)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            atomic_json(output, {'passed': False, 'error': str(error), 'capture_failure': report})
            return 1


def main():
    from rca_investigation import raw_schedule, cli
    raw_schedule.AdaptiveLabDriver = LegacyV2aDriver
    root = Path.cwd()
    def run_capture(output):
        return cli.main(['capture-raw', '--config', 'configs/investigation_v2a.yaml',
              '--schedule', 'configs/capture_v2a_evaluation.yaml',
              '--output', str(output)])
    def recover():
        sleep(15)
        subprocess.run([sys.executable, 'scripts/recover_adaptive_lab.py'], cwd=root, check=True, timeout=150)
    return capture_with_recovery(root, run_capture, recover)


if __name__ == '__main__':
    raise SystemExit(main())
