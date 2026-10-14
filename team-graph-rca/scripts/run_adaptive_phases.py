"""Run the three declared profiles in order, without overlapping lab work."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rca_investigation.capture import capture_lease
from rca_investigation.dataset import atomic_json

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--first-capture-running', action='store_true')
parser.add_argument('--start-profile', choices=('v2a', 'v2b', 'v2c'), default='v2a')
parser.add_argument('--prior-profile-running', choices=('v2a',))
args = parser.parse_args()
if args.prior_profile_running and args.start_profile != 'v2b':
    parser.error('the existing v2a offline sweeps can accompany only a v2b start')
report_path = ROOT / 'runs/adaptive_v2/phase-sequence.json'
report = {'passed': False, 'completed_profiles': [], 'phase': args.start_profile}
if args.start_profile == 'v2c':
    tracking = json.loads((ROOT / 'runs/adaptive_v2/follow-up-state.json').read_text())
    if not tracking.get('pre_v2c_report_delivered'):
        raise RuntimeError('deliver the requested benchmark report before starting v2c')
    for previous in ('v2a', 'v2b'):
        saved = json.loads((ROOT / f'runs/adaptive_{previous}/phase-progress.json').read_text())
        if not (saved.get('passed') and saved.get('phase') == 'complete'):
            raise RuntimeError(previous + ' is unfinished; do not start v2c')
        report['completed_profiles'].append(previous)
if args.prior_profile_running:
    report['prior_profile_running'] = args.prior_profile_running
atomic_json(report_path, report)


def run(*command):
    print('Running ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def wait_for_first_capture():
    state_path = ROOT / 'runs/adaptive_v2a/capture/schedule.json'
    previous = None
    while True:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        count = len(state.get('accepted', {}))
        if count != previous:
            print(f'v2a accepted capture slots: {count}', flush=True)
            report.update(phase='v2a_capture', accepted_capture_slots=count, expected_capture_slots=160)
            atomic_json(report_path, report)
            previous = count
        if state.get('complete'):
            return
        failure = ROOT / 'runs/adaptive_v2a/capture-evaluation.json'
        if failure.exists() and not json.loads(failure.read_text()).get('passed'):
            raise RuntimeError('v2a capture failed; accepted slots remain available for resume')
        time.sleep(10)


def prepare_lease_images():
    # Runs only between completed cohorts. Keep the preceding lab evidence.
    target = ROOT / 'runs/adaptive_v2/pre-lease-images'
    target.mkdir(parents=True, exist_ok=True)
    files = ('capabilities.json', 'checkout-verification.json', 'cart-verification.json',
             'cpu-pilot-tail-confirmation.json', 'latency-pilot.json', 'memory-pilot.json')
    with capture_lease(ROOT / 'runs/adaptive_v2/lab'):
        legacy = ROOT / 'runs/integration_v1/captures/sockshop_v1'
        if any((legacy / name).exists() for name in ('capture.lock', 'blocked-lab.json')):
            raise RuntimeError('v1 capture or unresolved recovery owns the lab')
        for name in files:
            source = ROOT / 'runs/adaptive_v2' / name
            if source.exists() and not (target / name).exists():
                shutil.copy2(source, target / name)
        command = ['docker', 'compose', '-p', 'rca-lab']
        for name in ('docker-compose.yml', 'configs/lab_pinned.yaml',
                     'configs/lab_adaptive_path_v2.yaml', 'configs/lab_adaptive_cpu_period_pilot.yaml',
                     'configs/lab_adaptive_helper_images_v2.yaml'):
            command += ['-f', name]
        run(*command, 'up', '-d', '--no-deps', 'payment', 'carts')
        from rca_investigation.workload_v2 import CheckoutWorkload
        deadline, attempts = time.monotonic() + 90, []
        while True:
            workload = CheckoutWorkload()
            try:
                workload.prepare()
                rows = [workload.checkout() for _ in range(5)]
                if not all(row['http_status'] == 201 and row['outcome'] == 'completed' for row in rows):
                    raise RuntimeError('checkout has not recovered after image replacement')
                workload.reset()
                attempts.append({'passed': True, 'requests': rows})
                break
            except (RuntimeError, ValueError, OSError) as error:
                attempts.append({'passed': False, 'error': str(error)})
                if time.monotonic() >= deadline:
                    raise RuntimeError('helper image startup did not recover within ninety seconds') from error
                time.sleep(2)
            finally:
                workload.close()
                atomic_json(ROOT / 'runs/adaptive_v2/lease-image-startup.json', {'attempts': attempts})
        for script, options in (('verify_checkout_v2.py', []), ('verify_cart_v2.py', []),
                                ('inventory_adaptive_v2.py', []),
                                ('pilot_cpu_v2.py', ['--confirm-tail']),
                                ('pilot_latency_v2.py', []), ('pilot_memory_v2.py', [])):
            run(sys.executable, 'scripts/' + script, *options)
    atomic_json(ROOT / 'runs/adaptive_v2/lease-images-ready.json', {'passed': True})


try:
    if sys.platform == 'win32':
        import ctypes
        # Thread-scoped request; Windows clears it if this process exits.
        # Leave display power behavior alone while timed captures run.
        report['automatic_sleep_inhibited'] = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    if args.first_capture_running:
        wait_for_first_capture()
    profiles = ('v2a', 'v2b', 'v2c')
    for profile in profiles[profiles.index(args.start_profile):]:
        report['phase'] = profile
        atomic_json(report_path, report)
        if profile == 'v2b' and not (ROOT / 'runs/adaptive_v2/lease-images-ready.json').exists():
            prepare_lease_images()
        run(sys.executable, 'scripts/run_adaptive_experiment.py', '--config',
            f'configs/investigation_{profile}.yaml', '--schedule',
            f'configs/capture_{profile}_evaluation.yaml', '--reserved-test', '--sweeps')
        report['completed_profiles'].append(profile)
        atomic_json(report_path, report)
    if args.prior_profile_running:
        prior_path = ROOT / 'runs/adaptive_v2a/phase-progress.json'
        while True:
            prior = json.loads(prior_path.read_text())
            if prior.get('passed') and prior.get('phase') == 'complete':
                report['completed_profiles'].insert(0, 'v2a')
                break
            if prior.get('error'):
                raise RuntimeError('the existing v2a offline sweeps failed; preserve their reports for recovery')
            report['phase'] = 'waiting_for_v2a_offline_sweeps'
            atomic_json(report_path, report)
            time.sleep(10)
    report.update(passed=True, phase='complete')
except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
    report['error'] = str(error)
finally:
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    atomic_json(report_path, report)
print(json.dumps(report, indent=2), flush=True)
raise SystemExit(0 if report['passed'] else 1)
