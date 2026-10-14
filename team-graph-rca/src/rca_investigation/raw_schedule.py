"""Session roles frozen before collection; accepted raw incidents resume by slot."""
from pathlib import Path
from uuid import uuid4
import json
import yaml
from statistics import median
from .contracts import canonical, digest
from .contracts_v2 import IncidentPrivateV2, PROFILES
from .dataset import atomic_json
from .capture_store import write_raw, read_raw
from .capture import capture_lease, RecoveryFailed
from .capture_v2 import collect_window
from .backends.container_stats import ContainerStatsCollector
from .faults_v2 import AdaptiveLabDriver
from .topology import sockshop_topology
from .workload_v2 import CheckoutWorkload


def settle_states(driver, expected, seconds=30):
    from time import monotonic, sleep
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        if driver.states() != expected:
            raise RuntimeError('resource state changed during settling')
        sleep(min(2, max(0, deadline - monotonic())))


def capture_raw(root, config_path, schedule_path):
    root = Path(root)
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    schedule = yaml.safe_load(Path(schedule_path).read_text(encoding='utf-8'))
    task = config['task_id']
    if task not in PROFILES or schedule['task_id'] != task:
        raise ValueError('capture profile mismatch')
    schedule_hash = digest(schedule)
    roles = {session['id']: session['role'] for session in schedule['sessions']}
    if len(roles) != len(schedule['sessions']) or not set(roles.values()) <= {'train', 'validation', 'test', 'pilot'}:
        raise ValueError('unique named sessions with declared roles required')
    if 'pilot' in roles.values() and set(roles.values()) != {'pilot'}:
        raise ValueError('pilot and evaluation sessions must have separate datasets')
    pilot = set(roles.values()) == {'pilot'}
    raw_root = root / (config['paths']['raw_dataset'] + ('_pilot' if pilot else ''))
    audit = root / config['paths']['runs'] / ('capture_pilot' if pilot else 'capture')
    state_path = audit / 'schedule.json'
    slots = [(session['id'], index, scenario) for session in schedule['sessions']
             for index, scenario in enumerate(session['scenarios'])]
    allowed = {'healthy', 'healthy_higher_load', 'catalogue_db_stop', 'orders_stop', 'payment_cpu'}
    if task in {'sockshop_adaptive_v2b', 'sockshop_adaptive_v2c'}:
        allowed.add('payment_path_delay')
    if task == 'sockshop_adaptive_v2c':
        allowed.update({'carts_memory', 'healthy_allocation_release'})
    if not slots or any(scenario not in allowed for _, _, scenario in slots):
        raise ValueError('unregistered capture mechanism')
    registry_payload = json.loads((root / 'runs/adaptive_v2/capabilities.json').read_text())
    if not registry_payload['passed']:
        raise ValueError('live capability inventory has not passed')
    registry = {resource: registry_payload['resources'][resource] for resource in config['resources']}
    for family, proof in [('payment_cpu', 'cpu-pilot-tail-confirmation'), ('payment_path_delay', 'latency-pilot'), ('carts_memory', 'memory-pilot')]:
        if any(scenario == family for _, _, scenario in slots):
            artifact = json.loads((root / f'runs/adaptive_v2/{proof}.json').read_text())
            if not artifact['passed']:
                raise ValueError(f'physical promotion blocked by {proof}')
    state = {'schedule_hash': schedule_hash, 'roles': roles, 'accepted': {}, 'complete': False}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['schedule_hash'] != schedule_hash or state['roles'] != roles:
            raise ValueError('capture schedule changed after collection began')
    atomic_json(state_path, state)
    driver = AdaptiveLabDriver(sockshop_topology(), root)
    driver.resolve()
    if any(driver.ids[resource] != item['container_id'] for resource, item in registry.items()):
        raise ValueError('capability registry container identities changed')
    stats = ContainerStatsCollector(registry)
    lease_root = root / 'runs/adaptive_v2/lab'
    for session, slot, scenario in slots:
        key = f'{session}/{slot}'
        if key in state['accepted']:
            read_raw(raw_root / state['accepted'][key], public_only=False)
            continue
        workload = CheckoutWorkload()
        workload.measure_cart = task == 'sockshop_adaptive_v2c'
        incident = str(uuid4())
        receipt = {'incident_id': incident, 'physical_episode': str(uuid4()), 'capture_session': session,
                   'lab_batch': schedule['lab_batch'], 'role': roles[session], 'status': 'CREATED',
                   'recovery_confirmed': False, 'schedule_hash': schedule_hash, 'registry_hash': registry_payload['hash']}
        mechanism, cause, affected = 'healthy', 'NO_FAULT', ()
        recovery = None
        observation = None
        with capture_lease(lease_root):
            # Respect the existing v1 controller's lock and recovery gate as well.
            legacy = root / 'runs/integration_v1/captures/sockshop_v1'
            if any((legacy / filename).exists() for filename in ('capture.lock', 'blocked-lab.json')):
                raise RecoveryFailed('v1 capture or unresolved recovery owns the lab')
            try:
                if any(value != 'running' for value in driver.states().values()):
                    raise RuntimeError('capture begins with a stopped lab resource')
                workload.prepare()
                baseline_window, baseline_raw = collect_window(driver, workload, stats, registry, ingestion_seconds=0)
                baseline = [row for row in baseline_raw['requests'] if row.get('operation', 'checkout') == 'checkout']
                receipt['baseline'] = baseline
                receipt['baseline_window'] = baseline_window
                receipt['protocol'] = {'baseline_seconds': 120, 'settling_seconds': 30, 'observation_seconds': 120, 'ingestion_seconds': 10, 'concurrency': 1}
                if not all(row['http_status'] == 201 and row['outcome'] == 'completed' for row in baseline):
                    raise RuntimeError('healthy capture baseline failed')
                if len(baseline) < 20:
                    raise RuntimeError('healthy baseline lacks twenty checkout responses')
                atomic_json(audit / 'attempts' / (incident + '.json'), receipt)
                if scenario.endswith('_stop'):
                    resource = 'catalogue-db' if scenario == 'catalogue_db_stop' else 'orders'
                    cause, mechanism, affected = 'resource/' + resource, 'container_stopped', (resource,)
                    recovery = lambda: driver.recover(resource)
                    atomic_json(lease_root / 'blocked-lab.json', {'incident_id': incident, 'cause': cause, 'resource': resource, 'container_id': driver.ids[resource], 'recovery_required': True})
                    receipt['injector'] = driver.inject_stop(resource)
                elif scenario == 'payment_cpu':
                    cause, mechanism, affected = 'resource/payment', 'cpu_contention', ('payment',)
                    recovery = lambda: driver.stop_helper('payment')
                    atomic_json(lease_root / 'blocked-lab.json', {'incident_id': incident, 'cause': cause, 'resource': 'payment', 'container_id': driver.ids['payment'], 'recovery_required': True})
                    receipt['injector'] = driver.inject_helper('payment', 'cpu', seconds=210, workers=8)
                elif scenario == 'payment_path_delay':
                    cause, mechanism, affected = 'edge/orders/payment', 'dependency_latency', ('orders',)
                    recovery = lambda: driver.proxy_control(0)
                    atomic_json(lease_root / 'blocked-lab.json', {'incident_id': incident, 'cause': cause, 'recovery_required': True})
                    receipt['proxy_before'] = driver.proxy_control()
                    receipt['injector'] = driver.proxy_control(800)
                elif scenario == 'carts_memory':
                    cause, mechanism, affected = 'resource/carts', 'memory_pressure', ('carts',)
                    sample = next(row for row in stats.poll() if row['resource_id'] == 'carts')
                    if sample['availability'] != 'OK':
                        raise RuntimeError('memory pressure requires a measured healthy baseline')
                    receipt['memory_baseline_bytes'] = sample['memory_bytes']
                    limit = registry['carts']['memory_limit_bytes']
                    target = min(int(limit * .83) - sample['memory_bytes'], int(limit * .5))
                    if target <= 0:
                        raise RuntimeError('healthy memory baseline leaves no pressure allocation margin')
                    recovery = lambda: driver.stop_helper('carts')
                    atomic_json(lease_root / 'blocked-lab.json', {'incident_id': incident, 'cause': cause, 'resource': 'carts', 'container_id': driver.ids['carts'], 'recovery_required': True})
                    receipt['injector'] = driver.inject_helper('carts', 'memory', seconds=210, allocation_bytes=target)
                elif scenario == 'healthy_allocation_release':
                    before = next(row for row in stats.poll() if row['resource_id'] == 'carts')
                    if before['availability'] != 'OK':
                        raise RuntimeError('allocation control requires a measured healthy baseline')
                    receipt['control_baseline'] = before
                    recovery = lambda: driver.stop_helper('carts')
                    atomic_json(lease_root / 'blocked-lab.json', {'incident_id': incident, 'resource': 'carts', 'container_id': driver.ids['carts'], 'recovery_required': True})
                    receipt['control'] = driver.inject_helper('carts', 'memory', seconds=210, allocation_bytes=32 * 1024**2)
                    marker = json.loads((lease_root / 'blocked-lab.json').read_text())
                    marker['injector_receipt'] = receipt['control']
                    atomic_json(lease_root / 'blocked-lab.json', marker)
                    atomic_json(audit / 'attempts' / (incident + '.json'), receipt)
                    from time import sleep, monotonic
                    sleep(5)
                    retained = next(row for row in stats.poll() if row['resource_id'] == 'carts')
                    receipt['control_retained'] = retained
                    if retained['availability'] != 'OK' or retained['memory_bytes'] < before['memory_bytes'] + 16 * 1024**2:
                        raise RuntimeError('allocation control lacks a measured retained increase')
                    receipt['control_release'] = recovery()
                    recovery = None
                    deadline = monotonic() + 30
                    while True:
                        released = next(row for row in stats.poll() if row['resource_id'] == 'carts')
                        if released['availability'] == 'OK' and released['memory_bytes'] <= before['memory_bytes'] + max(16 * 1024**2, registry['carts']['memory_limit_bytes'] * .05):
                            receipt['control_released'] = released
                            break
                        if monotonic() >= deadline:
                            raise RecoveryFailed('allocation control did not release within the recovery bound')
                        sleep(2)
                    (lease_root / 'blocked-lab.json').unlink(missing_ok=True)
                expected_states = {resource: 'running' for resource in driver.topology.resources}
                if mechanism == 'container_stopped':
                    expected_states[affected[0]] = 'exited'
                if recovery:
                    marker = json.loads((lease_root / 'blocked-lab.json').read_text())
                    marker['injector_receipt'] = receipt.get('injector', receipt.get('control', {}))
                    atomic_json(lease_root / 'blocked-lab.json', marker)
                atomic_json(audit / 'attempts' / (incident + '.json'), receipt)
                settle_states(driver, expected_states)
                window, raw = collect_window(driver, workload, stats, registry, interval=.1 if scenario == 'healthy_higher_load' else 2,
                                             expected_states=expected_states)
                observation = {'window': window, 'measurements': raw}
                checkout = [row for row in raw['requests'] if row.get('operation', 'checkout') == 'checkout']
                receipt['physical_states'] = driver.states()
                if mechanism == 'cpu_contention':
                    from .metrics.cpu import summarize as cpu_summary
                    samples = [row for row in raw['stats'] if row['resource_id'] == 'payment']
                    summary = cpu_summary(samples, 'payment', registry['payment']['container_id'],
                        registry['payment']['cpu_quota_cores'], window['window_start'], window['window_end'], .8)
                    measured = summary[0] == 'OK' and summary[2]['median'] >= .8
                    from .metrics.common import percentile
                    effect = checkout and (percentile([row['duration_ms'] for row in checkout]) > max(
                                percentile([row['duration_ms'] for row in baseline]) * 1.5, percentile([row['duration_ms'] for row in baseline]) + 100)
                              or any(row['http_status'] != 201 for row in checkout))
                    if not measured or not effect:
                        raise RuntimeError('CPU capture lacks verified contention and checkout effect')
                elif mechanism == 'container_stopped':
                    if driver.state(affected[0])['state'] not in {'exited', 'dead'}:
                        raise RuntimeError('stop effect absent')
                elif mechanism == 'dependency_latency':
                    proxy = driver.proxy_control()
                    if proxy['delay_ms'] != 800 or proxy['forwarded_requests'] - receipt['proxy_before']['forwarded_requests'] < 20 or not checkout or median(row['duration_ms'] for row in checkout) < median(row['duration_ms'] for row in baseline) + 500:
                        raise RuntimeError('path delay or actual proxy traffic not confirmed')
                elif mechanism == 'memory_pressure':
                    from .metrics.memory import summarize as memory_summary
                    samples = [row for row in raw['stats'] if row['resource_id'] == 'carts']
                    summary = memory_summary(samples, 'carts', registry['carts']['container_id'],
                        registry['carts']['memory_limit_bytes'], window['window_start'], window['window_end'], pressure=.75)
                    if summary[0] != 'OK' or summary[2]['limit_fraction'] < .75:
                        raise RuntimeError('retained memory pressure absent')
                    receipt['scope'] = 'resource_pressure_only'
                elif not checkout or any(row['http_status'] != 201 or row['outcome'] != 'completed' for row in checkout):
                    raise RuntimeError('healthy control degraded')
                receipt['effect_confirmed'] = True
            except BaseException as exc:
                receipt['status'] = 'REJECTED'
                receipt['failure'] = type(exc).__name__
                receipt['failure_message'] = str(exc)[:1024]
                raise
            finally:
                try:
                    if recovery:
                        receipt['recovery'] = recovery()
                    from time import sleep, monotonic
                    if mechanism == 'memory_pressure':
                        deadline = monotonic() + 30
                        dropped = False
                        while monotonic() < deadline:
                            sample = next(row for row in stats.poll() if row['resource_id'] == 'carts')
                            if sample['availability'] == 'OK' and sample['memory_bytes'] <= receipt['memory_baseline_bytes'] + max(16 * 1024**2, registry['carts']['memory_limit_bytes'] * .05):
                                receipt['memory_recovery_bytes'] = sample['memory_bytes']
                                dropped = True
                                break
                            sleep(2)
                        if not dropped:
                            raise RecoveryFailed('retained memory did not release within the recovery bound')
                    deadline = monotonic() + 90
                    checks = []
                    while monotonic() < deadline and len(checks) < 5:
                        try:
                            row = workload.checkout()
                            checks = checks + [row] if row['http_status'] == 201 and row['outcome'] == 'completed' else []
                        except Exception:
                            checks = []
                        if len(checks) < 5:
                            sleep(2)
                    receipt['recovery_confirmed'] = len(checks) == 5 and all(value == 'running' for value in driver.states().values())
                    if not receipt['recovery_confirmed']:
                        raise RecoveryFailed('capture checkout did not recover')
                    (lease_root / 'blocked-lab.json').unlink(missing_ok=True)
                    workload.reset()
                except BaseException as exc:
                    receipt['status'] = 'RECOVERY_FAILED'
                    receipt['recovery_failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1024]}
                    marker = json.loads((lease_root / 'blocked-lab.json').read_text()) if (lease_root / 'blocked-lab.json').exists() else {'incident_id': incident}
                    marker.update(recovery_required=True, cause=cause, recovery_failed=True)
                    atomic_json(lease_root / 'blocked-lab.json', marker)
                    raise
                finally:
                    workload.close()
                    if observation is not None and receipt['status'] in {'REJECTED', 'RECOVERY_FAILED'}:
                        # Audit failed measurements separately. They never enter
                        # the accepted raw dataset or any fitting split.
                        audit_value = observation
                        byte_count = len(canonical(observation).encode())
                        if byte_count > 8 * 1024 * 1024:
                            audit_value = {'window': window, 'omitted_oversized_measurements': True,
                                           'observation_bytes': byte_count, 'observation_hash': digest(observation)}
                        receipt['observation_audit_hash'] = digest(audit_value)
                        atomic_json(audit / 'rejected-observations' / (incident + '.json'), audit_value)
                    atomic_json(audit / 'attempts' / (incident + '.json'), receipt)
            receipt['status'] = 'ACCEPTED'
            atomic_json(audit / 'attempts' / (incident + '.json'), receipt)
            public = {'incident_id': incident, 'task_id': task, 'topology_hash': driver.topology.hash,
                      'registry_hash': digest(registry), **window}
            private = IncidentPrivateV2(2, incident, cause, 'healthy' if cause == 'NO_FAULT' else 'edge' if cause.startswith('edge/') else 'resource', mechanism,
                affected, canonical(receipt.get('injector', {})), canonical({'effect_confirmed': True}), canonical({'confirmed': True}))
            write_raw(raw_root, incident, public, raw, private, receipt)
            state['accepted'][key] = incident
            atomic_json(state_path, state)
    state['complete'] = True
    atomic_json(state_path, state)
    return state
