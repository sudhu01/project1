from pathlib import Path
from time import monotonic, sleep
from statistics import median
import json
from rca_investigation.dataset import atomic_json
from rca_investigation.workload_v2 import CheckoutWorkload
from rca_investigation.faults import LabDriver
from rca_investigation.topology import sockshop_topology
from rca_investigation.backends.docker import compose_rows, resolve_containers

root = Path(__file__).resolve().parents[1]
driver = LabDriver(sockshop_topology(), root)
output, _ = driver.command(['compose', '-p', 'rca-lab', 'ps', '--all', '--no-trunc', '--format', 'json'])
proxy = resolve_containers(compose_rows(output), ['payment-path-proxy'])['payment-path-proxy']
workload = CheckoutWorkload()
report = {'physical_engineering_only': True, 'passed': False, 'cause_id': 'edge/orders/payment', 'severities': {}}


def control(delay=None):
    args = ['exec', proxy, '/rca-path-proxy', '-mode', 'inspect' if delay is None else 'set']
    if delay is not None:
        args += ['-delay-ms', str(delay)]
    payload, receipt = driver.command(args)
    return json.loads(payload)


try:
    deadline = monotonic() + 90
    while True:
        try:
            workload.prepare()
            if workload.checkout()['http_status'] == 201:
                break
        except Exception:
            pass
        if monotonic() >= deadline:
            raise RuntimeError('healthy proxy-routed checkout unavailable')
        sleep(2)
    control(0)
    before = control()['forwarded_requests']
    baseline = [workload.checkout() for _ in range(12)]
    report['baseline'] = baseline
    report['routing_confirmed'] = control()['forwarded_requests'] - before >= 12
    for delay in (100, 300, 800):
        report['severities'][str(delay)] = {'configuration': control(delay)}
        rows = [workload.checkout() for _ in range(12)]
        report['severities'][str(delay)]['requests'] = rows
        report['severities'][str(delay)]['median_ms'] = median(row['duration_ms'] for row in rows)
        atomic_json(root / 'runs/adaptive_v2/latency-pilot-progress.json', report)
finally:
    report['recovery_configuration'] = control(0)
    try:
        recovered = [workload.checkout() for _ in range(12)]
        report['recovery'] = recovered
        driver.resolve()
        report['containers_running'] = all(driver.state(resource)['state'] == 'running' for resource in ('orders', 'payment'))
        if report.get('baseline') and report['severities'].get('800', {}).get('requests'):
            baseline_median = median(row['duration_ms'] for row in report['baseline'])
            recovered_median = median(row['duration_ms'] for row in recovered)
            report['effect_confirmed'] = report['severities']['800']['median_ms'] > baseline_median + 500
            report['recovery_confirmed'] = all(row['http_status'] == 201 for row in recovered) and recovered_median < baseline_median + 100
            report['passed'] = report['routing_confirmed'] and report['effect_confirmed'] and report['recovery_confirmed'] and report['containers_running']
        workload.reset()
    finally:
        workload.close()
        atomic_json(root / 'runs/adaptive_v2/latency-pilot.json', report)
        print(json.dumps({key: report.get(key) for key in ('passed', 'routing_confirmed', 'effect_confirmed', 'recovery_confirmed', 'containers_running')}, indent=2))
raise SystemExit(0 if report['passed'] else 1)
