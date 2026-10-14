from pathlib import Path
from time import sleep
from statistics import median
import json
from rca_investigation.dataset import atomic_json
from rca_investigation.backends.container_stats import ContainerStatsCollector
from rca_investigation.faults_v2 import AdaptiveLabDriver
from rca_investigation.topology import sockshop_topology
from rca_investigation.workload_v2 import CheckoutWorkload
from rca_investigation.capture_v2 import collect_window

root = Path(__file__).resolve().parents[1]
registry = json.loads((root / 'runs/adaptive_v2/capabilities.json').read_text())['resources']
stats = ContainerStatsCollector({'carts': registry['carts']})
driver = AdaptiveLabDriver(sockshop_topology(), root)
driver.resolve()
workload = CheckoutWorkload()
report = {'physical_engineering_only': True, 'passed': False, 'measurement': 'retained_resource_pressure'}
active = False
try:
    report['prepared'] = workload.prepare()
    report['baseline'] = [workload.checkout() for _ in range(12)]
    base = stats.poll()[0]['memory_bytes']
    limit = registry['carts']['memory_limit_bytes']
    target = min(int(limit * .83) - base, int(limit * .5))
    if target <= 0:
        raise RuntimeError('healthy baseline already exceeds proposed memory pressure target')
    report.update(baseline_bytes=base, memory_limit_bytes=limit, allocation_bytes=target)
    # Healthy allocate-and-release control.
    report['control_injection'] = driver.inject_helper('carts', 'memory', seconds=20, allocation_bytes=min(target, 32*1024**2))
    active = True
    sleep(2)
    report['control_requests'] = [workload.checkout() for _ in range(12)]
    driver.stop_helper('carts')
    active = False
    sleep(2)
    report['injection'] = driver.inject_helper('carts', 'memory', seconds=150, allocation_bytes=target)
    active = True
    sleep(6)
    window, raw = collect_window(driver, workload, stats, {'carts': registry['carts']})
    report.update(window=window, raw=raw)
finally:
    if active:
        report['cleanup'] = driver.stop_helper('carts')
        sleep(2)
    try:
        report['recovery_requests'] = [workload.checkout() for _ in range(12)]
        report['recovery_stats'] = stats.poll()
        if report.get('raw'):
            fractions = [row['memory_bytes'] / report['memory_limit_bytes'] for row in report['raw']['stats'] if row['availability'] == 'OK']
            report['pressure_confirmed'] = bool(fractions) and median(fractions) >= .75
            report['recovery_confirmed'] = (all(row['http_status'] == 201 for row in report['recovery_requests']) and
                report['recovery_stats'][0]['memory_bytes'] < report['baseline_bytes'] + report['allocation_bytes'] * .25)
            baseline_ms = median(row['duration_ms'] for row in report['baseline'])
            report['checkout_effect_confirmed'] = (median(row['duration_ms'] for row in report['raw']['requests']) > baseline_ms * 1.5 or
                any(row['http_status'] != 201 for row in report['raw']['requests']))
            report['passed'] = report['pressure_confirmed'] and report['recovery_confirmed']
        workload.reset()
    finally:
        workload.close()
        atomic_json(root / 'runs/adaptive_v2/memory-pilot.json', report)
        print(json.dumps({key: report.get(key) for key in ('passed', 'pressure_confirmed', 'checkout_effect_confirmed', 'recovery_confirmed')}, indent=2))
raise SystemExit(0 if report['passed'] else 1)
