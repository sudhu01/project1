"""Paired engineering windows. Never enters the evaluation dataset."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
import json
import argparse
from statistics import median
from rca_investigation.metrics.common import percentile
from rca_investigation.backends.container_stats import ContainerStatsCollector
from rca_investigation.dataset import atomic_json
from rca_investigation.faults_v2 import AdaptiveLabDriver
from rca_investigation.topology import sockshop_topology
from rca_investigation.workload_v2 import CheckoutWorkload

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--confirm-tail', action='store_true')
args = parser.parse_args()
stem = 'cpu-pilot-tail-confirmation' if args.confirm_tail else 'cpu-pilot'
registry = json.loads((root / 'runs/adaptive_v2/capabilities.json').read_text())['resources']
collector = ContainerStatsCollector({'payment': registry['payment']})
driver = AdaptiveLabDriver(sockshop_topology(), root)
driver.resolve()
workload = CheckoutWorkload()
report = {'physical_engineering_only': True, 'passed': False, 'windows': {}}
report['effect_criterion'] = 'p95 exceeds baseline by both 50 percent and 100 ms' if args.confirm_tail else 'median exceeds baseline by 50 percent, or checkout fails'
active = False


def window(name, duration=120, interval=2):
    stats, requests = [], []
    deadline = monotonic() + duration
    started_at = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=1) as executor:
        def traffic():
            while monotonic() < deadline:
                row = workload.checkout()
                if monotonic() <= deadline:
                    requests.append(row)
                sleep(interval)
        future = executor.submit(traffic)
        while monotonic() < deadline:
            before = monotonic()
            stats.extend(collector.poll())
            sleep(max(0, 2 - (monotonic() - before)))
        future.result()
    return {'window_start': started_at, 'window_end': datetime.now(timezone.utc).isoformat(),
            'stats': stats, 'requests': requests,
            'cpu_median': median(row['cpu_percent'] / (100 * row['quota_cores']) for row in stats if row['availability'] == 'OK'),
            'checkout_median_ms': median(row['duration_ms'] for row in requests),
            'checkout_p95_ms': percentile([row['duration_ms'] for row in requests]),
            'checkout_success_fraction': sum(row['http_status'] == 201 for row in requests) / len(requests)}


try:
    report['prepared'] = workload.prepare()
    report['windows']['normal'] = window('normal')
    atomic_json(root / f'runs/adaptive_v2/{stem}-progress.json', report)
    report['windows']['healthy_higher_load'] = window('healthy_higher_load', interval=.1)
    atomic_json(root / f'runs/adaptive_v2/{stem}-progress.json', report)
    report['injection'] = driver.inject_helper('payment', 'cpu', seconds=150, workers=8)
    active = True
    sleep(2)
    report['windows']['contention'] = window('contention')
finally:
    if active:
        report['recovery_command'] = driver.stop_helper('payment')
        sleep(2)
    try:
        report['recovery_requests'] = [workload.checkout() for _ in range(5)]
        report['recovery_cpu'] = collector.poll()
        report['recovery_confirmed'] = all(row['http_status'] == 201 for row in report['recovery_requests'])
        if 'contention' in report['windows']:
            normal, fault = report['windows']['normal'], report['windows']['contention']
            report['signal_confirmed'] = fault['cpu_median'] >= .8 and fault['cpu_median'] > normal['cpu_median'] + .3
            report['checkout_effect_confirmed'] = (fault['checkout_median_ms'] > normal['checkout_median_ms'] * 1.5 or
                                                   fault['checkout_success_fraction'] < normal['checkout_success_fraction'])
            if args.confirm_tail:
                report['checkout_effect_confirmed'] = (fault['checkout_p95_ms'] > normal['checkout_p95_ms'] * 1.5
                                                        and fault['checkout_p95_ms'] > normal['checkout_p95_ms'] + 100)
            report['passed'] = report['signal_confirmed'] and report['checkout_effect_confirmed'] and report['recovery_confirmed']
        workload.reset()
    finally:
        workload.close()
        atomic_json(root / f'runs/adaptive_v2/{stem}.json', report)
        print(json.dumps({key: value for key, value in report.items() if key != 'windows'}, indent=2))
raise SystemExit(0 if report['passed'] else 1)
