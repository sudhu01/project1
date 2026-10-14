"""Engineering proof of frontend checkout and payment request counters."""
from pathlib import Path
from datetime import datetime, timezone
import json
import requests
from rca_investigation.workload_v2 import CheckoutWorkload
from rca_investigation.dataset import atomic_json

root = Path(__file__).resolve().parents[1]
report = {'passed': False, 'physical_engineering_only': True, 'timestamp': datetime.now(timezone.utc).isoformat()}
workload = CheckoutWorkload()
try:
    report['prepared'] = workload.prepare()
    report['requests'] = [workload.checkout() for _ in range(12)]
    report['passed'] = all(row['http_status'] == 201 for row in report['requests'])
    # Save observed series identities. Prometheus scrapes may lag this workload.
    payload = requests.get('http://localhost:9090/api/v1/series', params={
        'match[]': 'request_duration_seconds_count{job="sockshop",instance="payment:80"}'}, timeout=5).json()
    report['payment_series'] = payload.get('data', [])
except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
finally:
    try:
        workload.reset()
    except Exception as exc:
        report['cleanup_error'] = type(exc).__name__
        report['passed'] = False
    workload.close()
atomic_json(root / 'runs/adaptive_v2/checkout-verification.json', report)
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['passed'] else 1)
