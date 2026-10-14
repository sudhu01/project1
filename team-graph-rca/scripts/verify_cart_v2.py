"""Verify actual frontend cart identifiers and separately timed operations."""
from pathlib import Path
from rca_investigation.workload_v2 import CheckoutWorkload
from rca_investigation.dataset import atomic_json

workload = CheckoutWorkload()
try:
    workload.prepare()
    workload.reset()
    for _ in range(5):
        workload.cart_sequence()
    passed = len(workload.cart_observations) == 20 and all(
        row['outcome'] == 'completed' and 200 <= row['http_status'] < 300
        for row in workload.cart_observations)
    report = {'passed': passed, 'observations': workload.cart_observations,
              'operations': sorted({row['operation'] for row in workload.cart_observations})}
    atomic_json(Path('runs/adaptive_v2/cart-verification.json'), report)
    print(report)
    if not passed:
        raise RuntimeError('cart operation verification failed')
finally:
    workload.reset()
    workload.close()
