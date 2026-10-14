"""Train both learning methods on the new controlled decision problems."""
from pathlib import Path
from rca_investigation.train import train_fixture
from rca_investigation.dataset import atomic_json

reports = []
for fixture in ('three-probe', 'contextual'):
    for method in ('expected-sarsa', 'two-step-sarsa'):
        for seed in (11, 22, 33, 44, 55):
            target = Path('runs/adaptive_v2/fixtures-current') / fixture / method / str(seed)
            report = train_fixture(fixture, target, seed=seed, episodes=10000, method=method,
                                   resume=(target / 'latest.json').exists())
            reports.append({'fixture': fixture, **report})
            print(f"{fixture} {method} {seed}: {report['selection']}", flush=True)
            atomic_json('runs/adaptive_v2/fixture-training.json', {'passed': False,
                'synthetic_only': True, 'reports': reports})
passed = all(row['complete'] and row['selection']['value'][0] >= (.85 - 1e-9 if row['fixture'] == 'three-probe' else .8)
             for row in reports)
atomic_json('runs/adaptive_v2/fixture-training.json', {'passed': passed, 'synthetic_only': True, 'reports': reports})
if not passed:
    raise RuntimeError('controlled fixture learning has not passed')
