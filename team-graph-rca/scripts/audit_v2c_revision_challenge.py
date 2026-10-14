"""Audit the completed main synthetic challenge against frozen definitions."""
import json
from pathlib import Path
from rca_investigation.dataset import atomic_json
from rca_investigation.train import runtime_hash

run = Path('runs/adaptive_v2c_memory_separation_r1')
c = json.loads((run / 'challenge.json').read_text())
f = json.loads((run / 'final.json').read_text())
assert c['passed'] and c['synthetic_perturbations_only'] and c['additional_physical_incidents'] == 0
assert c['definitions'] == f['definitions'] and f['runtime_hash'] == runtime_hash()
assert len(c['reports']) == 6
results = {}
for scenario, methods in c['reports'].items():
    assert set(methods) == set(f['definitions'])
    for row in methods.values():
        assert row['summary']['source_snapshots'] == 60 and len(row['outcomes']) == 60
    selected = methods[f['selected']]['summary']
    results[scenario] = {k: selected[k] for k in ('accuracy','mean_return','mean_probes')}
audit = dict(passed=True, scenarios_verified=6, frozen_definitions_match=True,
             additional_physical_incidents=0, selected=f['selected'], results=results)
atomic_json(run / 'main-challenge-verification.json', audit)
print(json.dumps(audit, indent=2))
