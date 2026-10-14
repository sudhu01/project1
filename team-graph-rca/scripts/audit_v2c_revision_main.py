"""Verify the completed frozen comparison and summarize main methods."""
import json
from pathlib import Path
from rca_investigation.contracts import digest
from rca_investigation.dataset import atomic_json
from rca_investigation.train import runtime_hash

run = Path('runs/adaptive_v2c_memory_separation_r1')
read = lambda name: json.loads((run / name).read_text())
frozen, validation, test = [read(n) for n in ('final.json', 'compare-validation.json', 'compare-test.json')]
copy = dict(frozen)
claimed = copy.pop('hash')
assert digest(copy) == claimed
assert frozen['runtime_hash'] == runtime_hash()
assert validation['passed'] and test['passed'] and test['split'] == 'test'
assert test['physical_incidents'] == 60 and test['frozen_hash'] == claimed
assert test['selected'] == frozen['selected'] == validation['selected']
assert frozen['diagnostic_gate_passed']
training = 0
for method in ('expected-sarsa', 'two-step-sarsa'):
    for seed in (11, 22, 33, 44, 55):
        report = read(f'training/{method}/{seed}/report.json')
        assert report['passed'] and len(report['runs']) == 1
        saved = report['runs'][0]
        assert saved['complete'] and saved['episodes_completed'] == 10000 and saved['seed'] == seed
        training += 1
summaries = test['summaries']
groups = {}
for method in ('expected-sarsa', 'two-step-sarsa'):
    for form in ('tabular', 'hybrid'):
        rows = [summaries[f'{method}-{seed}-{form}'] for seed in (11,22,33,44,55)]
        groups[f'{method}-{form}'] = {key: sum(r[key] for r in rows)/5
                                    for key in ('accuracy', 'mean_return', 'mean_probes')}
for method in ('voi', 'dp'):
    groups[method] = {key: summaries[method][key] for key in ('accuracy','mean_return','mean_probes')}
promoted = [name for name, row in test['performance_promotion'].items()
            if 'sarsa' in name and row['passed']]
audit = dict(passed=True, training_runs_verified=training, physical_incidents=60,
             frozen_hash=claimed, runtime_hash=runtime_hash(), selected=test['selected'],
             method_results=groups, promoted_learners=promoted)
atomic_json(run / 'main-comparison-verification.json', audit)
print(json.dumps(audit, indent=2))
