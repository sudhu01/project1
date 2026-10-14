"""Audit completed evidence variants without rerunning fitting or comparisons."""
import json
import argparse
from pathlib import Path
import yaml
from rca_investigation.train import runtime_hash
from rca_investigation.config import load_config
from rca_investigation.topology import sockshop_topology
from rca_investigation.dataset import atomic_json

def read(path):
    return json.loads(Path(path).read_text())

parser = argparse.ArgumentParser()
parser.add_argument('--family', choices=('ablations', 'costs'), default='ablations')
args = parser.parse_args()
manifest = read(f'configs/sockshop_adaptive_v2b-{args.family}-fcf9c90b8f89/manifest.json')
parent = read('runs/adaptive_v2b/phase-progress.json')
assert runtime_hash() == parent['source_hash']
results = []
training_runs = 0
for variant in manifest['variants']:
    config = yaml.safe_load(Path(variant['config']).read_text())
    assert config['paths']['raw_dataset'] == 'datasets/raw_v2b'
    assert config['diagnoses'] == variant['cause_scope']
    assert load_config(variant['config'], sockshop_topology()).hash == variant['config_hash']
    run = Path(config['paths']['runs'])
    phase, frozen, test, challenge, reference = [read(run / name) for name in (
        'phase-progress.json', 'final.json', 'compare-test.json', 'challenge.json', 'full-evidence-reference.json')]
    assert phase['passed'] and phase['phase'] == 'complete'
    assert phase['source_hash'] == frozen['runtime_hash'] == parent['source_hash']
    assert frozen['config_hash'] == variant['config_hash']
    assert test['passed'] and test['split'] == 'test' and test['physical_incidents'] == 50
    assert test['frozen_hash'] == frozen['hash'] and test['selected'] == frozen['selected']
    assert challenge['passed'] and challenge['synthetic_perturbations_only']
    assert challenge['additional_physical_incidents'] == 0
    assert challenge['definitions'] == frozen['definitions'] and len(challenge['reports']) == 6
    for method in ('expected-sarsa', 'two-step-sarsa'):
        for seed in config['training']['seeds']:
            training = read(run / 'training' / method / str(seed) / 'report.json')
            assert training['passed'] and len(training['runs']) == 1
            saved = training['runs'][0]
            assert saved['passed'] and saved['complete'] and saved['replay_training_only']
            assert saved['episodes_completed'] == 10000 and saved['seed'] == seed
            training_runs += 1
    selected = test['summaries'][test['selected']]
    promotions = [name for name, row in test['performance_promotion'].items()
                  if name.startswith(('expected-sarsa-', 'two-step-sarsa-')) and row['passed']]
    results.append(dict(evidence=variant.get('name', str(config['objective']['lambda_cost'])),
        lambda_cost=config['objective']['lambda_cost'], selected=test['selected'],
        diagnostic_accuracy=reference['accuracy'], diagnostic_gate_passed=reference['training_promotion_passed'],
        test_accuracy=selected['accuracy'], test_return=selected['mean_return'],
        test_probes=selected['mean_probes'], rl_promotions=len(promotions),
        promoted_methods=promotions, frozen_hash=frozen['hash']))
audit = dict(passed=True, variants_verified=len(results), training_runs_verified=training_runs,
             raw_cohort_shared=True, scope_unchanged=True, results=results)
destination = 'cost-sweep-verification.json' if args.family == 'costs' else 'evidence-sweep-verification.json'
atomic_json(Path('runs/adaptive_v2b') / destination, audit)
print(json.dumps(audit, indent=2))
