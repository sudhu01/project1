"""Run frozen benchmark agents on a saved validation incident, without training."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from rca_investigation.catalogue import build_catalogue
from rca_investigation.config import load_config
from rca_investigation.dataset import atomic_json, load_private_label, read_split
from rca_investigation.experiment import label_name, load_model, replay_environment
from rca_investigation.comparison import load_frozen, policy_from_definition, policy_identity
from rca_investigation.evaluate_decisions import investigate
from rca_investigation.environment import score_decision
from rca_investigation.topology import sockshop_topology

CASES = {
    'catalogue-db': 'resource/catalogue-db',
    'orders': 'resource/orders',
    'payment-cpu': 'resource/payment',
    'payment-delay': 'edge/orders/payment',
    'carts-memory': 'resource/carts',
    'healthy': 'NO_FAULT',
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', choices=tuple(CASES), default='healthy')
parser.add_argument('--incident', help='An incident ID from the validation split')
parser.add_argument('--sample', type=int, default=1, help='Case sample number, 1 through 10')
parser.add_argument('--policy', choices=('all', 'voi', 'dp', 'expected-sarsa', 'two-step-sarsa'), default='all')
parser.add_argument('--seed', type=int, choices=(11,22,33,44,55), default=11)
parser.add_argument('--hybrid', action='store_true', help='Use planner fallback for SARSA')
parser.add_argument('--list', action='store_true', help='List saved validation cases without running agents')
args = parser.parse_args()

config = load_config('configs/investigation_v2c_memory_separation_r1.yaml', sockshop_topology())
paths = config.values['paths']
manifest, incidents = read_split(paths['dataset'], 'validation')
# Labels only select an illustrative case and score it after the decision.
# Agents receive the unchanged public observation through the shared environment.
groups = {case: [p for p in incidents if label_name(load_private_label(p)) == cause]
          for case, cause in CASES.items()}
if args.list:
    for case, records in groups.items():
        print(f'{case}: {len(records)} validation incidents; sample 1 = {records[0].name}')
    raise SystemExit(0)
if args.incident:
    matches = [p for p in incidents if p.name == args.incident]
    if not matches:
        parser.error('--incident must identify a saved validation incident')
    incident = matches[0]
else:
    if not 1 <= args.sample <= len(groups[args.case]):
        parser.error('sample is outside the available case range')
    incident = groups[args.case][args.sample - 1]
specs = build_catalogue(config, sockshop_topology())
model = load_model(paths['model'], config, specs, manifest['hash'])
frozen = load_frozen(Path(paths['runs']) / 'final.json', paths['dataset'], config, model)
names = ['voi', 'dp', 'expected-sarsa', 'two-step-sarsa'] if args.policy == 'all' else [args.policy]
results = {}
print(f'Saved validation incident: {incident.name}. Replay only; no Docker injection.')
for name in names:
    key = name if name in ('voi', 'dp') else f"{name}-{args.seed}-{'hybrid' if args.hybrid else 'tabular'}"
    definition = frozen['definitions'][key]
    policy = policy_from_definition(definition, model, specs, config)
    env = replay_environment(incident, specs, model, config)
    try:
        result = investigate(env, policy, policy_identity(definition, model, config))
        truth = label_name(load_private_label(incident))
        objective = config.values['objective']
        score = result['probe_return'] + score_decision(env.decision, truth,
            correct=objective['correct'], incorrect=objective['incorrect'], defer=objective['defer'])
        result.update(score=score, truth=truth, correct=result['decision']['prediction'] == truth)
        results[key] = result
        print(f"\n{key}: {result['decision']['prediction']}, confidence {result['decision']['confidence']:.1%}")
        print(f"  probes={result['probes']}, credits={result['credits']}, score={score:.4f}, correct={result['correct']}")
        for step in result['trace']:
            print(f"  {step['action']} | choice {step['decision_latency_ms']:.3f} ms | total credits {step['credits']}")
    finally:
        env.close()
destination = Path('runs/manual_replay') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
atomic_json(destination, {'mode':'saved_validation_replay', 'incident':str(incident),
    'frozen_hash':frozen['hash'], 'results':results})
print(f'\nSaved demo output: {destination}')
