"""Describe the failed validation gate using frozen train and validation evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from rca_investigation.dataset import atomic_json

root = Path.cwd()
run = root / 'runs/adaptive_v2c'
splits = json.loads((root / 'datasets/adaptive_v2c/splits.json').read_text())
reference = json.loads((run / 'full-evidence-reference.json').read_text())
encoder = json.loads((root / 'experiments/v2c/encoder.json').read_text())['encoder']
result = {}
for partition in ('train', 'validation'):
    groups = defaultdict(lambda: defaultdict(Counter))
    fractions = defaultdict(list)
    for incident in splits['splits'][partition]:
        bundle = root / 'datasets/adaptive_v2c' / incident
        label = json.loads((bundle / 'private_label.json').read_text())
        cause = label['cause_id']
        rows = json.loads((bundle / 'responses.json').read_text())
        for row in rows:
            if row['action_id'] in {'memory/carts/v2', 'latency/front-end/v2', 'logs/carts/v2'}:
                groups[cause][row['action_id']][row['response_code']] += 1
            if row['action_id'] == 'memory/carts/v2':
                evidence = json.loads(row['evidence_json'])
                if 'limit_fraction' in evidence:
                    fractions[cause].append(evidence['limit_fraction'])
    result[partition] = {cause: dict(codes={action: dict(counts) for action, counts in actions.items()},
        memory_fraction_min=min(fractions[cause]) if fractions[cause] else None,
        memory_fraction_max=max(fractions[cause]) if fractions[cause] else None)
        for cause, actions in groups.items()}
audit = dict(validation_accuracy=reference['accuracy'], gate_passed=False,
             confusion=reference['confusion'], reserved_test_opened=False,
             memory_threshold=encoder['thresholds_by_resource']['memory']['carts'],
             results=result)
atomic_json(run / 'diagnostic-gate-investigation.json', audit)
print(json.dumps(audit, indent=2))
