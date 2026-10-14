"""Declare profiles before fitting. A zero hash explicitly means UNFITTED."""
from pathlib import Path
import json
import yaml
from rca_investigation.contracts import canonical
from rca_investigation.contracts_v2 import PROFILES, VOCABULARIES
from rca_investigation.topology import sockshop_topology

root = Path(__file__).resolve().parents[1]
base = yaml.safe_load((root / 'configs/investigation_v1.yaml').read_text())
topology = sockshop_topology()
for stage in ('v2a', 'v2b', 'v2c'):
    task = 'sockshop_adaptive_' + stage
    resources = list(base['resources']) + (['carts', 'carts-db'] if stage == 'v2c' else [])
    actions = [('status', resource) for resource in sorted(resources)] + [
        ('cpu', 'payment'), ('request_errors', 'front-end'), ('logs', 'orders'), ('logs', 'payment')]
    if stage != 'v2a':
        actions.append(('latency', 'front-end'))
    if stage == 'v2c':
        actions += [('memory', 'carts'), ('logs', 'carts')]
    definitions = []
    for family, resource in actions:
        params = {} if family == 'status' else {'window_seconds': 120}
        vocabulary = list(VOCABULARIES[family])
        if family == 'logs':
            params.update(max_lines=20, max_bytes=8192)
            vocabulary += ['DEPENDENCY_ERROR/' + peer for caller, peer in topology.calls if caller == resource]
        if family in {'request_errors', 'latency'}:
            params.update(operation='checkout', min_samples=20)
        definitions.append({'schema_version': 2, 'action_id': f'{family}/{resource}/v2', 'template': family,
            'resource_id': resource, 'parameters_json': canonical(params), 'credits': 1 if family == 'status' else 2,
            'required_capabilities': [family], 'response_vocabulary': vocabulary, 'encoding_hash': '0' * 64})
    value = {**base, 'schema_version': 2, 'task_id': task, 'resources': resources,
             'diagnoses': list(PROFILES[task]), 'actions': definitions, 'encoding_hash': '0' * 64,
             'episode': {'observation_window_seconds': 120, 'budget': 10, 'max_probes': 5},
             'paths': {'dataset': f'datasets/adaptive_{stage}', 'raw_dataset': f'datasets/raw_{stage}',
                       'encoder': f'experiments/{stage}/encoder.json', 'model': f'runs/adaptive_{stage}/model.json',
                       'runs': f'runs/adaptive_{stage}', 'resource_registry': 'configs/resources_sockshop_v2.yaml'}}
    del value['tools']
    target = root / f'configs/investigation_{stage}.yaml'
    if target.exists():
        raise ValueError('profile already declared')
    target.write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')
    scenarios = ['catalogue_db_stop', 'orders_stop', 'payment_cpu', 'healthy', 'healthy_higher_load']
    if stage != 'v2a':
        scenarios.append('payment_path_delay')
    if stage == 'v2c':
        scenarios.append('carts_memory')
    pilot = {'schema_version': 2, 'task_id': task, 'lab_batch': stage + '-pilot',
             'sessions': [{'id': stage + '-mechanics-1', 'role': 'pilot', 'scenarios': scenarios}]}
    (root / f'configs/capture_{stage}_pilot.yaml').write_text(yaml.safe_dump(pilot, sort_keys=False), encoding='utf-8')
