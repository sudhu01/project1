"""Registered adaptive task settings, independent of v1 settings and paths."""
from pathlib import Path
import yaml
from .contracts import canonical, integer, real
from .contracts_v2 import CAUSES, PROFILES, ToolSpecV2, sha256
from .config import TaskConfig, keys


FAMILIES = {
    'sockshop_adaptive_v2a': {'status', 'cpu', 'request_errors', 'logs'},
    'sockshop_adaptive_v2b': {'status', 'cpu', 'request_errors', 'logs', 'latency'},
    'sockshop_adaptive_v2c': {'status', 'cpu', 'request_errors', 'logs', 'latency', 'memory'},
}


def load_v2_config(value, topology):
    keys(value, {'schema_version', 'task_id', 'resources', 'diagnoses', 'entry_protocol',
                 'actions', 'episode', 'objective', 'model', 'planner', 'training', 'encoding_hash', 'paths'})
    if type(value['schema_version']) is not int or value['schema_version'] != 2 or value['task_id'] not in PROFILES:
        raise ValueError('unregistered adaptive task')
    if value['entry_protocol'] != 'manual_frontend' or value['diagnoses'] != list(PROFILES[value['task_id']]):
        raise ValueError('adaptive diagnosis or entry contract mismatch')
    resources = value['resources']
    required = {'front-end', 'catalogue', 'catalogue-db', 'orders', 'payment'}
    if value['task_id'].endswith('v2c'):
        required |= {'carts', 'carts-db'}
    if type(resources) is not list or any(type(r) is not str for r in resources) or len(resources) != len(set(resources)) or set(resources) != required:
        raise ValueError('adaptive resource scope mismatch')
    if not set(resources) <= set(topology.resources):
        raise ValueError('unknown adaptive resource')
    for cause in CAUSES:
        if cause.cause_id in value['diagnoses']:
            cause.validate_topology(topology)
    sha256(value['encoding_hash'])
    if type(value['actions']) is not list or not value['actions']:
        raise ValueError('explicit action list required')
    specs = [ToolSpecV2.from_dict(action) for action in value['actions']]
    if len({s.action_id for s in specs}) != len(specs):
        raise ValueError('duplicate adaptive action')
    for spec in specs:
        spec.validate_resource(topology)
        if spec.resource_id not in resources or spec.template not in FAMILIES[value['task_id']] or spec.encoding_hash != value['encoding_hash']:
            raise ValueError('action incompatible with adaptive profile')
    if any(s.template == 'status' for s in specs) and {s.resource_id for s in specs if s.template == 'status'} != set(resources):
        raise ValueError('every observed resource requires status')
    keys(value['episode'], {'observation_window_seconds', 'budget', 'max_probes'})
    for number in value['episode'].values():
        integer(number, 1)
    if value['episode']['observation_window_seconds'] != 120:
        raise ValueError('adaptive window must be 120 seconds')
    keys(value['objective'], {'lambda_cost', 'correct', 'incorrect', 'defer'})
    for number in value['objective'].values():
        real(number)
    if not value['objective']['incorrect'] <= value['objective']['defer'] <= value['objective']['correct']:
        raise ValueError('invalid terminal utilities')
    keys(value['model'], {'prior', 'smoothing'})
    real(value['model']['smoothing'], 0, 1)
    if value['model']['prior'] != 'balanced_classes' or value['model']['smoothing'] == 0:
        raise ValueError('invalid adaptive evidence model')
    keys(value['planner'], {'depth', 'max_expanded_states'})
    for number in value['planner'].values():
        integer(number, 1)
    keys(value['training'], {'algorithm', 'episodes', 'alpha', 'gamma', 'epsilon_start', 'epsilon_end',
                             'epsilon_decay_episodes', 'validation_interval', 'seeds'})
    settings = value['training']
    if settings['algorithm'] != 'expected_sarsa' or settings['gamma'] != 1:
        raise ValueError('unsupported adaptive algorithm')
    for field in ('episodes', 'epsilon_decay_episodes', 'validation_interval'):
        integer(settings[field], 1)
    for field in ('alpha', 'gamma', 'epsilon_start', 'epsilon_end'):
        real(settings[field], 0, 1)
    if settings['alpha'] == 0 or settings['epsilon_end'] > settings['epsilon_start']:
        raise ValueError('invalid adaptive learning settings')
    seeds = settings['seeds']
    if type(seeds) is not list or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('invalid adaptive seeds')
    for seed in seeds:
        integer(seed)
    keys(value['paths'], {'dataset', 'raw_dataset', 'encoder', 'model', 'runs', 'resource_registry'})
    if len(set(value['paths'].values())) != len(value['paths']):
        raise ValueError('adaptive artifact paths must differ')
    for path in value['paths'].values():
        if type(path) is not str or not path or Path(path).is_absolute() or '..' in Path(path).parts or '\\' in path or ':' in path:
            raise ValueError('adaptive paths must stay inside the experiment repository')
        if any(part in {'integration_v1', 'sockshop_v1', 'stop_v1'} for part in Path(path).parts):
            raise ValueError('adaptive profile cannot reuse v1 paths')
    return TaskConfig(canonical(value))
