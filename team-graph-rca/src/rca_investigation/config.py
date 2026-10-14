"""Strict task settings, stored as immutable canonical JSON."""
from dataclasses import dataclass
import json
from pathlib import Path
import yaml
from .contracts import canonical, digest, integer, real


def keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"expected settings {sorted(expected)}")


@dataclass(frozen=True)
class TaskConfig:
    canonical_json: str

    @property
    def values(self):
        return json.loads(self.canonical_json)

    @property
    def hash(self):
        return digest(self.values)


def load_config(path, topology):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if type(value) is dict and value.get('schema_version') == 2:
        from .config_v2 import load_v2_config
        load_v2_config(value, topology)
        if value.get('encoding_hash') == '0' * 64:
            encoder_path = Path(value.get('paths', {}).get('encoder', ''))
            if encoder_path.is_file():
                from .encoding import Encoder
                encoder = Encoder.load(encoder_path)
                required = set()
                for action in value['actions']:
                    if action['template'] in {'cpu', 'memory', 'latency', 'request_errors'}:
                        required.add(action['template'])
                    if action['template'] == 'memory':
                        required.add('memory_growth')
                    if action['template'] == 'request_errors':
                        required.add('transport_errors')
                if not required <= set(encoder.values['thresholds']):
                    raise ValueError('encoder lacks valid training measurements for enabled actions')
                encoding_hash = encoder.hash
                value['encoding_hash'] = encoding_hash
                for action in value['actions']:
                    action['encoding_hash'] = encoding_hash
                    if action['template'] == 'logs':
                        from .contracts_v2 import VOCABULARIES
                        action['response_vocabulary'] = list(VOCABULARIES['logs']) + [
                            'DEPENDENCY_ERROR/' + peer for peer in encoder.values.get('validated_log_peers', {}).get(action['resource_id'], [])]
        return load_v2_config(value, topology)
    keys(value, {"schema_version", "task_id", "resources", "diagnoses", "entry_protocol", "tools", "episode", "objective", "model", "planner", "training"})
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or value['entry_protocol'] != 'manual_frontend':
        raise ValueError("unsupported task schema or entry protocol")
    if value['task_id'] != 'sockshop_stop_v1':
        raise ValueError("new task profiles require an explicit implementation")
    for group in ('resources', 'diagnoses'):
        items = value[group]
        if type(items) is not list or not items or any(type(v) is not str for v in items) or len(set(items)) != len(items):
            raise ValueError("invalid resource or diagnosis list")
    if not set(value['resources']) <= set(topology.resources) or not set(value['diagnoses']) <= set(value['resources']) | {'NO_FAULT'}:
        raise ValueError("unknown resource or diagnosis")
    if value['diagnoses'] != ['catalogue-db', 'orders', 'NO_FAULT'] or set(value['resources']) != {'front-end', 'catalogue', 'catalogue-db', 'orders', 'payment'}:
        raise ValueError("task scope differs from stop_v1")
    keys(value['tools'], {'status', 'cpu', 'logs'})
    for name, spec in value['tools'].items():
        keys(spec, {'enabled', 'credits', 'max_lines'} if name == 'logs' else {'enabled', 'credits'})
        if type(spec['enabled']) is not bool:
            raise ValueError("enabled must be boolean")
        integer(spec['credits'], 1)
    if not value['tools']['status']['enabled'] or value['tools']['cpu']['enabled'] or value['tools']['logs']['enabled'] or value['tools']['logs']['max_lines'] != 20:
        raise ValueError("stop_v1 enables status only; CPU and logs require a measured promotion")
    keys(value['episode'], {'observation_window_seconds', 'budget', 'max_probes'})
    for v in value['episode'].values():
        integer(v, 1)
    if value['episode']['observation_window_seconds'] != 60:
        raise ValueError("v1 requires a fixed 60 second window")
    keys(value['objective'], {'lambda_cost', 'correct', 'incorrect', 'defer'})
    for v in value['objective'].values():
        real(v)
    if not value['objective']['incorrect'] <= value['objective']['defer'] <= value['objective']['correct']:
        raise ValueError("invalid terminal utilities")
    keys(value['model'], {'prior', 'smoothing'})
    real(value['model']['smoothing'], 0, 1)
    if value['model']['prior'] != 'balanced_classes' or value['model']['smoothing'] == 0:
        raise ValueError("production requires balanced classes and nonzero smoothing")
    keys(value['planner'], {'depth', 'max_expanded_states'})
    for v in value['planner'].values():
        integer(v, 1)
    keys(value['training'], {'algorithm', 'episodes', 'alpha', 'gamma', 'epsilon_start', 'epsilon_end', 'epsilon_decay_episodes', 'validation_interval', 'seeds'})
    training = value['training']
    if training['algorithm'] != 'expected_sarsa' or training['gamma'] != 1:
        raise ValueError("unsupported training objective")
    for field in ('episodes', 'epsilon_decay_episodes', 'validation_interval'):
        integer(training[field], 1)
    for field in ('alpha', 'gamma', 'epsilon_start', 'epsilon_end'):
        real(training[field], 0, 1)
    if training['alpha'] == 0 or training['epsilon_end'] > training['epsilon_start']:
        raise ValueError("invalid learning rate or exploration schedule")
    if type(training['seeds']) is not list or not training['seeds'] or len(set(training['seeds'])) != len(training['seeds']):
        raise ValueError("invalid training seeds")
    for seed in training['seeds']:
        integer(seed)
    return TaskConfig(canonical(value))
