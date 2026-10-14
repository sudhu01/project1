"""Declare separate experiment identities for evidence subsets of one raw cohort."""
from pathlib import Path
import json
import yaml
from .adaptive_comparison import evidence_subsets
from .contracts import canonical, digest
from .config_v2 import load_v2_config


def declare_ablations(config, topology, specs, destination):
    target = Path(destination)
    if target.exists():
        raise ValueError('ablation family already declared')
    variants = []
    for name, subset in evidence_subsets(specs).items():
        value = json.loads(config.canonical_json)
        value['actions'] = [json.loads(spec.to_json()) for spec in subset]
        identity = digest({'parent': config.hash, 'actions': value['actions']})[:12]
        folder = f"{value['task_id']}/ablations/{identity}"
        value['paths'].update(dataset='datasets/' + folder, model='runs/' + folder + '/model.json',
                              runs='runs/' + folder)
        declared = load_v2_config(value, topology)
        filename = target / (identity + '.yaml')
        variants.append({'name': name, 'config': str(filename), 'config_hash': declared.hash,
                         'cause_scope': value['diagnoses'], 'actions': [s.action_id for s in subset],
                         'model_fitting': 'train_only', 'checkpoint_family': value['paths']['runs'],
                         'raw_dataset': value['paths']['raw_dataset'], 'encoder': value['paths']['encoder']})
        target.mkdir(parents=True, exist_ok=True)
        filename.write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')
    from .dataset import atomic_json
    manifest = {'schema_version': 2, 'parent_config_hash': config.hash, 'variants': variants,
                'same_raw_cohort': True, 'scope_changes': False, 'separate_training_required': True}
    manifest['hash'] = digest(manifest)
    atomic_json(target / 'manifest.json', manifest)
    return manifest


def declare_cost_sweep(config, topology, destination, costs=(0., .025, .05, .1, .2)):
    target = Path(destination)
    if target.exists() or len(set(costs)) != len(costs):
        raise ValueError('new cost-sweep directory and distinct settings required')
    from .contracts import real
    variants = []
    for cost in costs:
        real(cost)
        value = json.loads(config.canonical_json)
        value['objective']['lambda_cost'] = cost
        identity = digest({'parent': config.hash, 'objective': value['objective']})[:12]
        folder = f"{value['task_id']}/costs/{identity}"
        value['paths'].update(dataset='datasets/' + folder, model='runs/' + folder + '/model.json', runs='runs/' + folder)
        declared = load_v2_config(value, topology)
        filename = target / (identity + '.yaml')
        target.mkdir(parents=True, exist_ok=True)
        filename.write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')
        variants.append({'lambda_cost': cost, 'config': str(filename), 'config_hash': declared.hash,
                         'cause_scope': value['diagnoses'], 'checkpoint_family': value['paths']['runs']})
    from .dataset import atomic_json
    manifest = {'schema_version': 2, 'parent_config_hash': config.hash, 'variants': variants,
                'same_raw_cohort': True, 'scope_changes': False, 'separate_training_required': True}
    manifest['hash'] = digest(manifest)
    atomic_json(target / 'manifest.json', manifest)
    return manifest
