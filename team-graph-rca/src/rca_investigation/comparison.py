"""Validation selection and a frozen, resumable reserved-test comparison."""
import json
from pathlib import Path
import numpy as np
from .contracts import digest
from .dataset import atomic_json,read_split
from .evaluate_decisions import evaluate_paths,paired_session_interval
from .policies.dp import BoundedDP
from .policies.rules import RulePolicy
from .policies.expected_sarsa import TabularPolicy,HybridPolicy
from .train import runtime_hash


def validation_reference(config, model, split_hash):
    path = Path(config.values['paths']['runs']) / 'full-evidence-reference.json'
    if not path.exists():
        raise ValueError('validation full-evidence reference required before freezing v2 selection')
    reference = json.loads(path.read_text(encoding='utf-8'))
    if (reference['config_hash'], reference['model_hash'], reference['split_hash']) != (config.hash, model.hash, split_hash):
        raise ValueError('full-evidence reference belongs to different experiment inputs')
    return reference


def policy_identity(definition,model,config):
    return digest({'definition':definition,'model_hash':model.hash,'config_hash':config.hash,
                   'runtime_hash':runtime_hash(),
                   'checkpoint_hash':digest(json.loads(Path(definition['checkpoint']).read_text(encoding='utf-8')))
                       if 'checkpoint' in definition else None})


def policy_from_definition(definition, model, specs, config):
    name=definition['kind']
    if name in {'fixed-sequence', 'generalized-rule'}:
        from .policies.rules import FixedSequencePolicy, GeneralizedRulePolicy
        if name == 'fixed-sequence':
            return FixedSequencePolicy(definition['sequence'], definition.get('confidence', .95))
        return GeneralizedRulePolicy(specs, config.values['diagnoses'], definition.get('sequence', ()), definition.get('confidence', .95))
    if name in ('dp','voi'):
        return BoundedDP(model,specs,depth=definition['depth'],max_expanded_states=config.values['planner']['max_expanded_states'],
                         defer=config.values['objective']['defer'],correct=config.values['objective']['correct'],incorrect=config.values['objective']['incorrect'])
    if name=='rule':
        if config.values['schema_version'] == 2 and definition['name'] == 'dependency-rule':
            from .policies.rules import FrozenRuleTransfer
            return FrozenRuleTransfer(definition.get('seed', 11), config.values['objective'])
        return RulePolicy(definition['name'],definition.get('seed',11),config.values['objective'])
    payload=json.loads(Path(definition['checkpoint']).read_text(encoding='utf-8'))
    tabular=TabularPolicy.restore(payload['policy'])
    if tabular.hash!=payload['policy_hash'] or tabular.model_hash!=model.hash or tabular.task_hash!=config.hash:
        raise ValueError('selected policy checksum or task mismatch')
    if name=='hybrid':
        return HybridPolicy(tabular,BoundedDP(model,specs,depth=config.values['planner']['depth'],
            max_expanded_states=config.values['planner']['max_expanded_states'],
            defer=config.values['objective']['defer'],correct=config.values['objective']['correct'],
            incorrect=config.values['objective']['incorrect']),min_visits=5)
    if name!='tabular': raise ValueError('unknown policy kind')
    return tabular


def compare_validation(dataset, specs, model, config, training_root, destination, frozen_path):
    if Path(frozen_path).exists():
        raise ValueError('selection manifest already frozen')
    manifest,paths=read_split(dataset,'validation')
    if any(count<minimum for role,minimum in (('train',20),('validation',10),('test',10))
           for count in manifest['class_counts'][role].values()):
        raise ValueError('independent evaluation collection requires 20/10/10 incidents per class')
    definitions={name:{'kind':'rule','name':name,'seed':11} for name in ('stop-immediately','defer-immediately','dependency-rule')}
    for seed in config.values['training']['seeds']:
        definitions[f'random-valid-{seed}']={'kind':'rule','name':'random-valid','seed':seed}
    definitions.update(voi={'kind':'voi','depth':1},dp={'kind':'dp','depth':2})
    if config.values['schema_version'] == 2:
        from .adaptive_comparison import fixed_candidates
        definitions['generalized-rule'] = {'kind': 'generalized-rule'}
        definitions.update({f'fixed-{i}-{confidence}': {'kind': 'fixed-sequence', 'sequence': list(sequence), 'confidence': confidence}
                            for i, sequence in enumerate(fixed_candidates(specs, model=model))
                            for confidence in (.8, .9, .95, .99, 1.)})
        if len(specs) <= 8:
            definitions['dp-depth3'] = {'kind': 'dp', 'depth': 3}
    for method in ('expected-sarsa','two-step-sarsa'):
        for seed in config.values['training']['seeds']:
            checkpoint=Path(training_root)/method/str(seed)/'best.json'
            if not checkpoint.exists(): raise ValueError('all declared training seeds and methods must finish before selection')
            latest=json.loads((checkpoint.parent/'latest.json').read_text(encoding='utf-8'))
            if latest['episode']!=config.values['training']['episodes']:
                raise ValueError('training run is incomplete')
            for kind in ('tabular','hybrid'):
                definitions[f'{method}-{seed}-{kind}']={'kind':kind,'checkpoint':str(checkpoint.resolve())}
    summaries,outcomes={},{}
    for name,definition in definitions.items():
        policy=policy_from_definition(definition,model,specs,config)
        summaries[name],outcomes[name]=evaluate_paths(paths,specs,model,config,policy,policy_identity(definition,model,config))
    # Favor the simpler declared baseline when performance and cost tie.
    priorities={'dependency-rule':0,'voi':1,'dp':2,'stop-immediately':3,'defer-immediately':4}
    if config.values['schema_version'] == 2:
        priorities.update({'generalized-rule': 0, **{name: 1 for name in definitions if name.startswith('fixed-')}})
    selected=max(definitions,key=lambda name:(summaries[name]['mean_return'],-summaries[name]['mean_credits'],-priorities.get(name,10)))
    fixed = [name for name in definitions if name.startswith('fixed-')]
    selected_fixed = max(fixed, key=lambda name: (summaries[name]['mean_return'], -summaries[name]['mean_credits'])) if fixed else None
    pairs={name:paired_session_interval(rows,outcomes['dependency-rule']) for name,rows in outcomes.items() if name!='dependency-rule'}
    spread={}
    for method in ('expected-sarsa','two-step-sarsa'):
        for kind in ('tabular','hybrid'):
            values=[summaries[f'{method}-{seed}-{kind}']['mean_return'] for seed in config.values['training']['seeds']]
            spread[f'{method}-{kind}']={'seeds':config.values['training']['seeds'],'returns':values,'mean':float(np.mean(values)),
                                      'standard_deviation':float(np.std(values,ddof=1)),'range':[min(values),max(values)]}
    report={'passed':True,'split':'validation','selected':selected,'summaries':summaries,'outcomes':outcomes,
            'paired_vs_dependency_rule':pairs,'training_seed_spread':spread,'physical_incidents':len(paths)}
    if selected_fixed:
        report.update(selected_fixed=selected_fixed, fixed_search_budget=32,
            fixed_confidence_grid=[.8, .9, .95, .99, 1.],
            paired_vs_selected_fixed={name: paired_session_interval(rows, outcomes[selected_fixed])
                                      for name, rows in outcomes.items() if name != selected_fixed})
    atomic_json(destination,report)
    frozen={'schema_version':1,'dataset':str(Path(dataset).resolve()),'split_hash':manifest['hash'],'config_hash':config.hash,
            'model_hash':model.hash,'runtime_hash':runtime_hash(),'definitions':definitions,'selected':selected,
            'dependency_lock_hash':digest(Path('uv.lock').read_text(encoding='utf-8')),
            'checkpoint_hashes':{d['checkpoint']:digest(json.loads(Path(d['checkpoint']).read_text(encoding='utf-8')))
                                 for d in definitions.values() if 'checkpoint' in d},'validation_report_hash':digest(report)}
    if selected_fixed:
        frozen['selected_fixed'] = selected_fixed
        reference = validation_reference(config, model, manifest['hash'])
        frozen['diagnostic_gate_passed'] = reference['training_promotion_passed']
        frozen['diagnostic_reference_hash'] = digest(reference)
        frozen['promotion_target'] = {'minimum_return_advantage': .02, 'paired_interval_lower_bound': 0.,
                                      'minimum_accuracy': .90, 'selection_split': 'validation'}
    frozen['hash']=digest(frozen)
    if Path(frozen_path).exists(): raise ValueError('selection manifest already frozen')
    atomic_json(frozen_path,frozen)
    return report


def load_frozen(path,dataset,config,model):
    frozen=json.loads(Path(path).read_text(encoding='utf-8'))
    claimed=frozen.pop('hash')
    if digest(frozen)!=claimed or frozen['config_hash']!=config.hash or frozen['model_hash']!=model.hash:
        raise ValueError('frozen selection checksum or task mismatch')
    if frozen['runtime_hash']!=runtime_hash() or frozen['dataset']!=str(Path(dataset).resolve()):
        raise ValueError('source or dataset changed after selection')
    if frozen['dependency_lock_hash']!=digest(Path('uv.lock').read_text(encoding='utf-8')):
        raise ValueError('dependency lock changed after selection')
    for checkpoint,expected in frozen['checkpoint_hashes'].items():
        if digest(json.loads(Path(checkpoint).read_text(encoding='utf-8')))!=expected:
            raise ValueError('selected checkpoint changed')
    frozen['hash']=claimed
    return frozen


def compare_test(dataset,specs,model,config,frozen_path,destination):
    frozen=load_frozen(frozen_path,dataset,config,model)
    manifest,paths=read_split(dataset,'test')
    if manifest['hash']!=frozen['split_hash']: raise ValueError('frozen split changed')
    destination=Path(destination)
    if destination.exists():
        report=json.loads(destination.read_text(encoding='utf-8'))
        if report['frozen_hash']!=frozen['hash']: raise ValueError('test was already opened under another selection')
        if report['passed']: raise ValueError('reserved test comparison already completed')
    else:
        report={'passed':False,'frozen_hash':frozen['hash'],'split':'test','selected':frozen['selected'],
                'summaries':{},'outcomes':{},'physical_incidents':len(paths)}
        atomic_json(destination,report)
    for name,definition in frozen['definitions'].items():
        if name in report['summaries']: continue
        policy=policy_from_definition(definition,model,specs,config)
        summary,rows=evaluate_paths(paths,specs,model,config,policy,policy_identity(definition,model,config))
        report['summaries'][name],report['outcomes'][name]=summary,rows
        atomic_json(destination,report)
    report['paired_vs_dependency_rule']={name:paired_session_interval(rows,report['outcomes']['dependency-rule'])
                                         for name,rows in report['outcomes'].items() if name!='dependency-rule'}
    if frozen.get('selected_fixed'):
        baseline = frozen['selected_fixed']
        report['paired_vs_selected_fixed'] = {name: paired_session_interval(rows, report['outcomes'][baseline])
            for name, rows in report['outcomes'].items() if name != baseline}
        from .adaptive_comparison import paired_mixture_sweep
        hard = {'resource/payment', 'edge/orders/payment', 'resource/carts'}
        report['frozen_method_mixture_sweeps'] = {name: paired_mixture_sweep(rows, report['outcomes'][baseline], hard,
            [0., .1, .25, .5, .75, .9, 1.]) for name, rows in report['outcomes'].items() if name != baseline}
        target = frozen['promotion_target']
        report['performance_promotion'] = {name: {
            'passed': frozen['diagnostic_gate_passed']
                      and pair['mean_paired_return_difference'] >= target['minimum_return_advantage']
                      and pair['session_bootstrap_95'][0] > target['paired_interval_lower_bound']
                      and report['summaries'][name]['accuracy'] >= target['minimum_accuracy'],
            'target': target, 'baseline': baseline,
            'diagnostic_gate_passed': frozen['diagnostic_gate_passed']}
            for name, pair in report['paired_vs_selected_fixed'].items()}
    report['passed']=True
    atomic_json(destination,report)
    return report
