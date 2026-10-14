"""Episode-boundary training with complete atomic continuation state."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .contracts import digest
from .dataset import atomic_json,read_split,load_private_label,load_public_bundle
from .experiment import replay_environment,label_name
from .evaluate_decisions import evaluate_paths
from .policies.expected_sarsa import TabularPolicy
from .policies.n_step_sarsa import learn_episode


def runtime_hash():
    root=Path(__file__).parent
    return digest({str(p.relative_to(root)):digest(p.read_text(encoding='utf-8')) for p in sorted(root.rglob('*.py'))})


def train_loop(factory, identifiers, validate, model_hash, task_hash, settings, destination, *, seed=11,
               method='expected-sarsa', input_hashes=None, resume=False, stop_after=None):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    latest=destination/'latest.json'
    identity={'settings':settings,'model_hash':model_hash,'task_hash':task_hash,'method':method,'seed':seed,
              'input_hashes':input_hashes or {},'runtime_hash':runtime_hash(),'sampler_ids':list(identifiers)}
    policy=TabularPolicy(model_hash,task_hash,seed,settings['alpha'],settings['gamma'])
    sampler=np.random.default_rng(seed+100000)
    state={'identity':identity,'episode':0,'metrics':[],'selection':None,'policy':policy.state(),'sampler':sampler.bit_generator.state}
    if resume:
        state=json.loads(latest.read_text(encoding='utf-8'))
        if state['identity']!=identity:
            raise ValueError('checkpoint task, inputs, runtime or training settings changed')
        policy=TabularPolicy.restore(state['policy'])
        sampler.bit_generator.state=state['sampler']
    elif latest.exists():
        raise ValueError('existing training run requires --resume')
    total=settings['episodes']
    boundary=min(total,stop_after) if stop_after is not None else total
    if boundary<state['episode']:
        raise ValueError('stop boundary precedes saved episode')
    last_trace=[]
    for episode in range(state['episode'],boundary):
        epsilon=settings['epsilon_start']+(settings['epsilon_end']-settings['epsilon_start'])*min(1,episode/settings['epsilon_decay_episodes'])
        identifier=identifiers[int(sampler.integers(len(identifiers)))]
        env,label=factory(identifier)
        try:
            result=learn_episode(env,label,policy,epsilon,method=method)
            last_trace=result['trace']
        finally:
            env.close()
        state['episode']=episode+1
        if state['episode']%settings['validation_interval']==0 or state['episode']==total:
            # Validation cannot populate the training Q table or consume its RNG.
            summary=validate(TabularPolicy.restore(json.loads(json.dumps(policy.state()))))
            metric={'episode':state['episode'],'epsilon':epsilon,'validation':summary}
            state['metrics'].append(metric)
            selection=(summary['mean_return'],-summary['mean_credits'])
            previous=state['selection']
            if previous is None or selection>tuple(previous['value']):
                state['selection']={'episode':state['episode'],'value':list(selection)}
                atomic_json(destination/'best.json',{'identity':identity,'episode':state['episode'],
                    'selection':state['selection'],'policy':policy.state(),'policy_hash':policy.hash})
        if state['episode']%50==0 or state['episode']==boundary:
            state['policy']=policy.state()
            state['sampler']=sampler.bit_generator.state
            state['last_behavior_trace']=last_trace
            atomic_json(latest,state)
            atomic_json(destination/'metrics.json',state['metrics'])
    return {'passed':True,'episodes_completed':state['episode'],'complete':state['episode']==total,
            'method':method,'seed':seed,'selection':state['selection'],'latest':str(latest),
            'policy_hash':policy.hash,'replay_training_only':True}


def train_dataset(dataset, specs, model, config, destination, *, seed=11,method='expected-sarsa',resume=False,stop_after=None):
    manifest,paths=read_split(dataset,'train')
    _,validation=read_split(dataset,'validation')
    # Frozen packets are immutable. Read each training bundle once, then create
    # fresh public environments without repeatedly touching OneDrive files.
    by_id={p.name:(load_public_bundle(p,specs),label_name(load_private_label(p))) for p in paths}
    def factory(identifier):
        from .backends.replay import ReplayBackend
        from .environment import InvestigationEnv
        (public,topology,packets),label=by_id[identifier]
        objective=config.values['objective']
        env=InvestigationEnv(topology,specs,model,ReplayBackend(packets,specs),defer_utility=objective['defer'],
                             correct_utility=objective['correct'],incorrect_utility=objective['incorrect'])
        env.reset(public)
        return env,label
    def validate(policy):
        return evaluate_paths(validation,specs,model,config,policy)[0]
    return train_loop(factory,tuple(by_id),validate,model.hash,config.hash,config.values['training'],destination,
                      seed=seed,method=method,resume=resume,stop_after=stop_after,
                      input_hashes={'split':manifest['hash'],'lock':digest(Path('uv.lock').read_text(encoding='utf-8'))})


def train_fixture(name, destination, *, seed=11,episodes=2500,method='expected-sarsa',resume=False,stop_after=None):
    from importlib.resources import files
    from .fixtures import fixture_environment
    from .environment import score_decision
    definitions=json.loads(files('rca_investigation').joinpath('decision_fixtures.json').read_text())
    from .adaptive_fixtures import three_probe_definition, contextual_definition
    definitions.update({'three-probe': three_probe_definition(), 'contextual': contextual_definition()})
    if name not in definitions or not 1<=episodes<=10000:
        raise ValueError('known fixture and bounded episode count required')
    definition=definitions[name]
    env,_=fixture_environment(definition)
    model_hash=env.model.hash
    env.close()
    def factory(row): return fixture_environment(definition,row=int(row))
    def validate(policy):
        returns,credits=[],[]
        for row in range(len(definition['labels'])):
            env,label=factory(row)
            try:
                while not env.terminated: env.step(policy.choose(env.observation()))
                returns.append(env.probe_return+score_decision(env.decision,label))
                credits.append(env.ledger.credits)
            finally: env.close()
        return {'mean_return':float(np.mean(returns)),'mean_credits':float(np.mean(credits))}
    settings={'episodes':episodes,'alpha':.1,'gamma':1,'epsilon_start':.2,'epsilon_end':.05,
              'epsilon_decay_episodes':2000,'validation_interval':250}
    return train_loop(factory,tuple(str(i) for i in range(len(definition['labels']))),validate,model_hash,
                      digest(definition),settings,destination,seed=seed,method=method,resume=resume,stop_after=stop_after,
                      input_hashes={'fixture':digest(definition)})
