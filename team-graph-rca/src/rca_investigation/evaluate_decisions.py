"""Independent decision scores and paired session uncertainty."""
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .contracts import digest
from .dataset import load_private_label, read_split
from .environment import score_decision
from .experiment import label_name, replay_environment


def investigate(env, policy, policy_hash):
    trace=[]
    while not env.terminated:
        observation=env.observation()
        started=perf_counter()
        action=policy.choose(observation)
        elapsed=(perf_counter()-started)*1000
        _,reward,done=env.step(action,policy_hash=policy_hash)
        trace.append({'action':action,'reward':reward,'terminal':done,'decision_latency_ms':elapsed,
                      'stats':{k:v for k,v in getattr(policy,'stats',{}).items() if k!='values'},
                      'evidence_ids':list(env.ledger.evidence_ids),'credits':env.ledger.credits})
    return {'decision':env.decision.to_dict(),'trace':trace,'credits':env.ledger.credits,'probes':env.probes,
            'probe_return':env.probe_return,'probabilities':env.model.diagnosis(env.history),
            'packets':[p.to_dict() for p in env.ledger.packets]}


def evaluate_paths(paths, specs, model, config, policy, policy_hash='evaluation'):
    outcomes=[]
    for path in paths:
        env=replay_environment(path,specs,model,config)
        try:
            outcome=investigate(env,policy,policy_hash)
            # Private truth enters only after the public investigation has ended.
            truth=label_name(load_private_label(path))
            receipt=json.loads((Path(path)/'capture_receipt.json').read_text(encoding='utf-8'))
            decision=env.decision
            outcome.update(incident_id=env.incident.incident_id,truth=truth,capture_session=receipt['capture_session'],
                           correct=decision.prediction==truth,covered=decision.prediction!='DEFER',
                           return_value=outcome['probe_return']+score_decision(decision,truth,
                               correct=config.values['objective']['correct'],incorrect=config.values['objective']['incorrect'],defer=config.values['objective']['defer']))
            outcomes.append(outcome)
        finally:
            env.close()
    return summarize(outcomes,model.classes),outcomes


def summarize(outcomes, classes):
    if not outcomes:
        raise ValueError('empty evaluation set')
    n=len(outcomes)
    covered=[o for o in outcomes if o['covered']]
    probes=sum(o['probes'] for o in outcomes)
    packets=[p for o in outcomes for p in o['packets']]
    steps=[t for o in outcomes for t in o['trace']]
    bins=[]
    for low in np.arange(0,1,.1):
        rows=[o for o in covered if low <= o['decision']['confidence'] < low+.1 or low>.89 and o['decision']['confidence']==1]
        if rows:
            bins.append({'lower':float(low),'count':len(rows),'mean_confidence':float(np.mean([o['decision']['confidence'] for o in rows])),
                         'accuracy':float(np.mean([o['correct'] for o in rows]))})
    brier=np.mean([sum((o['probabilities'][c]-(o['truth']==c))**2 for c in classes) for o in outcomes])
    report = {'physical_incidents':n,'accuracy':sum(o['correct'] for o in outcomes)/n,'coverage':len(covered)/n,
            'selective_accuracy':sum(o['correct'] for o in covered)/len(covered) if covered else None,
            'mean_credits':float(np.mean([o['credits'] for o in outcomes])),
            'mean_probes':probes/n,'mean_return':float(np.mean([o['return_value'] for o in outcomes])),
            'tool_failure_rate':sum(p['availability']!='OK' for p in packets)/len(packets) if packets else 0,
            'repeated_request_attempts':0,'termination_reasons':dict(Counter(o['decision']['stop_reason'] for o in outcomes)),
            'confusion':{c:dict(Counter(o['decision']['prediction'] for o in outcomes if o['truth']==c)) for c in classes},
            'brier_score':float(brier),'calibration_bins':bins,
            'decision_latency_ms':float(np.mean([s['decision_latency_ms'] for s in steps])),
            'planner_expanded_states':sum(s['stats'].get('expanded_states',0) for s in steps),
            'fallback_frequency':sum(s['stats'].get('fallback',False) for s in steps)/len(steps),
            'unseen_state_fraction':sum(s['stats'].get('unseen',False) for s in steps)/len(steps),
            'lightly_visited_fraction':sum(s['stats'].get('lightly_visited',False) for s in steps)/len(steps)}
    if any(c.startswith(('resource/', 'edge/')) for c in classes):
        for kind, prefix in (('resource', 'resource/'), ('path', 'edge/')):
            rows = [outcome for outcome in outcomes if outcome['truth'].startswith(prefix)]
            report[kind + '_accuracy'] = sum(row['correct'] for row in rows) / len(rows) if rows else None
        report['mechanism_prediction_enabled'] = False
    return report


def paired_session_interval(first, second, seed=11, samples=2000):
    left={o['incident_id']:o for o in first}
    right={o['incident_id']:o for o in second}
    if set(left)!=set(right):
        raise ValueError('paired comparisons require identical incident IDs')
    sessions={}
    for incident,o in left.items():
        if o['capture_session']!=right[incident]['capture_session']:
            raise ValueError('capture session mismatch')
        sessions.setdefault(o['capture_session'],[]).append(o['return_value']-right[incident]['return_value'])
    ids=sorted(sessions)
    rng=np.random.default_rng(seed)
    bootstrap=[]
    for _ in range(samples):
        values=[d for session in rng.choice(ids,size=len(ids),replace=True) for d in sessions[session]]
        bootstrap.append(float(np.mean(values)))
    return {'mean_paired_return_difference':float(np.mean([d for values in sessions.values() for d in values])),
            'session_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),'independent_sessions':len(ids),
            'seed':seed,'resamples':samples}
