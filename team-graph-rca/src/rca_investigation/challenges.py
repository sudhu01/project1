"""Synthetic perturbations of frozen snapshots, scored separately from real trials."""
from dataclasses import replace
from .backends.replay import ReplayBackend
from .environment import InvestigationEnv,score_decision
from .dataset import load_public_bundle,load_private_label,read_split
from .experiment import label_name
from .evaluate_decisions import investigate,summarize
from .comparison import load_frozen,policy_from_definition


def challenge_report(dataset,specs,model,config,frozen_path):
    if config.values['schema_version'] == 2:
        from .challenges_v2 import challenge_report_v2
        return challenge_report_v2(dataset, specs, model, config, frozen_path)
    frozen=load_frozen(frozen_path,dataset,config,model)
    _,paths=read_split(dataset,'test')
    definition=frozen['definitions'][frozen['selected']]
    reports={}
    for name in ('status-outage','double-prices','half-prices','unsupported-payment-stop','two-stops'):
        outcomes=[]
        for path in paths:
            public,topology,packets=load_public_bundle(path,specs)
            challenge_specs=specs
            if name=='status-outage':
                packets=tuple(replace(p,availability='ERROR',response_code='UNAVAILABLE',raw_json='{"reason":"synthetic_outage"}') for p in packets)
            elif name in ('double-prices','half-prices'):
                # Integer credits remain >= 1. Half-price has no effect on 1-credit tools.
                factor=2 if name=='double-prices' else .5
                challenge_specs=tuple(replace(s,credits=max(1,int(s.credits*factor))) for s in specs)
                from .catalogue import catalogue_hash
                public=replace(public,catalogue_hash=catalogue_hash(challenge_specs))
            elif name=='unsupported-payment-stop':
                packets=tuple(replace(p,response_code='STOPPED' if p.resource_id=='payment' else 'RUNNING',raw_json='{}') for p in packets)
            elif name=='two-stops':
                packets=tuple(replace(p,response_code='STOPPED' if p.resource_id in ('orders','catalogue-db') else 'RUNNING',raw_json='{}') for p in packets)
            objective=config.values['objective']
            env=InvestigationEnv(topology,challenge_specs,model,ReplayBackend(packets,challenge_specs),
                defer_utility=objective['defer'],correct_utility=objective['correct'],incorrect_utility=objective['incorrect'])
            policy=policy_from_definition(definition,model,challenge_specs,config)
            try:
                env.reset(public)
                outcome=investigate(env,policy,'synthetic-challenge')
                truth=label_name(load_private_label(path))
                out_of_scope=name in ('unsupported-payment-stop','two-stops')
                correct=env.decision.prediction=='DEFER' if out_of_scope else env.decision.prediction==truth
                # Multi-root/unsupported cases test acknowledged scope, not exact-root accuracy.
                utility=objective['defer'] if env.decision.prediction=='DEFER' else objective['incorrect'] if out_of_scope else score_decision(env.decision,truth,
                    correct=objective['correct'],incorrect=objective['incorrect'],defer=objective['defer'])
                outcome.update(incident_id=public.incident_id,truth='OUT_OF_SCOPE' if out_of_scope else truth,source_truth=truth,correct=correct,
                               covered=env.decision.prediction!='DEFER',return_value=outcome['probe_return']+utility)
                outcomes.append(outcome)
            finally: env.close()
        summary=summarize(outcomes,model.classes)
        summary['source_snapshots']=summary.pop('physical_incidents')
        if name in ('unsupported-payment-stop','two-stops'):
            summary['scope_acknowledgment_rate']=summary.pop('accuracy')
            for field in ('selective_accuracy','confusion','brier_score','calibration_bins'):
                summary.pop(field)
        reports[name]={'summary':summary,'outcomes':outcomes,'scope_acknowledgment_test':name in ('unsupported-payment-stop','two-stops')}
    return {'passed':True,'synthetic_perturbations_only':True,'additional_physical_incidents':0,
            'selected':frozen['selected'],'reports':reports,
            'not_applicable':['missing logs: log actions are disabled in stop_v1'],
            'physical_challenges_pending':['changed workload outside the two validated rates','unsupported fault','simultaneous faults']}
