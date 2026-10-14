"""All frozen policies face the same separately reported synthetic challenges."""
from dataclasses import replace
from .backends.replay import ReplayBackend
from .environment import InvestigationEnv, score_decision
from .dataset import read_split, load_public_bundle, load_private_label
from .experiment import label_name
from .evaluate_decisions import investigate, summarize
from .comparison import load_frozen, policy_from_definition
from .catalogue import catalogue_hash


def challenge_report_v2(dataset, specs, model, config, frozen_path):
    frozen = load_frozen(frozen_path, dataset, config, model)
    _, paths = read_split(dataset, 'test')
    report = {'passed': True, 'synthetic_perturbations_only': True, 'additional_physical_incidents': 0,
              'definitions': frozen['definitions'], 'reports': {}}
    scenarios = {'stats-outage': {'cpu', 'memory'}, 'logs-outage': {'logs'},
                 'request-metrics-outage': {'request_errors', 'latency'}, 'all-telemetry-outage': {'cpu', 'memory', 'logs', 'request_errors', 'latency'},
                 'double-prices': set(), 'half-prices': set()}
    for scenario, failed in scenarios.items():
        methods = {}
        for method, definition in frozen['definitions'].items():
            outcomes = []
            for path in paths:
                public, topology, packets = load_public_bundle(path, specs)
                altered_specs = specs
                if failed:
                    families = {spec.action_id: spec.template for spec in specs}
                    packets = tuple(replace(packet, availability='ERROR', response_code='UNAVAILABLE',
                                            evidence_json='{}', reason='synthetic_source_outage')
                                    if families[packet.action_id] in failed else packet for packet in packets)
                else:
                    factor = 2 if scenario == 'double-prices' else .5
                    altered_specs = tuple(replace(spec, credits=max(1, int(spec.credits * factor))) for spec in specs)
                    public = replace(public, catalogue_hash=catalogue_hash(altered_specs))
                objective = config.values['objective']
                env = InvestigationEnv(topology, altered_specs, model, ReplayBackend(packets, altered_specs),
                    defer_utility=objective['defer'], correct_utility=objective['correct'], incorrect_utility=objective['incorrect'])
                policy = policy_from_definition(definition, model, altered_specs, config)
                try:
                    env.reset(public)
                    outcome = investigate(env, policy, 'synthetic-v2-challenge')
                    truth = label_name(load_private_label(path))
                    outcome.update(incident_id=public.incident_id, truth=truth, correct=env.decision.prediction == truth,
                                   covered=env.decision.prediction != 'DEFER',
                                   return_value=outcome['probe_return'] + score_decision(env.decision, truth,
                                      correct=objective['correct'], incorrect=objective['incorrect'], defer=objective['defer']))
                    outcomes.append(outcome)
                finally:
                    env.close()
            summary = summarize(outcomes, model.classes)
            summary['source_snapshots'] = summary.pop('physical_incidents')
            methods[method] = {'summary': summary, 'outcomes': outcomes}
        report['reports'][scenario] = methods
    return report
