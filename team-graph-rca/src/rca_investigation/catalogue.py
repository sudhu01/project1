from .contracts import ToolSpec, canonical, digest


RESPONSES = {
    'status': ('RUNNING', 'STOPPED', 'OTHER_STATE', 'UNAVAILABLE'),
    'cpu': ('NORMAL', 'HIGH', 'INSUFFICIENT_DATA', 'UNAVAILABLE'),
    'logs': ('NO_RELEVANT_ERROR', 'DEPENDENCY_ERROR', 'RESOURCE_ERROR', 'OTHER_ERROR', 'UNAVAILABLE'),
}


def build_catalogue(config, topology):
    values = config.values
    if values['schema_version'] == 2:
        from .contracts_v2 import ToolSpecV2
        specs = tuple(ToolSpecV2.from_dict(action) for action in values['actions'])
        for spec in specs:
            spec.validate_resource(topology)
        catalogue_hash(specs)
        return specs
    specs = []
    for template in ('status', 'cpu', 'logs'):
        settings = values['tools'][template]
        if not settings['enabled']:
            continue
        for resource in sorted(values['resources']):
            params = {'status': {}, 'cpu': {'window_seconds': 60}, 'logs': {'max_lines': 20}}[template]
            spec = ToolSpec(f'{template}/{resource}/v1', resource, template, settings['credits'],
                            RESPONSES[template], (template,), canonical(params))
            spec.validate_resource(topology)
            specs.append(spec)
    return tuple(specs)


def catalogue_hash(specs):
    ids = [s.action_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate action IDs')
    return digest([s.to_dict() for s in specs])


def action_order(specs):
    catalogue_hash(specs)
    return tuple(s.action_id for s in specs) + ('STOP', 'DEFER')


def mask(specs, acquired, credits, probes_left, capabilities, terminated=False):
    if terminated:
        return ()
    permitted = tuple(s.action_id for s in specs if probes_left > 0 and s.credits <= credits
                      and s.action_id not in acquired and set(s.capabilities) <= set(capabilities.get(s.resource_id, ())))
    return permitted + ('STOP', 'DEFER')
