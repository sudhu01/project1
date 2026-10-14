from dataclasses import replace
import json
import pytest
from rca_investigation.contracts import IncidentPrivate
from rca_investigation.dataset import *
from rca_investigation.fixtures import fixture_environment

DEFINITION = {'prices':[1], 'classes':['r0','NO_FAULT'], 'labels':['r0','NO_FAULT'], 'responses':[['ZERO'],['ONE']]}


def bundle(root, index, session):
    env, label = fixture_environment(DEFINITION)
    public = replace(env.incident, incident_id=f'opaque-{index}')
    private = IncidentPrivate(public.incident_id, ('r0',), 'container_stopped', '{}','{}','{}')
    packets = [env.backend.execute(spec) for spec in env.specs]
    receipt = {'status':'ACCEPTED','recovery_confirmed':True,'capture_session':session,'lab_batch':'batch', 'physical_episode':f'event-{index}'}
    path = write_bundle(root, public, env.topology, packets, private, receipt)
    return path, env.specs


def test_integrity_private_boundary_and_immutability(tmp_path):
    path, specs = bundle(tmp_path, 0, 's0')
    public, topology, packets = load_public_bundle(path, specs)
    assert public.incident_id == 'opaque-0'
    assert load_private_label(path).resources == ('r0',)
    data = json.loads((path/'responses.json').read_text())
    data[0]['response_code'] = 'ONE'
    (path/'responses.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_public_bundle(path, specs)


def test_grouped_splits_never_cross_sessions(tmp_path):
    for i in range(12):
        bundle(tmp_path, i, f's{i//2}')
    manifest = freeze_splits(tmp_path)
    membership = {incident:split for split,ids in manifest['splits'].items() for incident in ids}
    assert len(membership) == 12
    for i in range(0,12,2):
        assert membership[f'opaque-{i}'] == membership[f'opaque-{i+1}']
    with pytest.raises(ValueError):
        freeze_splits(tmp_path)
