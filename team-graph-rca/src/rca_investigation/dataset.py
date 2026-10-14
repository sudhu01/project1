"""Checksummed, immutable incident bundles and grouped split manifests."""
from pathlib import Path
import json
import os
import random
from time import sleep
from .contracts import IncidentPublic, IncidentPrivate, TopologySnapshot, ToolResult, canonical, digest
from .contracts_v2 import IncidentPrivateV2, decode_record


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(canonical(value)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def write_bundle(root, public, topology, packets, private, receipt):
    target = Path(root)/public.incident_id
    if isinstance(private, IncidentPrivateV2) != (getattr(public, 'schema_version', 1) == 2):
        raise ValueError('public and private schema versions differ')
    if target.exists():
        raise ValueError('immutable incident already exists')
    if private.incident_id != public.incident_id or receipt.get('status') != 'ACCEPTED' or not receipt.get('recovery_confirmed'):
        raise ValueError('only verified recovered incidents may enter the dataset')
    if isinstance(private, IncidentPrivateV2):
        private.validate_context(topology, public.task_id)
    elif private.mechanism == 'healthy' and private.resources or any(r not in topology.resources for r in private.resources):
        raise ValueError('invalid private resource label')
    responses = [p.to_dict() for p in packets]
    if len({p.action_id for p in packets}) != len(packets):
        raise ValueError('duplicate responses')
    payloads = {'public.json': public.to_dict(), 'topology.json': topology.to_dict(), 'responses.json': responses,
                'private_label.json': private.to_dict(), 'capture_receipt.json': receipt}
    temporary = Path(root)/(public.incident_id+'.pending')
    temporary.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        atomic_json(temporary/name, payload)
    atomic_json(temporary/'checksums.json', {name: digest(payload) for name, payload in payloads.items()})
    # OneDrive/antivirus can briefly hold a newly written directory on Windows.
    # Keep the bundle unpublished until the same-volume rename succeeds.
    for attempt in range(10):
        try:
            temporary.rename(target)
            break
        except PermissionError:
            if attempt == 9:
                raise
            sleep(.5)
    return target


def verify_bundle(path, public_only=False):
    path = Path(path)
    checksums = json.loads((path/'checksums.json').read_text(encoding='utf-8'))
    expected = {'public.json','topology.json','responses.json','private_label.json','capture_receipt.json'}
    if set(checksums) != expected:
        raise ValueError('invalid checksum manifest')
    for name in expected if not public_only else {'public.json', 'topology.json', 'responses.json'}:
        value = json.loads((path/name).read_text(encoding='utf-8'))
        if digest(value) != checksums[name]:
            raise ValueError('incident checksum mismatch')
    return digest(checksums)


def load_public_bundle(path, specs):
    """Runner validates integrity but never returns the private record."""
    path = Path(path)
    verify_bundle(path, public_only=True)
    public = decode_record(json.loads((path/'public.json').read_text(encoding='utf-8')), 'public')
    topology = TopologySnapshot.from_json((path/'topology.json').read_text(encoding='utf-8'))
    public.validate_context(topology, specs)
    packets = tuple(decode_record(p, 'packet') for p in json.loads((path/'responses.json').read_text(encoding='utf-8')))
    if len(packets) != len(specs) or {p.action_id for p in packets} != {s.action_id for s in specs}:
        raise ValueError('incomplete or duplicate response bundle')
    by_id = {s.action_id: s for s in specs}
    for packet in packets:
        packet.validate_context(by_id[packet.action_id], public)
    return public, topology, packets


def load_private_label(path):
    verify_bundle(path)
    return decode_record(json.loads((Path(path)/'private_label.json').read_text(encoding='utf-8')), 'private')


def freeze_splits(dataset, *, seed=11, group='capture_session', proportions=(.5,.25,.25), filename='splits.json'):
    if group not in {'capture_session', 'lab_batch'} or len(proportions) != 3 or abs(sum(proportions)-1) > 1e-9 or any(v <= 0 for v in proportions):
        raise ValueError('invalid grouped split settings')
    dataset = Path(dataset)
    target = dataset/filename
    if target.exists():
        raise ValueError('split manifest already frozen')
    groups, records = {}, {}
    for path in sorted(dataset.iterdir()):
        if not path.is_dir() or path.name.endswith('.pending'):
            continue
        bundle_hash = verify_bundle(path)
        receipt = json.loads((path/'capture_receipt.json').read_text(encoding='utf-8'))
        group_id = receipt[group]
        event_id = receipt['physical_episode']
        if event_id in records:
            raise ValueError('physical episode appears more than once')
        records[event_id] = (group_id, path.name, bundle_hash)
        groups.setdefault(group_id, []).append(path.name)
    if len(groups) < 3:
        raise ValueError('at least three independent groups required')
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    ntrain = max(1, min(len(ids)-2, round(len(ids)*proportions[0])))
    nvalidation = max(1, min(len(ids)-ntrain-1, round(len(ids)*proportions[1])))
    assignments = {group_id: 'train' if i<ntrain else 'validation' if i<ntrain+nvalidation else 'test' for i,group_id in enumerate(ids)}
    splits = {split: sorted(incident for group_id, incidents in groups.items() if assignments[group_id] == split for incident in incidents)
              for split in ('train','validation','test')}
    manifest = {'schema_version':1, 'seed':seed, 'group':group, 'assignments':assignments, 'splits':splits,
                'bundle_hashes':{incident:hash_value for _,incident,hash_value in records.values()}}
    manifest['hash'] = digest(manifest)
    atomic_json(target, manifest)
    return manifest


def freeze_scheduled_splits(dataset, schedule_state, filename='splits.json'):
    """Honor roles assigned before collection; never move test sessions after fitting."""
    dataset = Path(dataset)
    target = dataset/filename
    if target.exists():
        raise ValueError('split manifest already frozen')
    state = json.loads(Path(schedule_state).read_text(encoding='utf-8'))
    if not state.get('complete'):
        raise ValueError('capture schedule has not finished')
    roles = state['roles']
    if set(roles.values()) != {'train','validation','test'}:
        raise ValueError('all three reserved roles required')
    expected = set(state['accepted'].values())
    splits = {role:[] for role in ('train','validation','test')}
    hashes, events = {}, set()
    class_counts = {role:{} for role in splits}
    for incident in sorted(expected):
        path = dataset/incident
        hashes[incident] = verify_bundle(path)
        receipt = json.loads((path/'capture_receipt.json').read_text(encoding='utf-8'))
        if receipt['physical_episode'] in events:
            raise ValueError('duplicate physical episode')
        events.add(receipt['physical_episode'])
        role = roles[receipt['capture_session']]
        splits[role].append(incident)
        label = load_private_label(path)
        from .experiment import label_name
        name = label_name(label)
        class_counts[role][name] = class_counts[role].get(name,0)+1
    classes = set(class_counts['train'])
    if not classes or any(set(counts) != classes for counts in class_counts.values()):
        raise ValueError('each split requires complete class coverage')
    manifest = {'schema_version':1,'group':'capture_session','assignments':roles,'splits':splits,
                'bundle_hashes':hashes,'class_counts':class_counts,'schedule_hash':state['schedule_hash']}
    manifest['hash'] = digest(manifest)
    atomic_json(target,manifest)
    return manifest


def read_split(dataset, partition):
    if partition not in ('train','validation','test'):
        raise ValueError('unknown partition')
    dataset = Path(dataset)
    manifest = json.loads((dataset/'splits.json').read_text(encoding='utf-8'))
    claimed = manifest.pop('hash')
    if digest(manifest) != claimed:
        raise ValueError('split manifest checksum mismatch')
    manifest['hash'] = claimed
    if set(manifest['splits'])!={'train','validation','test'}:
        raise ValueError('invalid split names')
    all_ids=[incident for ids in manifest['splits'].values() for incident in ids]
    if len(set(all_ids))!=len(all_ids) or set(all_ids)!=set(manifest['bundle_hashes']):
        raise ValueError('split IDs overlap or differ from frozen bundles')
    ids = manifest['splits'][partition]
    for incident in ids:
        if verify_bundle(dataset/incident) != manifest['bundle_hashes'][incident]:
            raise ValueError('frozen incident changed')
    return manifest,tuple(dataset/incident for incident in ids)
