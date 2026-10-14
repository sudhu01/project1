"""Immutable raw measurements with separately validated private labels."""
from pathlib import Path
import json
from time import sleep
from .contracts import canonical, digest, identifier, timestamp
from .dataset import atomic_json
from .contracts_v2 import IncidentPrivateV2


def write_raw(root, incident_id, public, measurements, private, receipt):
    identifier(incident_id)
    if '/' in incident_id or incident_id in {'.', '..'}:
        raise ValueError('incident ID must be a directory name')
    start, end = timestamp(public['window_start']), timestamp(public['window_end'])
    if start >= end or private.incident_id != incident_id or not isinstance(private, IncidentPrivateV2):
        raise ValueError('invalid raw incident context')
    if receipt.get('status') != 'ACCEPTED' or not receipt.get('recovery_confirmed'):
        raise ValueError('raw capture requires measured effect and recovery')
    if type(measurements) is not dict:
        raise ValueError('invalid raw measurements')
    # Measurements are keyed by source, contain only public observations and times.
    forbidden = {'cause_id', 'mechanism', 'scenario', 'severity', 'injector', 'private_label'}
    def validate(value):
        if type(value) is dict:
            if forbidden & set(value):
                raise ValueError('private injection data in public measurements')
            for key, item in value.items():
                if key == 'sample_time' and not start <= timestamp(item) <= end:
                    raise ValueError('raw sample outside observation window')
                validate(item)
        elif type(value) in (tuple, list):
            for item in value:
                validate(item)
    validate(measurements)
    if len(canonical(measurements).encode()) > 8 * 1024 * 1024:
        raise ValueError('raw capture exceeds byte limit')
    target = Path(root) / incident_id
    temporary = Path(root) / (incident_id + '.pending')
    if target.exists() or temporary.exists():
        raise ValueError('raw capture already exists')
    temporary.mkdir(parents=True)
    payloads = {'public.json': public, 'measurements.json': measurements,
                'private_label.json': private.to_dict(), 'capture_receipt.json': receipt}
    for name, value in payloads.items():
        atomic_json(temporary / name, value)
    atomic_json(temporary / 'checksums.json', {name: digest(value) for name, value in payloads.items()})
    publish_staged(temporary, target)
    return target


def publish_staged(temporary, target):
    """Retry transient Windows sharing locks without changing staged bytes."""
    for attempt in range(11):
        if target.exists():
            raise ValueError('raw capture already exists')
        try:
            temporary.rename(target)
            return
        except PermissionError:
            if attempt == 10:
                raise
            sleep(.5)


def read_raw(path, public_only=True):
    path = Path(path)
    hashes = json.loads((path / 'checksums.json').read_text(encoding='utf-8'))
    if set(hashes) != {'public.json', 'measurements.json', 'private_label.json', 'capture_receipt.json'}:
        raise ValueError('invalid raw manifest')
    names = ('public.json', 'measurements.json') if public_only else tuple(hashes)
    payloads = {}
    for name in names:
        value = json.loads((path / name).read_text(encoding='utf-8'))
        if digest(value) != hashes[name]:
            raise ValueError('raw capture checksum mismatch')
        payloads[name] = value
    return payloads, digest(hashes)
