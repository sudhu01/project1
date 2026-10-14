import hashlib
import json
from pathlib import Path
import pytest
from rca_investigation.faults_v2 import AdaptiveLabDriver


def driver(tmp_path):
    helper = tmp_path / 'lab/adaptive-helper/rca-helper-lease'
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b'controlled-helper-fixture')
    pin = tmp_path / 'configs/lab_adaptive_v2_lease.json'
    pin.parent.mkdir()
    pin.write_text(json.dumps({'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest()}))
    value = object.__new__(AdaptiveLabDriver)
    value.root, value.ids, value.helper_leases = tmp_path, {'payment': 'a' * 64, 'carts': 'b' * 64}, {}
    value.state = lambda resource: {'state': 'running'}
    value.calls = []
    def command(argv):
        value.calls.append(argv)
        return ('[{"HostConfig":{"Memory":536870912}}]' if argv[0] == 'inspect' else '', {'argv': argv})
    value.command = command
    return value


def test_helper_dispatch_and_cleanup_require_matching_private_lease(tmp_path):
    value = driver(tmp_path)
    receipt = value.inject_helper('payment', 'cpu', seconds=210, workers=8)
    assert value.calls[-1][-2:] == ['-lease', receipt['lease']]
    assert receipt['container_id'] == 'a' * 64
    value.stop_helper('payment')
    assert value.calls[-1][-1] == receipt['lease']
    assert '$1' in value.calls[-1][-3] and 'readlink' in value.calls[-1][-3]
    with pytest.raises(ValueError, match='lease'):
        value.stop_helper('carts')


def test_helper_bounds_checked_before_dispatch(tmp_path):
    value = driver(tmp_path)
    for kwargs in ({'seconds': 241}, {'workers': True}, {'allocation_bytes': 1}):
        with pytest.raises(ValueError):
            value.inject_helper('payment', 'cpu', **kwargs)
    assert not value.calls
    with pytest.raises(ValueError, match='cap'):
        value.inject_helper('carts', 'memory', allocation_bytes=300 * 1024**2)
    assert not any(argv[0] in {'cp', 'exec'} for argv in value.calls)
