"""Exact-identity, bounded helpers with receipt-based cleanup."""
import hashlib
import json
from pathlib import Path
from .faults import LabDriver


class AdaptiveLabDriver(LabDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper_leases = {}

    def proxy_control(self, delay=None):
        from .backends.docker import compose_rows, resolve_containers
        output, _ = self.command(['compose', '-p', 'rca-lab', 'ps', '--all', '--no-trunc', '--format', 'json'])
        proxy = resolve_containers(compose_rows(output), ['payment-path-proxy'])['payment-path-proxy']
        if delay is not None and (type(delay) is not int or not 0 <= delay <= 1000):
            raise ValueError('unbounded path delay')
        args = ['exec', proxy, '/rca-path-proxy', '-mode', 'inspect' if delay is None else 'set']
        if delay is not None:
            args += ['-delay-ms', str(delay)]
        output, receipt = self.command(args)
        return {**json.loads(output), 'receipt': receipt}

    def inject_helper(self, resource, mode, seconds=120, workers=4, allocation_bytes=0):
        if (resource, mode) not in {('payment', 'cpu'), ('carts', 'memory')}:
            raise ValueError('unapproved helper target')
        if type(seconds) is not int or not 1 <= seconds <= 240 or type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError('unbounded helper')
        if self.state(resource)['state'] != 'running':
            raise RuntimeError('helper target not running')
        if mode == 'cpu' and allocation_bytes != 0:
            raise ValueError('CPU helper cannot allocate memory')
        binary = Path(self.root) / 'lab/adaptive-helper/rca-helper-lease'
        pin = json.loads((Path(self.root) / 'configs/lab_adaptive_v2_lease.json').read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest() != pin['helper_sha256']:
            raise ValueError('helper hash mismatch')
        if mode == 'memory':
            output, _ = self.command(['inspect', self.ids[resource]])
            limit = json.loads(output)[0]['HostConfig']['Memory']
            if type(allocation_bytes) is not int or not 0 < allocation_bytes <= min(512 * 1024**2, limit * .5):
                raise ValueError('allocation exceeds verified cap')
        self.command(['cp', str(binary), self.ids[resource] + ':/tmp/rca-helper'])
        from uuid import uuid4
        lease = str(uuid4())
        self.helper_leases[resource] = lease
        _, receipt = self.command(['exec', '-d', self.ids[resource], '/tmp/rca-helper', '-mode', mode,
                                  '-seconds', str(seconds), '-workers', str(workers), '-bytes', str(allocation_bytes), '-lease', lease])
        return {**receipt, 'resource': resource, 'mode': mode, 'helper_sha256': pin['helper_sha256'], 'lease': lease,
                'container_id': self.ids[resource]}

    def stop_helper(self, resource):
        if resource not in {'payment', 'carts'}:
            raise ValueError('unapproved cleanup target')
        self.state(resource)
        lease = self.helper_leases.get(resource)
        if lease is None:
            raise ValueError('helper cleanup requires its controller lease')
        # The command verifies the process executable before signalling it.
        command = 'if [ -f /tmp/rca-helper.pid ]; then p=$(head -n 1 /tmp/rca-helper.pid); l=$(tail -n 1 /tmp/rca-helper.pid); case "$p" in *[!0-9]*|"") exit 2;; esac; [ "$l" = "$1" ] || exit 4; if [ "$(readlink /proc/$p/exe)" = /tmp/rca-helper ]; then kill -TERM "$p"; else exit 3; fi; for i in 1 2 3 4 5 6 7 8 9 10; do [ -f /tmp/rca-helper.pid ] || exit 0; sleep 0.2; done; exit 5; fi'
        _, receipt = self.command(['exec', self.ids[resource], 'sh', '-c', command, 'rca-cleanup', lease])
        return receipt
