from ..contracts import ToolResult
from ..contracts_v2 import decode_record


class ReplayBackend:
    """Answers purchased actions from packets, without loading labels."""
    version = 'frozen_replay_v1'

    def __init__(self, packets, specs):
        self._specs = {s.action_id: s for s in specs}
        self._packets = {}
        for packet in packets:
            if not isinstance(packet, ToolResult) or packet.action_id in self._packets or packet.action_id not in self._specs:
                raise ValueError('duplicate, invalid or unknown packet')
            self._packets[packet.action_id] = packet
        if set(self._specs) != set(self._packets):
            raise ValueError('incomplete response bundle')
        self._incident = None

    def capabilities(self, topology):
        return {r: tuple(sorted({s.template for s in self._specs.values() if s.resource_id == r})) for r in topology.resources}

    def begin(self, incident):
        for action, packet in self._packets.items():
            packet.validate_context(self._specs[action], incident)
        self._incident = incident

    def execute(self, spec):
        if self._incident is None:
            raise RuntimeError('backend has no observation window')
        if self._specs.get(spec.action_id) != spec:
            raise ValueError('unapproved action specification')
        return decode_record(self._packets[spec.action_id].to_dict(), 'packet')

    def close(self):
        self._incident = None
