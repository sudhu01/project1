class EvidenceLedger:
    def __init__(self):
        self._packets = {}
        self._credits = 0

    @property
    def packets(self):
        return tuple(self._packets.values())

    @property
    def evidence_ids(self):
        return tuple(p.evidence_id for p in self.packets)

    @property
    def acquired_actions(self):
        return tuple(self._packets)

    @property
    def credits(self):
        return self._credits

    def acquire(self, spec, packet, incident):
        packet.validate_context(spec, incident)
        existing = self._packets.get(spec.action_id)
        if existing is not None:
            if existing != packet:
                raise ValueError('conflicting duplicate evidence')
            return False
        if packet.evidence_id in self.evidence_ids:
            raise ValueError('duplicate packet identity under another action')
        self._packets[spec.action_id] = packet
        self._credits += spec.credits
        return True
