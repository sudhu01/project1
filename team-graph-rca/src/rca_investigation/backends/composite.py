"""Route each approved paid action to one evidence handler."""


class CompositeBackend:
    version = 'composite_v2'

    def __init__(self, specs, handlers, capabilities):
        self.specs = {s.action_id: s for s in specs}
        if len(self.specs) != len(specs) or any(s.template not in handlers for s in specs):
            raise ValueError('duplicate action or missing handler')
        self.handlers = dict(handlers)
        self.public_capabilities = dict(capabilities)
        self.incident = None

    def capabilities(self, topology):
        return {r: tuple(self.public_capabilities.get(r, ())) for r in topology.resources}

    def begin(self, incident):
        self.incident = incident
        for handler in self._unique_handlers():
            handler.begin(incident)

    def _unique_handlers(self):
        return {id(handler): handler for handler in self.handlers.values()}.values()

    def execute(self, spec):
        if self.incident is None or self.specs.get(spec.action_id) != spec:
            raise ValueError('unapproved action or missing window')
        packet = self.handlers[spec.template].execute(spec)
        packet.validate_context(spec, self.incident)
        return packet

    def close(self):
        for handler in self._unique_handlers():
            handler.close()
        self.incident = None
