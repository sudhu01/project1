"""Frontend-observed checkout metrics from bounded workload observations."""
from ..encoding import encode_packet
from ..metrics.requests import summarize as errors
from ..metrics.latency import summarize as latency


class RequestMetricsBackend:
    version = 'workload_checkout_observations_v2'

    def __init__(self, specs, observations, encoder):
        self.specs = {s.action_id: s for s in specs}
        self.observations, self.encoder = observations, encoder
        self.incident = None

    def begin(self, incident):
        self.incident = incident

    def execute(self, spec):
        if self.incident is None or self.specs.get(spec.action_id) != spec or spec.resource_id != 'front-end':
            raise ValueError('unapproved request observation')
        params = __import__('json').loads(spec.parameters_json)
        if spec.template not in {'request_errors', 'latency'}:
            raise ValueError('unsupported request metric')
        threshold = self.encoder.values['thresholds'].get(spec.template)
        if threshold is None:
            raise ValueError('request metric threshold not fitted')
        from ..contracts import timestamp
        observations = [row for row in self.observations if row['operation'] == params['operation'] and timestamp(self.incident.window_start) <= timestamp(row['sample_time']) <= timestamp(self.incident.window_end)]
        kwargs = {'transport_threshold': self.encoder.values['thresholds']['transport_errors']} if spec.template == 'request_errors' else {}
        summary = (errors if spec.template == 'request_errors' else latency)(
            observations, spec.resource_id, params['operation'], threshold, params['min_samples'], **kwargs)
        return encode_packet(spec, self.incident, summary, self.version)

    def close(self):
        self.incident = None
