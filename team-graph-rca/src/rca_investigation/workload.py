"""Seeded, bounded requests against explicitly verified front-end routes."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic, perf_counter, sleep
import numpy as np
import requests


@dataclass(frozen=True)
class WorkloadSpec:
    seed: int
    request_count: int = 12
    interval_seconds: float = 1.0
    concurrency: int = 1

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0 or type(self.request_count) is not int or not 1 <= self.request_count <= 120:
            raise ValueError('bounded seed and request count required')
        if not .1 <= self.interval_seconds <= 10 or not 1 <= self.concurrency <= 4:
            raise ValueError('invalid workload rate or concurrency')


class SockShopWorkload:
    def __init__(self, session=None, base_url='http://localhost'):
        if base_url != 'http://localhost':
            raise ValueError('workload may target only the local lab')
        self.base_url = base_url
        self.session = session or requests.Session()
        self.authenticated = False

    def prepare(self):
        # Seeded demo credentials belong only to the isolated Sock Shop dataset.
        login = self.session.get(self.base_url+'/login', auth=('user', 'password'), timeout=5)
        self.authenticated = login.status_code == 200 and 'logged_in' in self.session.cookies
        if not self.authenticated:
            raise RuntimeError('demo login did not establish an orders session')
        return {'login_status': login.status_code, 'authenticated': True}

    def request(self, route):
        if route not in ('/catalogue', '/orders') or route == '/orders' and not self.authenticated:
            raise ValueError('unapproved or unprepared workload route')
        started = perf_counter()
        try:
            response = self.session.get(self.base_url+route, timeout=5)
            try:
                payload = response.json()
            except ValueError:
                payload = None
            # Empty catalogue does not establish successful database access.
            success = 200 <= response.status_code < 300 and isinstance(payload, list) and (bool(payload) if route == '/catalogue' else True)
            return {'route': route, 'success': success, 'status_code': response.status_code,
                    'latency_ms': (perf_counter()-started)*1000, 'items': len(payload) if isinstance(payload, list) else None}
        except requests.RequestException:
            return {'route': route, 'success': False, 'status_code': None, 'latency_ms': (perf_counter()-started)*1000}

    def run(self, spec, routes=('/catalogue', '/orders')):
        rng = np.random.default_rng(spec.seed)
        started = monotonic()
        offsets = np.maximum(0, np.arange(spec.request_count)*spec.interval_seconds + rng.uniform(-.1,.1,spec.request_count)*spec.interval_seconds)
        results = []
        # Each worker creates requests through a shared prepared session only
        # for the first supported concurrency. Parallel promotion needs per-worker sessions.
        if spec.concurrency != 1:
            raise ValueError('the verified v1 workload supports concurrency 1')
        for index, offset in enumerate(offsets):
            wait = started+float(offset)-monotonic()
            if wait > 0:
                sleep(wait)
            results.append(self.request(routes[index % len(routes)]))
        elapsed = monotonic()-started
        return {'seed': spec.seed, 'request_count': spec.request_count, 'concurrency': spec.concurrency,
                'scheduled_offsets': offsets.tolist(), 'achieved_requests_per_second': len(results)/max(elapsed,1e-9),
                'successes': sum(r['success'] for r in results), 'elapsed_seconds': elapsed, 'requests': results}

    def close(self):
        self.session.close()
