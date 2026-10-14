"""Repeated frontend checkout, with timing only for the observed operation."""
from datetime import datetime, timezone
from time import perf_counter
import requests
from .workload import SockShopWorkload


class CheckoutWorkload(SockShopWorkload):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measure_cart = False
        self.cart_observations = []

    def cart_call(self, operation, method, path, **kwargs):
        started = perf_counter()
        row = {'resource_id': 'front-end', 'operation': operation}
        try:
            response = self.session.request(method, self.base_url + path, timeout=5, **kwargs)
            row.update(http_status=response.status_code, outcome='completed', duration_ms=(perf_counter() - started) * 1000)
        except requests.RequestException as exc:
            row.update(http_status=None, outcome='transport_failure', duration_ms=(perf_counter() - started) * 1000,
                       transport_reason='timeout' if isinstance(exc, requests.Timeout) else 'connection_error')
            raise
        finally:
            row['sample_time'] = datetime.now(timezone.utc).isoformat()
            self.cart_observations.append(row)
        return response

    def cart_sequence(self):
        first = self.cart_call('cart_read', 'GET', '/cart')
        first.raise_for_status()
        added = self.cart_call('cart_add', 'POST', '/cart', json={'id': self.item_id})
        if added.status_code != 201:
            raise RuntimeError('cart add failed')
        checked = self.cart_call('cart_verify', 'GET', '/cart')
        checked.raise_for_status()
        if not any(item.get('itemId', item.get('id')) == self.item_id for item in checked.json()):
            self.cart_observations[-1]['outcome'] = 'business_failure'
            raise RuntimeError('cart update not observed')
        removed = self.cart_call('cart_reset', 'DELETE', '/cart')
        if removed.status_code not in (200, 202, 204):
            raise RuntimeError('cart reset failed')

    def prepare(self):
        result = super().prepare()
        response = self.session.get(self.base_url + '/catalogue', timeout=5)
        response.raise_for_status()
        affordable = [row for row in response.json() if 0 < float(row['price']) < 80]
        if not affordable:
            raise RuntimeError('no product below demo payment limit')
        self.item_id = min(affordable, key=lambda row: float(row['price']))['id']
        return result

    def checkout(self):
        if not self.authenticated:
            raise ValueError('checkout requires its own prepared session')
        if self.measure_cart:
            self.cart_sequence()
        cleared = self.session.delete(self.base_url + '/cart', timeout=5)
        if cleared.status_code not in (200, 202, 204):
            raise RuntimeError('cart reset failed')
        cart = self.session.post(self.base_url + '/cart', json={'id': self.item_id}, timeout=5)
        if cart.status_code != 201:
            raise RuntimeError('cart preparation failed')
        verified = self.session.get(self.base_url + '/cart', timeout=5)
        verified.raise_for_status()
        if not any(item.get('itemId', item.get('id')) == self.item_id for item in verified.json()):
            raise RuntimeError('checkout cart update not observed')
        started = perf_counter()
        row = {'resource_id': 'front-end', 'operation': 'checkout', 'sample_time': datetime.now(timezone.utc).isoformat()}
        try:
            response = self.session.post(self.base_url + '/orders', json={}, timeout=5)
            row.update(http_status=response.status_code, outcome='business_failure' if 200 <= response.status_code < 500 and response.status_code != 201 else 'completed',
                       duration_ms=(perf_counter() - started) * 1000)
            if response.status_code == 201:
                try:
                    payload = response.json()
                    if not isinstance(payload, dict) or not payload.get('id'):
                        row['outcome'] = 'business_failure'
                except ValueError:
                    row['outcome'] = 'business_failure'
        except requests.RequestException as exc:
            row.update(http_status=None, outcome='transport_failure', duration_ms=(perf_counter() - started) * 1000,
                       transport_reason='timeout' if isinstance(exc, requests.Timeout) else 'connection_error')
        # The timestamp is completion time, so a request that finishes after the
        # window is excluded from that capture rather than leaking recovery.
        row['sample_time'] = datetime.now(timezone.utc).isoformat()
        return row

    def reset(self):
        if self.authenticated:
            response = self.session.delete(self.base_url + '/cart', timeout=5)
            if response.status_code not in (200, 202, 204):
                raise RuntimeError('cart cleanup failed')
