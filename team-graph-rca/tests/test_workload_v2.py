from types import SimpleNamespace
import pytest
from rca_investigation.workload_v2 import CheckoutWorkload


class Response:
    def __init__(self, status, payload):
        self.status_code, self.payload = status, payload
    def json(self): return self.payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError('HTTP failure')


def worker(order):
    workload = CheckoutWorkload()
    workload.authenticated, workload.item_id = True, 'product'
    workload.session = SimpleNamespace(delete=lambda *a, **kw: Response(202, {}),
        get=lambda *a, **kw: Response(200, [{'itemId': 'product'}]),
        post=lambda url, **kw: Response(201, {}) if url.endswith('/cart') else order)
    return workload


def test_checkout_separates_http_and_semantic_failures():
    assert worker(Response(500, {})).checkout()['outcome'] == 'completed'
    assert worker(Response(201, {})).checkout()['outcome'] == 'business_failure'
    assert worker(Response(201, {'id': 'new-order'})).checkout()['outcome'] == 'completed'
    assert worker(Response(409, {})).checkout()['outcome'] == 'business_failure'


def test_cart_verification_has_its_own_timed_operation():
    workload = worker(Response(201, {'id': 'new-order'}))
    calls = []
    def request(method, url, **kwargs):
        calls.append(method)
        return Response(201 if method == 'POST' else 202 if method == 'DELETE' else 200,
                        [{'itemId': 'product'}] if method == 'GET' else {})
    workload.session.request = request
    workload.cart_sequence()
    assert calls == ['GET', 'POST', 'GET', 'DELETE']
    assert [row['operation'] for row in workload.cart_observations] == ['cart_read', 'cart_add', 'cart_verify', 'cart_reset']
    workload.session.request = lambda *a, **kw: Response(200, [])
    with pytest.raises(RuntimeError):
        workload.cart_sequence()
