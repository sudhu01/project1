import pytest
from rca_investigation.workload import SockShopWorkload, WorkloadSpec


class Session:
    def __init__(self, status=201, payload=None):
        self.status = status
        self.payload = [] if payload is None else payload
        self.cookies = {'logged_in':'demo-session'}
        self.routes = []
    def get(self, url, **kwargs):
        self.routes.append(url)
        class Response:
            status_code = 200 if url.endswith('/login') else self.status
            def json(inner): return self.payload
        return Response()
    def close(self): pass


def test_prepared_orders_and_nonempty_catalogue():
    workload = SockShopWorkload(Session())
    with pytest.raises(ValueError):
        workload.request('/orders')
    workload.prepare()
    assert workload.request('/orders')['success']
    assert not workload.request('/catalogue')['success']
    workload = SockShopWorkload(Session(payload=[{'id':'item'}]))
    assert workload.request('/catalogue')['success']
    with pytest.raises(ValueError):
        workload.request('/admin')


def test_workload_limits():
    with pytest.raises(ValueError):
        WorkloadSpec(1, request_count=100000)
    with pytest.raises(ValueError):
        SockShopWorkload(base_url='https://production.example')
