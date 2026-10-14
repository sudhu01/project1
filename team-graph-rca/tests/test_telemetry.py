import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

PIPELINE = Path(__file__).resolve().parents[1] / 'pipeline'


def module(name):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / (name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def builder():
    kg = module('kg_builder').KGBuilder.__new__(module('kg_builder').KGBuilder)
    kg.queries = []
    kg.graph = SimpleNamespace(query=lambda query, params=None: kg.queries.append((query, params)))
    kg.query_prometheus = lambda query: []
    kg.is_container_running = lambda name: None
    return kg


def test_missing_telemetry_and_docker_do_not_mean_health():
    kg = builder()
    assert kg.update_node_metrics('catalogue') is False
    query, params = kg.queries[0]
    assert params['cpu'] is None and params['mem'] is None and params['err'] is None
    assert params['unknown'] and params['availability'] == 'NO_DATA'
    assert "WHEN $unknown THEN 'unknown'" in query


def test_failed_scrape_does_not_prove_stopped_container():
    kg = builder()
    kg.is_container_running = lambda name: True
    kg.query_prometheus = lambda query: [{'value': [1791280800, '0']}] if query.startswith('up{') else []
    kg.update_node_metrics('orders')
    params = kg.queries[0][1]
    assert params['anomaly'] is False and params['prom_up'] is False and params['unknown'] is True


def test_confirmed_stop_and_exact_queries():
    kg = builder()
    kg.is_container_running = lambda name: False
    queries = []
    def response(query):
        queries.append(query)
        return []
    kg.query_prometheus = response
    assert kg.update_node_metrics('catalogue') is True
    assert kg.queries[0][1]['anomaly_type'] == 'service_crash'
    assert all('instance="catalogue:80"' in query for query in queries)
    assert not any('instance=~' in query for query in queries)


def test_startup_preserves_graph_and_propagation_has_no_fake_confidence():
    kg = builder()
    kg.build_initial_graph()
    assert not any('DELETE' in query for query, _ in kg.queries)
    node_query = kg.queries[0][0]
    assert node_query.index('ON CREATE SET') < node_query.index('SET n.role')
    assert 'n.anomaly_since = null' in node_query
    assert any('ON CREATE SET r.avg_latency' in query for query,_ in kg.queries)
    kg.create_triggers_edge('catalogue-db', 'catalogue')
    assert 'r.confidence = null' in kg.queries[-1][0]
    assert 'rule_propagation_hypothesis' in kg.queries[-1][0]


def test_metric_source_units_sample_time_are_kept():
    kg = builder()
    kg.is_container_running = lambda name: True
    kg.query_prometheus = lambda query: [{'value': [1791280800, '0.2']}]
    kg.update_node_metrics('payment')
    receipts = json.loads(kg.queries[0][1]['receipts'])
    assert receipts['cpu']['sample_time'] == 1791280800
    assert receipts['errors']['units'] == '5xx_events_per_second'


def test_only_one_graph_writer(tmp_path):
    lease = module('graph_writer').writer_lease
    path = tmp_path / 'writer.lock'
    with lease(path):
        with pytest.raises(RuntimeError):
            with lease(path):
                pass
    assert not path.exists()
