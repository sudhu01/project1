from rca_investigation.logs.parser import summarize


def test_peer_from_observed_error_and_exact_match():
    assert summarize(['connection refused http://payment:80'], 'orders', ['payment', 'orders-db'])[1] == 'DEPENDENCY_ERROR/payment'
    assert summarize(['connection refused http://payment-db:80'], 'orders', ['payment'])[1] == 'DEPENDENCY_ERROR'
    assert summarize(['payment completed'], 'orders', ['payment'])[1] == 'NO_RELEVANT_ERROR'
    assert summarize(['rca-helper error payment'], 'orders', ['payment'])[1] == 'NO_RELEVANT_ERROR'
    assert summarize([], 'orders', ['payment'])[0:2] == ('OK', 'NO_RELEVANT_ERROR')


def test_bounded_logs_and_ambiguous_peers():
    result = summarize(['connection refused payment and orders-db'] * 30, 'orders', ['payment', 'orders-db'])
    assert result[1] == 'DEPENDENCY_ERROR'
    assert result[2]['truncated'] and result[2]['sample_count'] == 20
    assert summarize(['x' * 10000], 'orders', [], max_bytes=8192)[2]['truncated']
