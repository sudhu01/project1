from rca_investigation.adaptive_fixtures import validate_adaptive_fixtures


def test_three_probe_requires_longer_lookahead():
    assert validate_adaptive_fixtures()['passed']
