import pytest
from scripts.revise_v2c_memory import fit_boundary


def cohort():
    return [dict(incident_id=f'{cause}-{i}', cause=cause, session=f'session-{i}',
                 fraction=fraction, effect_confirmed=True, recovery_confirmed=True)
            for cause, fraction in [('NO_FAULT', .58), ('resource/carts', .83)]
            for i in range(5)]


def test_training_gap_allows_normal_drift_but_retains_fault_detection():
    rows = cohort()
    threshold = fit_boundary(rows, [r['incident_id'] for r in rows])['threshold']
    assert .582 < threshold < .83
    assert threshold == pytest.approx(.705)


def test_rejects_additional_validation_incident():
    rows = cohort()
    ids = [r['incident_id'] for r in rows]
    rows.append({**rows[0], 'incident_id': 'validation-incident'})
    with pytest.raises(ValueError, match='training cohort'):
        fit_boundary(rows, ids)


def test_overlapping_groups_fail_instead_of_tuning_to_validation():
    rows = cohort()
    rows[0]['fraction'] = .84
    with pytest.raises(ValueError, match='overlap'):
        fit_boundary(rows, [r['incident_id'] for r in rows])


def test_unverified_fault_is_not_a_fitting_example():
    rows = cohort()
    rows[-1]['effect_confirmed'] = False
    with pytest.raises(ValueError, match='verified'):
        fit_boundary(rows, [r['incident_id'] for r in rows])
