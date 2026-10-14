"""Declare independent roles and control mixtures before collecting any cohort."""
from pathlib import Path
import yaml

for stage in ('v2a', 'v2b', 'v2c'):
    sessions = []
    for role, count in (('train', 20), ('validation', 10), ('test', 10)):
        for index in range(count):
            scenarios = ['catalogue_db_stop', 'orders_stop', 'payment_cpu']
            if stage != 'v2a':
                scenarios += ['payment_path_delay']
            if stage == 'v2c':
                scenarios += ['carts_memory']
            controls = ['healthy', 'healthy_higher_load'] + (['healthy_allocation_release'] if stage == 'v2c' else [])
            scenarios += [controls[index % len(controls)]]
            # Rotate the order to avoid associating cause with session position.
            shift = index % len(scenarios)
            scenarios = scenarios[shift:] + scenarios[:shift]
            sessions.append({'id': f'{stage}-{role}-{index + 1:02}', 'role': role, 'scenarios': scenarios})
    target = Path(f'configs/capture_{stage}_evaluation.yaml')
    if target.exists():
        raise ValueError('evaluation schedule already declared')
    target.write_text(yaml.safe_dump({'schema_version': 2, 'task_id': 'sockshop_adaptive_' + stage,
        'lab_batch': stage + '-evaluation-protocol120', 'protocol': {'baseline_seconds': 120,
        'settling_seconds': 30, 'observation_seconds': 120, 'ingestion_seconds': 10, 'concurrency': 1},
        'sessions': sessions}, sort_keys=False), encoding='utf-8')
