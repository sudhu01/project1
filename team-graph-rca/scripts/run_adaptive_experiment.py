"""Run a declared physical profile through frozen fitting and final comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml
from rca_investigation.dataset import atomic_json
from rca_investigation.train import runtime_hash
from rca_investigation.provenance import freeze_runtime_source

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--schedule', type=Path)
parser.add_argument('--wait-for-capture', action='store_true')
parser.add_argument('--reserved-test', action='store_true')
parser.add_argument('--sweeps', action='store_true')
parser.add_argument('--allow-ambiguous-evidence', action='store_true')
args = parser.parse_args()
value = yaml.safe_load(args.config.read_text())
paths = value['paths']
run = Path(paths['runs'])
state_path = run / 'capture/schedule.json'
report_path = run / 'phase-progress.json'
progress = {'passed': False, 'phase': 'capture', 'config': str(args.config), 'completed': []}

def command(name, *options):
    print(f'Running {name}', flush=True)
    if name == 'capture-raw' and value['task_id'] in {'sockshop_adaptive_v2b', 'sockshop_adaptive_v2c'}:
        result = subprocess.run([sys.executable, 'scripts/capture_adaptive_with_recovery.py',
            '--profile', value['task_id'].removeprefix('sockshop_adaptive_'),
            '--config', str(args.config), '--schedule', str(args.schedule)], check=False)
    else:
        result = subprocess.run([sys.executable, '-m', 'rca_investigation.cli', name,
                                 '--config', str(args.config), *map(str, options)], check=False)
    if result.returncode:
        raise RuntimeError(f'{name} failed; its report contains the reason')
    progress['completed'].append(name)
    atomic_json(report_path, progress)

try:
    if (Path(paths['raw_dataset']) / 'splits.json').exists():
        pass
    elif args.wait_for_capture:
        previous = None
        while True:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            count = len(state.get('accepted', {}))
            if count != previous:
                print(f'Accepted capture slots: {count}', flush=True)
                previous = count
            if state.get('complete'):
                break
            failure_path = run / 'capture-evaluation.json'
            if failure_path.exists() and not json.loads(failure_path.read_text()).get('passed'):
                raise RuntimeError('capture failed; preserve accepted slots and fix the reported cause before resume')
            time.sleep(10)
    elif not state_path.exists() or not json.loads(state_path.read_text()).get('complete'):
        if args.schedule is None:
            raise ValueError('declare a schedule before physical collection')
        command('verify-profile')
        command('capture-raw', '--schedule', args.schedule, '--output', run / 'capture-evaluation.json')
    progress['phase'] = 'fit'
    source_hash = runtime_hash()
    progress['source_hash'] = source_hash
    snapshot = freeze_runtime_source(run / 'source-snapshot.json')
    if snapshot['runtime_hash'] != source_hash:
        raise RuntimeError('source changed while its experiment snapshot was saved')
    progress['source_snapshot_hash'] = snapshot['hash']
    if not (Path(paths['raw_dataset']) / 'splits.json').exists():
        command('split-raw', '--dataset', paths['raw_dataset'], '--schedule-state', state_path,
                '--output', run / 'split-raw.json')
    if not Path(paths['encoder']).exists():
        command('fit-encoder', '--dataset', paths['raw_dataset'], '--split', 'train')
    if not (Path(paths['dataset']) / 'splits.json').exists():
        command('encode-dataset')
    if not Path(paths['model']).exists():
        command('fit-model')
    command('full-evidence-reference', '--split', 'validation')
    reference = json.loads((run / 'full-evidence-reference.json').read_text())
    progress['full_evidence_reference_passed'] = reference['training_promotion_passed']
    if not reference['training_promotion_passed'] and not args.allow_ambiguous_evidence:
        raise RuntimeError('full evidence misses the validation target; improve measurements before policy training')
    sweep_manifests = []
    if args.sweeps:
        from rca_investigation.config import load_config
        from rca_investigation.topology import sockshop_topology
        parent = load_config(args.config, sockshop_topology())
        # Declare the entire family before any reserved comparison opens.
        for operation, folder in (('declare-ablations', 'ablations'), ('declare-cost-sweep', 'costs')):
            directory = Path('configs') / (value['task_id'] + '-' + folder + '-' + parent.hash[:12])
            if not (directory / 'manifest.json').exists():
                command(operation, '--output', directory)
            declared = json.loads((directory / 'manifest.json').read_text())
            if declared['parent_config_hash'] != parent.hash:
                raise ValueError('sweep declaration belongs to a different parent experiment')
            sweep_manifests.append(declared)
    progress['phase'] = 'train'
    for method in ('expected-sarsa', 'two-step-sarsa'):
        for seed in value['training']['seeds']:
            target = run / 'training' / method / str(seed)
            options = ['--policy', method, '--episodes', str(value['training']['episodes']),
                       '--seeds', str(seed), '--output', target / 'report.json']
            if (target / 'latest.json').exists():
                options += ['--resume']
            if runtime_hash() != source_hash:
                raise RuntimeError('source changed during training; use a new experiment identity')
            command('train', *options)
    progress['phase'] = 'validation'
    frozen = run / 'final.json'
    if not frozen.exists():
        command('compare', '--split', 'validation')
    if args.reserved_test:
        progress['phase'] = 'reserved_test'
        if not (run / 'compare-test.json').exists() or not json.loads((run / 'compare-test.json').read_text())['passed']:
            command('compare', '--split', 'test')
        command('challenge')
    if args.sweeps:
        for manifest in sweep_manifests:
            for variant in manifest['variants']:
                # Every subset keeps all roots and gets its own model and checkpoints.
                # Weak subsets still get trained checkpoints, with their failed
                # diagnostic gate reported and the complete root scope retained.
                result = subprocess.run([sys.executable, __file__, '--config', variant['config'],
                    '--wait-for-capture', '--allow-ambiguous-evidence',
                    *(['--reserved-test'] if args.reserved_test else [])], check=False)
                progress.setdefault('sweep_results', []).append({'config': variant['config'], 'passed': result.returncode == 0})
    if any(not result['passed'] for result in progress.get('sweep_results', [])):
        raise RuntimeError('one or more declared sweep runs failed; inspect their phase reports')
    progress.update(passed=True, phase='complete', physical_policy_results=bool(args.reserved_test),
                    training_promotion_eligible=progress['full_evidence_reference_passed'])
except (ValueError, RuntimeError, OSError) as exc:
    progress.update(error=str(exc))
finally:
    atomic_json(report_path, progress)
print(json.dumps(progress, indent=2), flush=True)
raise SystemExit(0 if progress['passed'] else 1)
