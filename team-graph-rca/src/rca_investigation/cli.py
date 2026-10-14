import argparse
import json
import hashlib
from importlib.resources import files
from pathlib import Path
import subprocess
from .catalogue import build_catalogue, catalogue_hash, action_order
from .config import load_config
from .fixtures import fixture_environment
from .policies.dp import BoundedDP
from .preflight import run_preflight
from .topology import sockshop_topology


def validate_fixtures():
    definitions = json.loads(files('rca_investigation').joinpath('decision_fixtures.json').read_text(encoding='utf-8'))
    results = {}
    for name, definition in definitions.items():
        env, _ = fixture_environment(definition)
        planner = BoundedDP(env.model, env.specs)
        action = planner.choose(env.observation())
        value = max(planner.stats['values'].values())
        passed = abs(value-definition['expected_depth2']) < 1e-10
        expected = definition['expected_action']
        passed &= action not in ('STOP', 'DEFER') if expected == 'PROBE' else action == expected
        if name == 'complementary':
            passed &= BoundedDP(env.model, env.specs, depth=1).choose(env.observation()) == 'STOP'
        results[name] = {'passed': bool(passed), 'action': action, 'expected_return': value}
        env.close()
    return {'passed': all(v['passed'] for v in results.values()), 'synthetic_only': True, 'checks': results}


def write_report(path, report):
    root = Path.cwd()
    lock = root/'uv.lock'
    try:
        revision = subprocess.run(['git','-c',f'safe.directory={root.as_posix()}','rev-parse','HEAD'],
                                  capture_output=True, text=True, timeout=5, shell=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        revision = 'unavailable'
    source = Path(__file__).parent
    code_hashes = {str(file): hashlib.sha256(file.read_bytes()).hexdigest()
                   for file in source.rglob('*') if file.is_file() and file.suffix in ('.py', '.json')}
    report['manifest'] = {'source_revision': revision, 'source_hashes': code_hashes,
                          'dependency_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
                          'backend_version': 'frozen_replay_v1', 'seeds': [], 'split_hashes': [], 'checkpoint': None}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    temporary.replace(destination)


def main(argv=None):
    import sys
    from .adaptive_cli import COMMANDS, main as adaptive_main
    command_line = sys.argv[1:] if argv is None else argv
    if command_line and command_line[0] in COMMANDS:
        return adaptive_main(command_line)
    parser = argparse.ArgumentParser(description="Bounded RCA investigation tools")
    parser.add_argument("--version", action="store_true")
    commands = parser.add_subparsers(dest='command')
    config_command = commands.add_parser('validate-config')
    config_command.add_argument('--config', type=Path, required=True)
    config_command.add_argument('--output', type=Path, default=Path('runs/integration_v1/config.json'))
    fixtures_command = commands.add_parser('validate-fixtures')
    fixtures_command.add_argument('--output', type=Path, default=Path('runs/integration_v1/fixtures.json'))
    preflight_command = commands.add_parser('preflight')
    preflight_command.add_argument('--profile', choices=['sockshop_v1'], required=True)
    preflight_command.add_argument('--output', type=Path, default=Path('runs/integration_v1/preflight.json'))
    for command in ('pin-images', 'verify-live'):
        live_command = commands.add_parser(command)
        live_command.add_argument('--output', type=Path, default=Path(f'runs/integration_v1/{command}.json'))
    capture_command = commands.add_parser('capture')
    capture_command.add_argument('--profile',choices=['sockshop_v1'],required=True)
    capture_command.add_argument('--schedule',type=Path,required=True)
    capture_command.add_argument('--config',type=Path,default=Path('configs/investigation_v1.yaml'))
    capture_command.add_argument('--output',type=Path,default=Path('runs/integration_v1/capture.json'))
    split_command=commands.add_parser('split')
    split_command.add_argument('--dataset',type=Path,required=True)
    split_command.add_argument('--group',choices=['capture_session'],default='capture_session')
    split_command.add_argument('--schedule-state',type=Path,required=True)
    split_command.add_argument('--output',type=Path,default=Path('runs/integration_v1/split.json'))
    fixture_train=commands.add_parser('train-fixture')
    fixture_train.add_argument('--name',required=True)
    fixture_train.add_argument('--episodes',type=int,default=2500)
    fixture_train.add_argument('--seed',type=int,default=11)
    fixture_train.add_argument('--policy',choices=['expected-sarsa','two-step-sarsa'],default='expected-sarsa')
    fixture_train.add_argument('--run-dir',type=Path,required=True)
    fixture_train.add_argument('--resume',action='store_true')
    fixture_train.add_argument('--stop-after',type=int)
    fixture_train.add_argument('--output',type=Path,default=Path('runs/integration_v1/train-fixture.json'))
    for command in ('fit-model','train','evaluate'):
        sub=commands.add_parser(command)
        sub.add_argument('--config',type=Path,default=Path('configs/investigation_v1.yaml'))
        sub.add_argument('--dataset',type=Path,default=Path('datasets/sockshop_v1'))
        sub.add_argument('--model',type=Path,default=Path('runs/integration_v1/model.json'))
        sub.add_argument('--output',type=Path,default=Path(f'runs/integration_v1/{command}.json'))
        if command=='fit-model':
            sub.add_argument('--split',choices=['train'],default='train')
        elif command=='train':
            sub.add_argument('--policy',choices=['expected-sarsa','two-step-sarsa'],default='expected-sarsa')
            sub.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
            sub.add_argument('--run-dir',type=Path,default=Path('runs/integration_v1/training'))
            sub.add_argument('--episodes',type=int,default=10000)
            sub.add_argument('--resume',action='store_true')
            sub.add_argument('--stop-after',type=int)
        else:
            sub.add_argument('--split',choices=['train','validation'],default='validation')
            sub.add_argument('--policy',choices=['stop-immediately','defer-immediately','random-valid','dependency-rule','dp','voi'],required=True)
            sub.add_argument('--depth',type=int,default=2)
            sub.add_argument('--seed',type=int,default=11)
    challenge=commands.add_parser('challenge')
    challenge.add_argument('--config',type=Path,default=Path('configs/investigation_v1.yaml'))
    challenge.add_argument('--dataset',type=Path,default=Path('datasets/sockshop_v1'))
    challenge.add_argument('--model',type=Path,default=Path('runs/integration_v1/model.json'))
    challenge.add_argument('--frozen-manifest',type=Path,default=Path('experiments/local/final_v1.json'))
    challenge.add_argument('--test-report',type=Path,default=Path('runs/integration_v1/compare-test.json'))
    challenge.add_argument('--output',type=Path,default=Path('runs/integration_v1/challenges.json'))
    compare=commands.add_parser('compare')
    compare.add_argument('--split',choices=['validation','test'],required=True)
    compare.add_argument('--config',type=Path,default=Path('configs/investigation_v1.yaml'))
    compare.add_argument('--dataset',type=Path,default=Path('datasets/sockshop_v1'))
    compare.add_argument('--model',type=Path,default=Path('runs/integration_v1/model.json'))
    compare.add_argument('--training-root',type=Path,default=Path('runs/integration_v1/training'))
    compare.add_argument('--frozen-manifest',type=Path,default=Path('experiments/local/final_v1.json'))
    compare.add_argument('--output',type=Path)
    investigation=commands.add_parser('investigate')
    investigation.add_argument('--backend',choices=['replay'],required=True)
    investigation.add_argument('--incident',type=Path,required=True)
    investigation.add_argument('--policy',choices=['selected','dependency-rule','dp'],default='selected')
    investigation.add_argument('--config',type=Path,default=Path('configs/investigation_v1.yaml'))
    investigation.add_argument('--dataset',type=Path,default=Path('datasets/sockshop_v1'))
    investigation.add_argument('--model',type=Path,default=Path('runs/integration_v1/model.json'))
    investigation.add_argument('--frozen-manifest',type=Path,default=Path('experiments/local/final_v1.json'))
    investigation.add_argument('--output',type=Path,default=Path('runs/integration_v1/investigation.json'))
    args = parser.parse_args(argv)
    if args.command=='compare' and args.output is None:
        args.output=Path(f'runs/integration_v1/compare-{args.split}.json')
    if args.version:
        print("rca-investigation 0.1.0")
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if hasattr(args, 'config'):
            import yaml
            raw_settings = yaml.safe_load(args.config.read_text(encoding='utf-8'))
            if type(raw_settings) is dict and raw_settings.get('schema_version') == 2 and not any(
                    arg == '--output' or arg.startswith('--output=') for arg in command_line):
                args.output = Path('runs/adaptive_v2') / (args.command + '.json')
            resolved = load_config(args.config, sockshop_topology())
            if resolved.values['schema_version'] == 2:
                paths = resolved.values['paths']
                replacements = {'dataset': paths['dataset'], 'model': paths['model'],
                    'run_dir': paths['runs'] + '/training', 'training_root': paths['runs'] + '/training',
                    'frozen_manifest': paths['runs'] + '/final.json',
                    'test_report': paths['runs'] + '/compare-test.json',
                    'output': paths['runs'] + '/' + args.command + (('-' + args.split) if args.command == 'compare' else '') + '.json'}
                for field, replacement in replacements.items():
                    flag = '--' + field.replace('_', '-')
                    if hasattr(args, field) and not any(arg == flag or arg.startswith(flag + '=') for arg in command_line):
                        setattr(args, field, Path(replacement))
                if args.command == 'capture':
                    raise ValueError('v2 requires capture-raw; the v1 capture controller cannot collect adaptive evidence')
                if args.command in {'fit-model', 'train', 'evaluate', 'compare', 'investigate', 'challenge'} and resolved.values['encoding_hash'] == '0' * 64:
                    raise ValueError('adaptive evidence encoder is not fitted; collect and split raw training incidents first')
        if args.command == 'validate-config':
            topology = sockshop_topology()
            config = load_config(args.config, topology)
            specs = build_catalogue(config, topology)
            report = {'passed': True, 'resolved_config': config.values, 'config_hash': config.hash,
                      'topology_hash': topology.hash, 'catalogue_hash': catalogue_hash(specs), 'action_order': action_order(specs)}
        elif args.command == 'validate-fixtures':
            report = validate_fixtures()
        elif args.command == 'preflight':
            report = run_preflight(Path.cwd())
        elif args.command == 'pin-images':
            from .live_validation import pin_images
            report = {'passed':True, 'images':pin_images(Path.cwd())}
        elif args.command == 'capture':
            from .capture import run_schedule
            report = run_schedule(Path.cwd(),args.schedule,args.config)
        elif args.command == 'split':
            from .dataset import freeze_scheduled_splits
            report={'passed':True,'split':freeze_scheduled_splits(args.dataset,args.schedule_state)}
        elif args.command == 'train-fixture':
            from .train import train_fixture
            report=train_fixture(args.name,args.run_dir,seed=args.seed,episodes=args.episodes,method=args.policy,
                                 resume=args.resume,stop_after=args.stop_after)
        elif args.command in ('fit-model','train','evaluate'):
            from .dataset import read_split
            from .experiment import fit_dataset,load_model
            config=load_config(args.config,sockshop_topology())
            specs=build_catalogue(config,sockshop_topology())
            manifest,_=read_split(args.dataset,'train')
            if args.command=='fit-model':
                model=fit_dataset(args.dataset,config,specs,args.model)
                report={'passed':True,'model_hash':model.hash,'training_records':len(model.labels),'split_hash':manifest['hash']}
            else:
                model=load_model(args.model,config,specs,manifest['hash'])
                if args.command=='train':
                    from .train import train_dataset
                    if args.episodes!=config.values['training']['episodes'] or len(set(args.seeds))!=len(args.seeds):
                        raise ValueError('episode count must match resolved config; seeds must be unique')
                    reports=[train_dataset(args.dataset,specs,model,config,args.run_dir/args.policy/str(seed),
                        seed=seed,method=args.policy,resume=args.resume,stop_after=args.stop_after) for seed in args.seeds]
                    report={'passed':all(r['passed'] for r in reports),'runs':reports}
                else:
                    from .evaluate_decisions import evaluate_paths
                    from .policies.rules import RulePolicy
                    objective = config.values['objective']
                    policy=BoundedDP(model,specs,depth=1 if args.policy=='voi' else args.depth,
                        defer=objective['defer'],correct=objective['correct'],incorrect=objective['incorrect']) if args.policy in ('dp','voi') else RulePolicy(args.policy,args.seed,objective)
                    _,paths=read_split(args.dataset,args.split)
                    summary,outcomes=evaluate_paths(paths,specs,model,config,policy)
                    report={'passed':True,'policy':args.policy,'split':args.split,'summary':summary,'outcomes':outcomes}
        elif args.command in ('compare','investigate','challenge'):
            from .dataset import read_split
            from .experiment import load_model,replay_environment
            from .comparison import compare_validation,compare_test,load_frozen,policy_from_definition,policy_identity
            config=load_config(args.config,sockshop_topology())
            specs=build_catalogue(config,sockshop_topology())
            manifest,_=read_split(args.dataset,'train')
            model=load_model(args.model,config,specs,manifest['hash'])
            if args.command=='challenge':
                from .challenges import challenge_report
                completed=json.loads(args.test_report.read_text(encoding='utf-8')) if args.test_report.exists() else {}
                frozen=load_frozen(args.frozen_manifest,args.dataset,config,model)
                if not completed.get('passed') or completed.get('frozen_hash')!=frozen['hash']:
                    raise ValueError('complete the frozen main test comparison before challenge analysis')
                report=challenge_report(args.dataset,specs,model,config,args.frozen_manifest)
                write_report(args.output,report)
                print(json.dumps({'passed':True,'synthetic_perturbations_only':True,'output':str(args.output)}))
                return 0
            if args.command=='compare':
                report=compare_validation(args.dataset,specs,model,config,args.training_root,args.output,args.frozen_manifest) if args.split=='validation' else \
                    compare_test(args.dataset,specs,model,config,args.frozen_manifest,args.output)
                print(json.dumps({'passed':report['passed'],'selected':report['selected'],'split':args.split,'output':str(args.output)}))
                return 0 if report['passed'] else 1
            from .evaluate_decisions import investigate
            from .explain import explain_result
            definition={'kind':'rule','name':'dependency-rule'} if args.policy=='dependency-rule' else {'kind':'dp','depth':2}
            if args.policy=='selected':
                frozen=load_frozen(args.frozen_manifest,args.dataset,config,model)
                definition=frozen['definitions'][frozen['selected']]
            policy=policy_from_definition(definition,model,specs,config)
            env=replay_environment(args.incident,specs,model,config)
            try:
                result=investigate(env,policy,policy_identity(definition,model,config))
                report={'passed':True,'mode':'delayed_frozen_window_shadow','result':result,'explanation':explain_result(result)}
            finally: env.close()
        else:
            from .live_validation import verify_live
            report = verify_live(Path.cwd())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        report = {'passed': False, 'error': str(error), 'error_type': type(error).__name__}
        # The comparison owns its durable partial/completed report. An error
        # must not replace it and discard completed reserved-test decisions.
        if args.command == 'compare' and args.output.exists():
            args.output = args.output.with_suffix(args.output.suffix + '.failure.json')
    write_report(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'manifest'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
