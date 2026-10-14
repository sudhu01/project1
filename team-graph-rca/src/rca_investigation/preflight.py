"""Read-only host checks. No stack startup or fault injection."""
from pathlib import Path
import json
import shutil
import subprocess


def run_preflight(project_root, which=shutil.which, runner=subprocess.run):
    root = Path(project_root)
    docker = which('docker')
    if docker is None:
        candidate = Path('C:/Program Files/Docker/Docker/resources/bin/docker.exe')
        if candidate.is_file():
            docker = str(candidate)
    report = {'profile': 'sockshop_v1', 'project': 'rca-lab', 'passed': False, 'checks': {},
              'workload_verified': False, 'captures_accepted': 0}
    free = shutil.disk_usage(root).free
    report['checks']['disk_space'] = {'passed': free >= 10 * 1024**3, 'free_bytes': free,
                                      'minimum_free_bytes': 10 * 1024**3}
    if docker is None:
        report['blocker'] = 'Docker executable unavailable. Install or locate Docker Desktop and enable its Linux engine.'
        report['checks']['docker_executable'] = {'passed': False}
        return report
    report['checks']['docker_executable'] = {'passed': True, 'path': docker}
    commands = {
        'docker_version': [docker, 'version', '--format', '{{json .}}'],
        'compose_version': [docker, 'compose', 'version', '--short'],
        'linux_engine': [docker, 'info', '--format', '{{json .}}'],
        'compose_config': [docker, 'compose', '-p', 'rca-lab', 'config', '--quiet'],
    }
    for name, argv in commands.items():
        try:
            result = runner(argv, cwd=root, capture_output=True, text=True, timeout=15, shell=False)
            passed = result.returncode == 0
            evidence = result.stdout[:8192]
            if name == 'linux_engine' and passed:
                info = json.loads(result.stdout)
                passed = info.get('OSType') == 'linux'
                evidence = {'os_type': info.get('OSType'), 'cpus': info.get('NCPU'), 'memory_bytes': info.get('MemTotal')}
            report['checks'][name] = {'passed': passed, 'evidence': evidence}
        except (OSError, subprocess.TimeoutExpired, ValueError):
            report['checks'][name] = {'passed': False, 'reason': 'command_unavailable_or_failed'}
    report['passed'] = all(v['passed'] for v in report['checks'].values())
    if not report['passed']:
        report['blocker'] = 'Docker/Compose preflight failed. Check the daemon, Linux engine and Compose configuration.'
    else:
        report['next_gate'] = 'Verify stack readiness, image digests, log ingestion and catalogue/orders workload paths.'
    return report
