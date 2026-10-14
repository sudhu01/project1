import json
from pathlib import Path
from types import SimpleNamespace
from rca_investigation.cli import main
from rca_investigation.preflight import run_preflight


def test_actual_offline_commands(tmp_path):
    path = tmp_path/'fixtures.json'
    assert main(['validate-fixtures','--output',str(path)]) == 0
    report = json.loads(path.read_text())
    assert report['synthetic_only'] and report['manifest']['source_hashes']
    config = Path(__file__).resolve().parents[1]/'configs/investigation_v1.yaml'
    assert main(['validate-config','--config',str(config),'--output',str(tmp_path/'config.json')]) == 0


def test_preflight_rejects_windows_engine(tmp_path):
    def runner(argv, **kwargs):
        assert kwargs['shell'] is False and kwargs['timeout'] == 15
        return SimpleNamespace(returncode=0, stdout=json.dumps({'OSType':'windows'}) if argv[1]=='info' else '')
    report = run_preflight(tmp_path, which=lambda _: 'docker', runner=runner)
    assert not report['passed'] and not report['checks']['linux_engine']['passed']
    assert report['captures_accepted'] == 0


def test_capture_timeout_writes_terminal_failure_report(tmp_path, monkeypatch):
    import subprocess
    from rca_investigation import raw_schedule
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(['docker', 'inspect'], 15)
    monkeypatch.setattr(raw_schedule, 'capture_raw', timeout)
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'capture-failure.json'
    assert main(['capture-raw', '--config', str(root / 'configs/investigation_v2a.yaml'),
                 '--schedule', str(root / 'configs/capture_v2a_evaluation.yaml'),
                 '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert report['passed'] is False and report['error_type'] == 'TimeoutExpired'
