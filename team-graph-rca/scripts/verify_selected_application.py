"""Check selected CLI/application behavior on one validation capture per class."""
import json
from pathlib import Path
import subprocess
import sys

from rca_investigation.dataset import atomic_json, load_private_label, read_split
from rca_investigation.contracts import ToolResult
from rca_investigation.experiment import label_name


def main():
    root = Path(__file__).resolve().parents[1]
    dataset = root / 'datasets/sockshop_v1'
    _, paths = read_split(dataset, 'validation')
    representatives = {}
    for path in paths:
        representatives.setdefault(label_name(load_private_label(path)), path)
    if set(representatives) != {'catalogue-db', 'orders', 'NO_FAULT'}:
        raise ValueError('validation captures must cover all supported cases')
    destination = root / 'runs/integration_v1/application'
    destination.mkdir(parents=True, exist_ok=True)
    checks = []
    for label, path in sorted(representatives.items()):
        reports = []
        arguments = ['investigate', '--backend', 'replay', '--incident', str(path),
                     '--policy', 'selected', '--dataset', str(dataset)]
        for name, prefix in (
            ('cli', [sys.executable, '-c',
                     'from rca_investigation.cli import main; raise SystemExit(main())']),
            ('application', [sys.executable, 'pipeline/investigate.py']),
        ):
            output = destination / f'{label}-{name}.json'
            execution = subprocess.run([*prefix, *arguments, '--output', str(output)],
                                       cwd=root, capture_output=True, text=True, timeout=60)
            if execution.returncode:
                raise RuntimeError(f'{name} failed: {execution.stderr}\n{execution.stdout}')
            report = json.loads(output.read_text(encoding='utf-8'))
            if not report['passed']:
                raise ValueError(f'{name} did not pass')
            reports.append(report)
        cli, application = reports
        if cli['result']['decision'] != application['result']['decision']:
            raise ValueError('CLI/application decision mismatch')
        actions = [step['action'] for step in cli['result']['trace']]
        if actions != [step['action'] for step in application['result']['trace']]:
            raise ValueError('CLI/application semantic trace mismatch')
        for report in reports:
            result, explanation = report['result'], report['explanation']
            packet_ids = [ToolResult.from_dict(packet).evidence_id for packet in result['packets']]
            if explanation['evidence_ids'] != packet_ids:
                raise ValueError('explanation references unacquired evidence')
            if len(explanation['evidence']) != len(packet_ids):
                raise ValueError('explanation evidence count mismatch')
        checks.append({'validation_incident': path.name, 'case': label,
                       'decision': cli['result']['decision'], 'actions': actions,
                       'credits': cli['result']['credits'], 'matching': True})
    report = {'passed': True, 'mode': 'delayed_frozen_window_shadow',
              'additional_physical_incidents': 0, 'checks': checks}
    atomic_json(destination / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
