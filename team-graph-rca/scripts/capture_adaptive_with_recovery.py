"""Retry leased physical captures only after verified owned lab recovery."""
import argparse
from pathlib import Path
import subprocess
import sys

from resume_v2a_legacy import capture_with_recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('v2b', 'v2c'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--schedule', required=True)
    args = parser.parse_args()
    root = Path.cwd()

    def capture(output):
        return subprocess.run([sys.executable, '-m', 'rca_investigation.cli',
            'capture-raw', '--config', args.config, '--schedule', args.schedule,
            '--output', str(output)], check=False).returncode

    def recover():
        subprocess.run([sys.executable, 'scripts/recover_adaptive_lab.py'],
                       check=True, timeout=150)

    return capture_with_recovery(root, capture, recover, profile=args.profile)


if __name__ == '__main__':
    raise SystemExit(main())
