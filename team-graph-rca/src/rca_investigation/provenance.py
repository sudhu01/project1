"""Preserve the package source behind an experiment's runtime identity."""
import json
from pathlib import Path
from .contracts import digest
from .dataset import atomic_json


def freeze_runtime_source(destination, source_root=None):
    root = Path(source_root) if source_root is not None else Path(__file__).parent
    files = {str(path.relative_to(root)): path.read_text(encoding='utf-8')
             for path in sorted(root.rglob('*.py'))}
    if not files:
        raise ValueError('runtime source snapshot requires Python source')
    payload = {'schema_version': 1, 'files': files,
               'runtime_hash': digest({name: digest(source) for name, source in files.items()})}
    payload['hash'] = digest(payload)
    destination = Path(destination)
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != payload:
            raise ValueError('frozen runtime source differs; use a new experiment path')
    else:
        atomic_json(destination, payload)
    return payload
