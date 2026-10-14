"""One graph writer per local lab. A stale lease requires operator review."""
from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def writer_lease(path=None):
    target = Path(path) if path is not None else Path(__file__).resolve().parents[1] / 'data/graph_writer.lock'
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError(f'Graph writer lease exists at {target}. Stop the other writer or inspect the stale lease before removing it.') from error
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump({'pid': os.getpid(), 'project': 'rca-lab'}, stream)
        yield
    finally:
        target.unlink()
