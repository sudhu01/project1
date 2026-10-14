import json
import pytest
from rca_investigation.contracts import digest
from rca_investigation.provenance import freeze_runtime_source
from rca_investigation.train import runtime_hash


def test_snapshot_preserves_actual_package_runtime_identity(tmp_path):
    snapshot = freeze_runtime_source(tmp_path / 'source.json')
    assert snapshot['runtime_hash'] == runtime_hash()
    assert 'contracts.py' in snapshot['files']
    assert digest({key: value for key, value in snapshot.items() if key != 'hash'}) == snapshot['hash']
    assert freeze_runtime_source(tmp_path / 'source.json') == snapshot


@pytest.mark.parametrize('change', ['edit', 'add', 'delete', 'tamper'])
def test_existing_snapshot_rejects_source_or_archive_changes(tmp_path, change):
    root = tmp_path / 'package'
    root.mkdir()
    source = root / 'model.py'
    source.write_text('smoothing = .1\n')
    archive = tmp_path / 'source.json'
    original = freeze_runtime_source(archive, root)
    original_bytes = archive.read_bytes()
    if change == 'edit':
        source.write_text('smoothing = .2\n')
    elif change == 'add':
        (root / 'other.py').write_text('value = 1\n')
    elif change == 'delete':
        (root / 'other.py').write_text('value = 1\n')
        source.unlink()
    else:
        original['files']['model.py'] = 'smoothing = .2\n'
        archive.write_text(json.dumps(original))
        original_bytes = archive.read_bytes()
    with pytest.raises(ValueError, match='frozen runtime source differs'):
        freeze_runtime_source(archive, root)
    assert archive.read_bytes() == original_bytes
