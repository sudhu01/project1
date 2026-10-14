from pathlib import Path
import pytest
import yaml
from rca_investigation.topology import sockshop_topology
from rca_investigation.config import load_config
from rca_investigation.catalogue import *

CONFIG = Path(__file__).resolve().parents[1] / 'configs/investigation_v1.yaml'


def test_static_task_and_prices():
    topology = sockshop_topology()
    assert len(topology.resources) == 13
    config = load_config(CONFIG, topology)
    specs = build_catalogue(config, topology)
    assert len(specs) == 5
    assert action_order(specs)[-2:] == ('STOP', 'DEFER')
    assert catalogue_hash(specs) != catalogue_hash(tuple(reversed(specs)))
    capabilities = {r: ('status',) for r in topology.resources}
    assert len(mask(specs, (), 6, 4, capabilities)) == 7
    assert mask(specs, (), 0, 4, capabilities) == ('STOP', 'DEFER')
    assert mask(specs, (), 6, 4, {}, terminated=True) == ()


def test_unknown_config_fails(tmp_path):
    values = yaml.safe_load(CONFIG.read_text())
    values['tools']['status']['credit'] = 2
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError):
        load_config(path, sockshop_topology())
