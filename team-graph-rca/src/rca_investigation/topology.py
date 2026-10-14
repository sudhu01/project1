from importlib.resources import files
from .contracts import TopologySnapshot


def sockshop_topology():
    return TopologySnapshot.from_json(files('rca_investigation').joinpath('sockshop_topology.json').read_text(encoding='utf-8'))
