from dataclasses import FrozenInstanceError
import pytest
from rca_investigation.contracts import *


def test_topology_serialization_and_membership():
    graph = TopologySnapshot("v1", ("catalogue", "catalogue-db"), ("Service", "Database"), (("catalogue", "catalogue-db"),))
    assert graph == TopologySnapshot.from_json(graph.to_json())
    with pytest.raises(FrozenInstanceError):
        graph.version = "v2"
    with pytest.raises(ValueError):
        TopologySnapshot("v1", ("a",), ("Service",), (("a", "missing"),))
    with pytest.raises(ValueError):
        TopologySnapshot.from_dict({**graph.to_dict(), "label": "orders"})


def test_packet_boundaries():
    args = dict(action_id="status/orders/v1", resource_id="orders", availability="OK", response_code="STOPPED",
                window_start="2026-10-06T10:00:00Z", window_end="2026-10-06T10:01:00Z",
                sample_time="2026-10-06T10:01:00Z", source="docker-inspect", duration_ms=180)
    packet = ToolResult(**args)
    assert packet == ToolResult.from_json(packet.to_json())
    assert len(packet.evidence_id) == 64
    for changed in ({"availability": "TIMEOUT"}, {"duration_ms": True}, {"sample_time": "2026-10-06T09:00:00Z"}):
        with pytest.raises(ValueError):
            ToolResult(**{**args, **changed})
