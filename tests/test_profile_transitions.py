from pathlib import Path

from rca_sim.profile_transitions import profile_transitions


def test_transition_profiler_reports_components() -> None:
    result = profile_transitions(
        config_path=Path("configs/sim_v0.yaml"), transitions=20, seed=12
    )
    assert result["transitions"] == 20
    assert result["transitions_per_second"] > 0
    assert result["world_generation_seconds"] > 0
    assert result["inference_seconds"] >= 0
    assert result["observation_construction_seconds"] > 0
