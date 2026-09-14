"""Gymnasium compatibility checks required by plan step 8.5."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from rca_sim.environment import InvestigationEnv


def test_checker_accepts_ten_service_all_actions_valid_environment() -> None:
    environment = InvestigationEnv(n_services=10, initial_budget=8)
    observation, _ = environment.reset(seed=850)

    assert observation["action_mask"].shape == (41,)
    assert observation["action_mask"].all()
    check_env(environment, skip_render_check=True)


def test_default_environment_supports_mask_aware_multi_step_rollout() -> None:
    environment = InvestigationEnv()
    action_rng = np.random.default_rng(851)
    observation, _ = environment.reset(seed=852)
    terminations = 0

    for transition_index in range(500):
        assert environment.observation_space.contains(observation)
        valid_actions = np.flatnonzero(observation["action_mask"])
        action = int(action_rng.choice(valid_actions))

        observation, _, terminated, truncated, _ = environment.step(action)

        assert environment.observation_space.contains(observation)
        assert not truncated
        if terminated:
            terminations += 1
            if transition_index < 499:
                observation, _ = environment.reset(seed=853 + transition_index)

    assert terminations > 0
