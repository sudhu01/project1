"""Step 10 tests for the shared policy and invalid-action masking."""

from __future__ import annotations

import torch

from rca_sim.model import SmallPolicyNetwork, _masked_pool
from rca_sim.observation import OBSERVATION_SHAPES
from rca_sim.tools import (
    ACTION_SPACE_SIZE,
    MAX_SERVICE_SLOTS,
    STOP_ACTION_INDEX,
    TEMPLATES_PER_SERVICE,
)


SEED = 20260915


def _observation(active_services: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED + active_services)
    node_features = torch.rand(
        (1, *OBSERVATION_SHAPES["node_features"]),
        generator=generator,
    )
    node_features[:, active_services:] = 0.0
    node_mask = torch.zeros((1, MAX_SERVICE_SLOTS), dtype=torch.bool)
    node_mask[:, :active_services] = True
    adjacency = torch.zeros((1, *OBSERVATION_SHAPES["adjacency"]))
    adjacency[:, :active_services, :active_services] = torch.randint(
        0,
        2,
        (1, active_services, active_services),
        generator=generator,
    )
    global_features = torch.rand(
        (1, *OBSERVATION_SHAPES["global_features"]),
        generator=generator,
    )
    action_mask = torch.zeros((1, ACTION_SPACE_SIZE), dtype=torch.bool)
    action_mask[:, : active_services * TEMPLATES_PER_SERVICE] = True
    action_mask[:, STOP_ACTION_INDEX] = True
    return {
        "node_features": node_features,
        "node_mask": node_mask,
        "adjacency": adjacency,
        "global_features": global_features,
        "action_mask": action_mask,
    }


def _batch(*observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: torch.cat([observation[name] for observation in observations], dim=0)
        for name in observations[0]
    }


def test_policy_output_shapes() -> None:
    torch.manual_seed(SEED)
    policy = SmallPolicyNetwork()
    observation = _batch(_observation(8), _observation(5), _observation(1))

    logits, values = policy(observation)

    assert logits.shape == (3, ACTION_SPACE_SIZE)
    assert values.shape == (3,)


def test_invalid_actions_have_zero_probability_and_cannot_be_sampled() -> None:
    torch.manual_seed(SEED)
    policy = SmallPolicyNetwork()
    observation = _observation(3)
    observation["action_mask"][:] = False
    valid_actions = torch.tensor([0, 5, STOP_ACTION_INDEX])
    observation["action_mask"][:, valid_actions] = True

    distribution, _ = policy.distribution(observation)
    invalid = ~observation["action_mask"]
    samples = distribution.sample((50_000,))

    assert torch.count_nonzero(distribution.probs[invalid]) == 0
    assert bool(observation["action_mask"][0, samples].all())


def test_active_action_probabilities_sum_to_one() -> None:
    torch.manual_seed(SEED)
    distribution, _ = SmallPolicyNetwork().distribution(_observation(6))

    torch.testing.assert_close(distribution.probs.sum(dim=-1), torch.ones(1))
    assert torch.isfinite(distribution.entropy()).all()


def test_reordering_service_slots_reorders_candidate_probabilities() -> None:
    torch.manual_seed(SEED)
    policy = SmallPolicyNetwork().eval()
    original = _observation(4)
    reordered = {name: value.clone() for name, value in original.items()}
    permutation = torch.tensor([2, 0, 3, 1])
    reordered["node_features"][:, :4] = original["node_features"][:, permutation]
    reordered["adjacency"][:, :4, :4] = original["adjacency"][:, permutation][
        :, :, permutation
    ]
    original_mask = original["action_mask"][:, :-1].reshape(
        1, MAX_SERVICE_SLOTS, TEMPLATES_PER_SERVICE
    )
    reordered_mask = reordered["action_mask"][:, :-1].reshape(
        1, MAX_SERVICE_SLOTS, TEMPLATES_PER_SERVICE
    )
    reordered_mask[:, :4] = original_mask[:, permutation]

    original_distribution, original_values = policy.distribution(original)
    reordered_distribution, reordered_values = policy.distribution(reordered)
    original_candidates = original_distribution.probs[:, :-1].reshape(
        1, MAX_SERVICE_SLOTS, TEMPLATES_PER_SERVICE
    )
    reordered_candidates = reordered_distribution.probs[:, :-1].reshape(
        1, MAX_SERVICE_SLOTS, TEMPLATES_PER_SERVICE
    )

    torch.testing.assert_close(
        reordered_candidates[:, :4],
        original_candidates[:, permutation],
        atol=1e-7,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        reordered_distribution.probs[:, STOP_ACTION_INDEX],
        original_distribution.probs[:, STOP_ACTION_INDEX],
    )
    torch.testing.assert_close(reordered_values, original_values)


def test_changing_only_padding_does_not_change_valid_outputs() -> None:
    torch.manual_seed(SEED)
    policy = SmallPolicyNetwork().eval()
    original = _observation(4)
    changed = {name: value.clone() for name, value in original.items()}
    generator = torch.Generator().manual_seed(SEED + 99)
    changed["node_features"][:, 4:] = (
        torch.rand((1, 6, 42), generator=generator) * 100.0 - 50.0
    )
    changed["adjacency"][:, 4:, :] = 1.0
    changed["adjacency"][:, :, 4:] = 1.0

    original_distribution, original_values = policy.distribution(original)
    changed_distribution, changed_values = policy.distribution(changed)
    valid = original["action_mask"]

    torch.testing.assert_close(
        changed_distribution.probs[valid], original_distribution.probs[valid]
    )
    torch.testing.assert_close(changed_values, original_values)


def test_forward_is_finite_for_evidence_and_single_action_edge_states() -> None:
    torch.manual_seed(SEED)
    policy = SmallPolicyNetwork()
    no_evidence = _observation(4)
    no_evidence["node_features"][..., 8:35] = 0.0
    full_evidence = _observation(4)
    full_evidence["node_features"][..., 14:20] = 1.0
    full_evidence["node_features"][..., 32:35] = 1.0
    one_action = _observation(4)
    one_action["action_mask"][:] = False
    one_action["action_mask"][:, STOP_ACTION_INDEX] = True

    for observation in (no_evidence, full_evidence, one_action):
        logits, values = policy(observation)
        distribution, masked_values = policy.distribution(observation)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(values).all()
        assert torch.isfinite(masked_values).all()
        assert torch.isfinite(distribution.probs).all()
        assert torch.isfinite(distribution.entropy()).all()


def test_recomputed_action_log_probability_is_identical() -> None:
    torch.manual_seed(SEED)
    policy = SmallPolicyNetwork().eval()
    observation = _batch(_observation(7), _observation(2))

    stored = policy.act(observation)
    recomputed_log_probabilities, entropies, values = policy.evaluate_actions(
        observation,
        stored.actions,
    )

    assert torch.equal(recomputed_log_probabilities, stored.log_probabilities)
    assert torch.equal(entropies, stored.entropies)
    assert torch.equal(values, stored.values)


def test_masked_max_pool_never_uses_a_padded_zero_or_positive_value() -> None:
    embeddings = torch.tensor([[[-3.0, -4.0], [-2.0, -5.0], [99.0, 99.0]]])
    node_mask = torch.tensor([[True, True, False]])

    pooled = _masked_pool(embeddings, node_mask)

    torch.testing.assert_close(pooled, torch.tensor([[[-2.5, -4.5, -2.0, -4.0]]]).squeeze(0))


def test_adjacency_is_preserved_input_but_not_used_by_first_mlp() -> None:
    torch.manual_seed(SEED)
    policy = SmallPolicyNetwork().eval()
    original = _observation(5)
    changed = {name: value.clone() for name, value in original.items()}
    changed["adjacency"] = 1.0 - changed["adjacency"]

    original_logits, original_values = policy(original)
    changed_logits, changed_values = policy(changed)

    assert torch.equal(changed_logits, original_logits)
    assert torch.equal(changed_values, original_values)
