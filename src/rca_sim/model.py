"""Policy and value network definitions for the small simulator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from rca_sim.observation import GLOBAL_FEATURE_COUNT, NODE_FEATURE_COUNT
from rca_sim.tools import (
    ACTION_SPACE_SIZE,
    MAX_SERVICE_SLOTS,
    PROBE_TEMPLATES,
    PROBE_TEMPLATE_NAMES,
    TEMPLATES_PER_SERVICE,
)


SERVICE_EMBEDDING_COUNT = 64
POOLED_FEATURE_COUNT = 2 * SERVICE_EMBEDDING_COUNT
CANDIDATE_FEATURE_COUNT = (
    SERVICE_EMBEDDING_COUNT
    + POOLED_FEATURE_COUNT
    + GLOBAL_FEATURE_COUNT
    + TEMPLATES_PER_SERVICE
    + 2
)


@dataclass(frozen=True)
class PolicyStep:
    """One batched policy decision ready for rollout storage."""

    actions: Tensor
    log_probabilities: Tensor
    entropies: Tensor
    values: Tensor


class MaskedCategorical(Categorical):
    """Categorical distribution with one consistently applied action mask."""

    def __init__(self, *, logits: Tensor, action_mask: Tensor) -> None:
        if logits.ndim < 2 or logits.shape[-1] != ACTION_SPACE_SIZE:
            raise ValueError("logits must end with the 41-action dimension")
        if action_mask.shape != logits.shape:
            raise ValueError("action_mask must have the same shape as logits")
        action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
        if not torch.all(action_mask.any(dim=-1)):
            raise ValueError("each distribution row must have a valid action")
        self.action_mask = action_mask
        super().__init__(logits=logits.masked_fill(~action_mask, -torch.inf))

    def entropy(self) -> Tensor:
        """Return entropy without multiplying zero probabilities by infinity."""
        finite_log_probabilities = self.logits.masked_fill(~self.action_mask, 0.0)
        return -(self.probs * finite_log_probabilities).sum(dim=-1)


class SmallPolicyNetwork(nn.Module):
    """Shared MLP candidate scorer and value estimator for ``sim_v0``.

    This first policy uses the topology columns already present in each
    service's node features and a permutation-invariant pooled context. It
    deliberately does not perform adjacency-based message passing and is not
    a graph neural network. The observation keeps ``adjacency`` for a later
    graph encoder.
    """

    def __init__(self) -> None:
        super().__init__()
        self.service_encoder = nn.Sequential(
            nn.Linear(NODE_FEATURE_COUNT, SERVICE_EMBEDDING_COUNT),
            nn.ReLU(),
            nn.Linear(SERVICE_EMBEDDING_COUNT, SERVICE_EMBEDDING_COUNT),
            nn.ReLU(),
        )
        self.candidate_scorer = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURE_COUNT, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        state_feature_count = POOLED_FEATURE_COUNT + GLOBAL_FEATURE_COUNT
        self.stop_head = nn.Sequential(
            nn.Linear(state_feature_count, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(state_feature_count, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.register_buffer(
            "template_one_hot",
            torch.eye(TEMPLATES_PER_SERVICE),
            persistent=False,
        )
        self.register_buffer(
            "template_costs",
            torch.tensor(
                [
                    PROBE_TEMPLATES[name].cost_credits / 4.0
                    for name in PROBE_TEMPLATE_NAMES
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "template_record_masks",
            _template_record_masks(),
            persistent=False,
        )

    def forward(self, observation: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return unmasked action logits and one critic value per sample."""
        # ``adjacency`` remains part of the observation contract, but this MLP
        # does not consume it. See the class docstring before changing this.
        node_features = observation["node_features"]
        node_mask = observation["node_mask"].to(dtype=torch.bool)
        global_features = observation["global_features"]
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [batch, services, 42]")
        if node_features.shape[1:] != (MAX_SERVICE_SLOTS, NODE_FEATURE_COUNT):
            raise ValueError("node_features has the wrong service or feature count")
        if node_mask.shape != node_features.shape[:2]:
            raise ValueError("node_mask must have shape [batch, services]")
        if global_features.shape != (node_features.shape[0], GLOBAL_FEATURE_COUNT):
            raise ValueError("global_features must have shape [batch, 10]")
        if not torch.all(node_mask.any(dim=1)):
            raise ValueError("each observation must contain an active service")

        embeddings = self.service_encoder(node_features)
        pooled = _masked_pool(embeddings, node_mask)
        state_features = torch.cat((pooled, global_features), dim=-1)

        batch_size = node_features.shape[0]
        service_context = embeddings[:, :, None, :].expand(
            -1, -1, TEMPLATES_PER_SERVICE, -1
        )
        pooled_context = pooled[:, None, None, :].expand(
            -1, MAX_SERVICE_SLOTS, TEMPLATES_PER_SERVICE, -1
        )
        global_context = global_features[:, None, None, :].expand(
            -1, MAX_SERVICE_SLOTS, TEMPLATES_PER_SERVICE, -1
        )
        one_hot = self.template_one_hot.to(dtype=node_features.dtype).view(
            1, 1, TEMPLATES_PER_SERVICE, TEMPLATES_PER_SERVICE
        ).expand(batch_size, MAX_SERVICE_SLOTS, -1, -1)
        costs = self.template_costs.to(dtype=node_features.dtype).view(
            1, 1, TEMPLATES_PER_SERVICE, 1
        ).expand(batch_size, MAX_SERVICE_SLOTS, -1, -1)
        fraction_new = self._fraction_of_records_new(node_features).unsqueeze(-1)

        candidate_features = torch.cat(
            (
                service_context,
                pooled_context,
                global_context,
                one_hot,
                costs,
                fraction_new,
            ),
            dim=-1,
        )
        candidate_logits = self.candidate_scorer(candidate_features).reshape(
            batch_size, MAX_SERVICE_SLOTS * TEMPLATES_PER_SERVICE
        )
        stop_logit = self.stop_head(state_features)
        logits = torch.cat((candidate_logits, stop_logit), dim=-1)
        if logits.shape[1] != ACTION_SPACE_SIZE:  # pragma: no cover
            raise RuntimeError("policy action count does not match the action schema")
        values = self.critic_head(state_features).squeeze(-1)
        return logits, values

    def distribution(
        self,
        observation: Mapping[str, Tensor],
    ) -> tuple[MaskedCategorical, Tensor]:
        """Build the masked distribution used by collection and PPO updates."""
        logits, values = self(observation)
        distribution = MaskedCategorical(
            logits=logits,
            action_mask=observation["action_mask"],
        )
        return distribution, values

    def act(
        self,
        observation: Mapping[str, Tensor],
        *,
        deterministic: bool = False,
    ) -> PolicyStep:
        """Sample masked actions and return the rollout quantities to store."""
        distribution, values = self.distribution(observation)
        actions = (
            distribution.probs.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        return PolicyStep(
            actions=actions,
            log_probabilities=distribution.log_prob(actions),
            entropies=distribution.entropy(),
            values=values,
        )

    def evaluate_actions(
        self,
        observation: Mapping[str, Tensor],
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Recompute PPO log probabilities, entropy, and critic values."""
        distribution, values = self.distribution(observation)
        return distribution.log_prob(actions), distribution.entropy(), values

    def _fraction_of_records_new(self, node_features: Tensor) -> Tensor:
        metric_present = node_features[..., 14:20]
        log_present = node_features[..., 32:35]
        record_present = torch.cat((metric_present, log_present), dim=-1)
        masks = self.template_record_masks.to(dtype=node_features.dtype)
        present_counts = torch.einsum("bsr,tr->bst", record_present, masks)
        record_counts = masks.sum(dim=-1).view(1, 1, TEMPLATES_PER_SERVICE)
        return 1.0 - present_counts / record_counts


def _masked_pool(embeddings: Tensor, node_mask: Tensor) -> Tensor:
    mask = node_mask.unsqueeze(-1)
    active_count = mask.sum(dim=1).clamp_min(1).to(dtype=embeddings.dtype)
    mean_pool = (embeddings * mask).sum(dim=1) / active_count
    max_pool = embeddings.masked_fill(~mask, -torch.inf).max(dim=1).values
    return torch.cat((mean_pool, max_pool), dim=-1)


def _template_record_masks() -> Tensor:
    """Map each template to the nine per-service evidence-presence columns."""
    masks = torch.zeros((TEMPLATES_PER_SERVICE, 9))
    for template_index, name in enumerate(PROBE_TEMPLATE_NAMES):
        template = PROBE_TEMPLATES[name]
        if template.tool.value == "metrics":
            for metric_index in range(3):
                for reading_index in template.reading_indices:
                    masks[template_index, metric_index * 2 + reading_index] = 1.0
        else:
            for reading_index in template.reading_indices:
                masks[template_index, 6 + reading_index] = 1.0
    return masks
