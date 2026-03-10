from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    # Optional debug
    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # TODO(student): yield RolloutBatch minibatches of size minibatch_size.
    # Requirements:
    # - Let N = batch.input_ids.shape[0] be the number of sampled completions.
    # - If shuffle=True, permute indices with torch.randperm using the provided generator.
    # - Otherwise iterate in the original order 0, 1, ..., N-1.
    # - Slice ALL tensor fields consistently with the same minibatch indices.
    # - Keep task_names / completion_texts aligned with the same indices when present.
    # - If device is not None, move the minibatch to that device before yielding.
    N = batch.input_ids.shape[0]
    indices = torch.randperm(N, generator=generator) if shuffle else torch.arange(N)
    for start in range(0, N, minibatch_size):
        end = start + minibatch_size
        batch_indices = indices[start:end]
        minibatch = RolloutBatch(
            input_ids=batch.input_ids[batch_indices],
            attention_mask=batch.attention_mask[batch_indices],
            completion_mask=batch.completion_mask[batch_indices],
            old_logprobs=batch.old_logprobs[batch_indices],
            ref_logprobs=batch.ref_logprobs[batch_indices],
            rewards=batch.rewards[batch_indices],
            advantages=batch.advantages[batch_indices],
            task_names=[batch.task_names[i] for i in batch_indices.tolist()] if batch.task_names else None,
            completion_texts=[batch.completion_texts[i] for i in batch_indices.tolist()] if batch.completion_texts else None,
        )
        if device is not None:
            minibatch = minibatch.to(device)
        yield minibatch
    

