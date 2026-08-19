"""
Helper class for manually grouping batches by head.
The class is initialised with a completed atomic dataset object
It first builds indices by head, then uses that dictionary to construct
batches from it such that each batch has only one head 
The batches are shuffled randomly if specified. 

__iter__ will be accessed by dataloader 
"""

import random

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterator, List, Optional, Sequence

from .atomic_data import AtomicData


@dataclass
class HeadBatchSampler:
    atomic_dataset: Sequence[AtomicData]
    batch_size: int
    shuffle: bool = False
    seed: Optional[int] = None
    balance_heads: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self._rng = random.Random(self.seed)
        self.indices_by_head = {}

        for dataset_index, atomic_data in enumerate(self.atomic_dataset):
            head_index = int(atomic_data.head.item())
            self.indices_by_head.setdefault(head_index, []).append(dataset_index)

    def __iter__(self) -> Iterator[List[int]]:
        batches_by_head = defaultdict(list)
        samples_per_head = None
        if self.balance_heads:
            samples_per_head = min(
                len(indices) for indices in self.indices_by_head.values()
            )

        for head, head_indices in self.indices_by_head.items():
            indices = head_indices.copy()

            if self.shuffle:
                self._rng.shuffle(indices)
            if samples_per_head is not None:
                indices = indices[:samples_per_head]

            for start in range(0, len(indices), self.batch_size):
                batches_by_head[head].append(indices[start:start + self.batch_size])

            # Each head itself shuffle the batches
            if self.shuffle:
                self._rng.shuffle(batches_by_head[head])

        for batch_group in zip(*batches_by_head.values()):
            for batch in batch_group:
                yield batch

    def __len__(self) -> int:
        batches_per_head = min(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.indices_by_head.values()
        )
        return len(self.indices_by_head) * batches_per_head