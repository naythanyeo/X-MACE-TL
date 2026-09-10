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
from itertools import zip_longest

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
            # Update indices by head as a dict containing head index: indices in the dataset
            # head: [0, 1, 2 ....]
            self.indices_by_head.setdefault(head_index, []).append(dataset_index)

        self.samples_per_head = None

        if self.balance_heads:
            self.samples_per_head = min(
                len(indices) for indices in self.indices_by_head.values()
            )

    def __iter__(self) -> Iterator[List[int]]:
        batches_by_head = defaultdict(list)
        for head, head_indices in self.indices_by_head.items():
            # For every head, group the indices into batches by head based on batch size
            indices = head_indices.copy()

            # Must shuffle first so that for each epoch, before cutting down by indices it uses a different set
            # Ensures that when balancing data it will be done fairly and evenly
            if self.shuffle:
                self._rng.shuffle(indices)

            # If balance head in enabled, restrict the total number of head indices to the minimum value of indices 
            if self.samples_per_head is not None:
                indices = indices[:self.samples_per_head]

            for start in range(0, len(indices), self.batch_size):
                batches_by_head[head].append(indices[start:start + self.batch_size])

            # Each head itself shuffle the batches
            if self.shuffle:
                self._rng.shuffle(batches_by_head[head])

        # Use zip longest so that if balance heads is false, unbalanced remainder is not cut out
        for batch_group in zip_longest(*batches_by_head.values()):
            for batch in batch_group:
                if batch is not None:
                    yield batch

    def __len__(self) -> int:
        if self.balance_heads:
            # If heads are balanced, count batches per head 
            # Add batch size - 1 to prevent rounding down
            batches_per_head = (self.samples_per_head + self.batch_size - 1) // self.batch_size
            return batches_per_head * len(self.indices_by_head)

        # If balance heads is false, then count from indices_by_head
        # This takes all the batches
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.indices_by_head.values()
        )