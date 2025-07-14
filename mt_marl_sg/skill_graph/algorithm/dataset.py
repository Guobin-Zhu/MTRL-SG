import torch
from typing import List, Tuple, Dict, Any
from typing_extensions import Self
import numpy as np
import random

class Dataset:
    def __init__(self) -> None:
        self._h = None  
        self._w = None  
        self._d = None  
        self._t = None  
        self._i = None  
        self._len = 0   # Current dataset size

    def add(self, hwdti: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]):
        h, w, d, t, i = hwdti
        assert h.size(0) == w.size(0) == d.size(0) == t.size(0) == len(i)

        # Initialize storage if dataset is empty
        if self._len == 0:
            self._h = h
            self._w = w
            self._d = d
            self._t = t
            self._i = i
            self._len += h.size(0)
            return self
        
        # Ensure storage exists before concatenation
        assert all(x is not None for x in [self._h, self._w, self._d, self._t, self._i])

        # Concatenate new data to existing storage
        self._h = torch.cat((self._h, h))
        self._w = torch.cat((self._w, w))
        self._d = torch.cat((self._d, d))
        self._t = torch.cat((self._t, t))
        self._i.extend(i)  # Extend metadata list
        self._len += h.size(0)  # Update dataset size

        return self

    def len(self) -> int:
        """Return current number of entries in dataset"""
        return self._len

    def sample(self, batch_size: int, repeats: int = 1, all: bool = False):
        """Sample batch data with optional repetition and full-dataset override"""
        # Ensure dataset is initialized
        assert self._h is not None and self._w is not None and self._d is not None and self._t is not None and self._i is not None

        # Sampling logic:
        if all:
            indices = list(range(self._len))  # Use all entries
        elif batch_size > self._len:
            indices = random.choices(range(self._len), k=batch_size)  # Allow duplicates
        else:
            indices = random.sample(range(self._len), k=batch_size)  # Unique sampling
            
        # Prepare metadata with repeated entries if needed
        _is = []
        for id in indices:
            for _ in range(repeats):
                _is.append(self._i[id])

        # Return tensors repeated along batch dimension
        return (
            self._h[indices].repeat_interleave(repeats, dim=0),
            self._w[indices].repeat_interleave(repeats, dim=0),
            self._d[indices].repeat_interleave(repeats, dim=0),
            self._t[indices].repeat_interleave(repeats, dim=0),
            _is
        )

    def _validate(self):
        """Internal method to verify dataset initialization"""
        assert self._h is not None and self._w is not None and self._d is not None and self._t is not None and self._i is not None

    def split(self, percents: Tuple[float, float]) -> Tuple[Self, Self]:
        """Split dataset into partitions based on percentage ratios"""
        # Validate dataset state
        assert self._h is not None and self._w is not None and self._d is not None and self._t is not None and self._i is not None

        l = self._len
        # Convert percentages to absolute sizes
        ns = tuple(map(lambda n: int(n * l), percents))
        assert sum(ns) == l  # Verify coverage

        # Shuffle indices and partition
        all_ids = list(range(l))
        np.random.shuffle(all_ids)
        ids = []
        for i in range(len(percents)):
            partition_size = ns[i]
            ids.append(all_ids[-partition_size:])  # Take from end
            del all_ids[-partition_size:]  # Remove used indices

        # Verify partition integrity
        assert len(all_ids) == 0 and sum(map(len, ids)) == l
        
        # Create new Dataset instances for partitions
        rlts = tuple([Dataset() for _ in range(len(percents))])
        for i, r in enumerate(rlts):
            partition_indices = ids[i]
            partition_metadata = [self._i[j] for j in partition_indices]
            # Populate partition dataset
            r.add((
                self._h[partition_indices], 
                self._w[partition_indices],
                self._d[partition_indices],
                self._t[partition_indices],
                partition_metadata
            ))

        return rlts