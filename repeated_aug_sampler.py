"""RepeatedAugSampler module: A custom PyTorch Sampler for repeated data augmentation."""

import random
from torch.utils.data import Sampler


class RepeatedAugSampler(Sampler):
    """A sampler that repeats each dataset sample multiple times for augmentation.

    Args:
        data_source (Dataset): The dataset to sample from.
        num_repeats (int): Number of times each sample is repeated.
        shuffle (bool): Whether to shuffle the repeated indices.
    """

    def __init__(self, data_source, num_repeats=3, shuffle=True):
        self.dataset = data_source
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.num_samples = len(self.dataset) * self.num_repeats

    def __iter__(self):
        indices = list(range(len(self.dataset))) * self.num_repeats
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples
