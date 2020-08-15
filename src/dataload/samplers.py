"""
sample -> batch_samples
"""

import numpy as np
from torch.utils.data.sampler import Sampler


class TimeBasedSampler(Sampler):
    def __init__(self, dataset, duration=200, ngpu=1, shuffle=False): # 200s
        self.dataset = dataset
        self.dur = duration
        self.shuffle = shuffle

        batchs = []
        batch = []
        batch_dur = 0.
        for idx in range(len(self.dataset)):
            batch.append(idx)
            batch_dur += self.dataset[idx]["duration"]
            if batch_dur >= self.dur and len(batch)%ngpu==0:
                # To make the numbers of batchs are equal for each GPU.
                batchs.append(batch)
                batch = []
                batch_dur = 0.
        if batch:
            if len(batch)%ngpu==0:
                batchs.append(batch)
            else:
                b = len(batch)
                batchs.append(batch[b//ngpu*ngpu:])
        self.batchs = batchs

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batchs)
        for b in self.batchs:
            yield b

    def __len__(self):
        return len(self.batchs)


class FrameBasedSampler(TimeBasedSampler):
    def __init__(self, dataset, frames=200, ngpu=1, shuffle=False):
        self.dataset = dataset
        self.frames = frames
        self.shuffle = shuffle

        batchs = []
        batch = []
        batch_frames = 0
        for idx in range(len(self.dataset)):
            batch.append(idx)
            batch_frames += self.dataset[idx]["feat_length"]
            if batch_frames >= self.frames and len(batch)%ngpu==0:
                # To make the numbers of batchs are equal for each GPU.
                batchs.append(batch)
                batch = []
                batch_frames = 0
        if batch:
            if len(batch)%ngpu==0:
                batchs.append(batch)
            else:
                b = len(batch)
                batchs.append(batch[b//ngpu*ngpu:])
        self.batchs = batchs
