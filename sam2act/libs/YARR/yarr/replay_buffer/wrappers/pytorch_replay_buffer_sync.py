

# From: https://github.com/stepjam/YARR/blob/main/yarr/replay_buffer/wrappers/pytorch_replay_buffer.py

import time
from threading import Lock, Thread

import os
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
import torch.multiprocessing as mp
from multiprocessing import Array

from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.replay_buffer.wrappers import WrappedReplayBuffer
import time


class PyTorchIterableReplayDataset(IterableDataset):

    def __init__(self, replay_buffer: ReplayBuffer, sample_mode, sample_distribution_mode = 'transition_uniform', task_losses_avg=None):
        self._replay_buffer = replay_buffer
        self._sample_mode = sample_mode
        if self._sample_mode == 'enumerate':
            self._num_samples = self._replay_buffer.prepare_enumeration()
        self._sample_distribution_mode = sample_distribution_mode

        self._task_losses_avg = task_losses_avg

    def get_task_losses_avg_np(self):
        # Convert the shared array to a NumPy array before passing it to the dataset or sampling methods
        return np.frombuffer(self._task_losses_avg.get_obj())

    def _generator(self):
        while True:
            if self._sample_mode == 'random':
                yield self._replay_buffer.sample_transition_batch(pack_in_dict=True, distribution_mode = self._sample_distribution_mode, task_losses_avg=self.get_task_losses_avg_np())
            elif self._sample_mode == 'enumerate':
                yield self._replay_buffer.enumerate_next_transition_batch(pack_in_dict=True)

    def __iter__(self):
        return iter(self._generator())

    def __len__(self): # enumeration will throw away the last incomplete batch
        return self._num_samples // self._replay_buffer._batch_size

class PyTorchReplayBuffer_sync(WrappedReplayBuffer):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

    Usage:
      To add a transition:  call the add function.

      To sample a batch:    Construct operations that depend on any of the
                            tensors is the transition dictionary. Every sess.run
                            that requires any of these tensors will sample a new
                            transition.
      sample_mode: the mode to sample data, choose from ['random', 'enumerate']
    """

    def __init__(self, replay_buffer: ReplayBuffer, num_workers: int = 2, sample_mode = 'random', sample_distribution_mode = 'transition_uniform'):
        super(PyTorchReplayBuffer_sync, self).__init__(replay_buffer)
        self._num_workers = num_workers
        self._sample_mode = sample_mode
        self._sample_distribution_mode = sample_distribution_mode

        self._num_tasks = len(replay_buffer._task_names)
        BIG_CONSTANT = 999999.0
        self._task_losses_avg = Array('d', [BIG_CONSTANT] * self._num_tasks)  # Shared array for task losses avg

    def update_task_losses_avg(self, task_losses):
        with self._task_losses_avg.get_lock():
            avg_losses = np.mean(task_losses, axis=1)
            np.copyto(np.frombuffer(self._task_losses_avg.get_obj()), avg_losses)

    def get_task_losses_avg_np(self):
        # Convert the shared array to a NumPy array before passing it to the dataset or sampling methods
        return np.frombuffer(self._task_losses_avg.get_obj())

    def dataset(self) -> DataLoader:
        d = PyTorchIterableReplayDataset(self._replay_buffer, self._sample_mode, self._sample_distribution_mode, self._task_losses_avg)

        # Batch size None disables automatic batching
        return DataLoader(d, batch_size=None, pin_memory=True,
                          num_workers=self._num_workers)
    
