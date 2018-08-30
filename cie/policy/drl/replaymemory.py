"""
    Replay Memory for DQN Agent
"""
import logging
import pickle
import random

import numpy as np

logger = logging.getLogger(__name__)


class ReplayMemory(object):
    def __init__(self, memory_size, load_path, **kwargs):
        self.memory_size = int(memory_size)
        self.storage = list()
        self.ptr = 0

        # Load from designated path
        if load_path:
            msg = "Loading buffered memory from {}...".format(load_path)
            print(msg)
            logger.info(msg)
            with open(load_path, 'rb') as fin:
                self.storage = pickle.load(fin)
                self.ptr = len(self.storage) - 1

    def __len__(self):
        return len(self.storage)

    def size(self):
        return len(self.storage)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self.ptr >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.memory_size

    def sample_encode(self, batch_size):
        """
        Sample from memory and encode to numpy array
        """
        # Validate batch size
        batch_size = min(batch_size, len(self.storage))

        sampled_indices = random.sample(range(len(self.storage)), batch_size)

        # Sample to list
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_done = []
        for sampled_idx in sampled_indices:
            (state, action, reward, next_state,
             done) = self.storage[sampled_idx]
            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_states.append(next_state)
            batch_done.append(done)

        # Encode to numpy array
        batch_states = np.asarray(batch_states, dtype=np.float32)
        batch_actions = np.asarray(batch_actions, dtype=np.int16)
        batch_rewards = np.asarray(batch_rewards, dtype=np.float32)
        batch_next_states = np.asarray(batch_next_states, dtype=np.float32)
        batch_done = np.asarray(batch_done, dtype=np.float32)
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_done

    def clear(self):
        self.storage.clear()
        self.ptr = 0


if __name__ == "__main__":
    """  Debug """
    memory = ReplayMemory(10)
    # Test add
    for x in range(15):
        memory.add(x, x, x, x, False)
        print(memory.ptr)

    print(memory.storage)

    # Test encocde_sample
    print(memory.sample_encode(5))
