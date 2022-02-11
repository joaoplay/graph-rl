import collections

import numpy as np

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceBuffer:

    def __init__(self, capacity) -> None:
        super().__init__()

        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def append_many(self, states, actions, rewards, terminals, next_states):
        for exp_idx in range(len(states)):
            self.append(Experience(states[exp_idx], actions[exp_idx], rewards[exp_idx], terminals[exp_idx],
                                   next_states[exp_idx]))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return zip(*[self.buffer[idx] for idx in indices])
