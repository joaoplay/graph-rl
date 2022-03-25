from collections import deque, namedtuple
import numpy as np

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "new_state"], )


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity) -> None:
        super().__init__()

        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def append_many(self, states, actions, rewards, terminals, next_states):
        for exp_idx in range(len(states)):
            self.append(Experience(states[exp_idx], actions[exp_idx], rewards[exp_idx], terminals[exp_idx],
                                   next_states[exp_idx]))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states, dtype=np.float32),
        )
