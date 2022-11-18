from collections import deque, namedtuple
import numpy as np

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "solved", "new_state", "goal"], )


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

    def append_many(self, states, actions, rewards, terminals, next_states, goals):
        experiences = []

        for exp_idx in range(len(states)):
            goals_np = np.array([goals[exp_idx]], dtype=np.float32)
            state_plus_goal = np.concatenate((states[exp_idx], goals_np))
            next_state_plus_goal = np.concatenate((next_states[exp_idx], goals_np))
            experience = Experience(state_plus_goal, actions[exp_idx], rewards[exp_idx], terminals[exp_idx],
                                    next_state_plus_goal, goals[exp_idx])
            experiences += [experience]
            self.append(experience)

        return experiences

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, solved, next_states, goals = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(solved, dtype=np.bool),
            np.array(next_states, dtype=np.float32),
            np.array(goals, dtype=np.float32),
        )
