import numpy as np

from agents.replay_memory.replay_buffer import ReplayBuffer


class MultiActionReplayBuffer:

    def __init__(self, action_modes: tuple[int], capacity=10 ** 5) -> None:
        super().__init__()

        self.action_modes = action_modes
        self.experience_buffer_by_action_mode = {action_mode: ReplayBuffer(capacity=capacity)
                                                 for action_mode in self.action_modes}

    def append(self, action_mode, experience):
        experience_buffer = self.experience_buffer_by_action_mode[action_mode]
        experience_buffer.append(experience)

    def append_many(self, action_mode, states, actions, rewards, terminals, next_states):
        experience_buffer = self.experience_buffer_by_action_mode[action_mode]
        experience_buffer.append_many(states, actions, rewards, terminals, next_states)

    def get_experience_buffer(self, action_mode):
        return self.experience_buffer_by_action_mode[action_mode]

    def sample(self, batch_size):
        action_mode_idx = np.random.randint(len(self.action_modes))
        action_mode = self.action_modes[action_mode_idx]
        experience_buffer = self.experience_buffer_by_action_mode[action_mode]

        return action_mode, *experience_buffer.sample(batch_size)
