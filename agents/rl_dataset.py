from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

from agents.replay_memory.multi_action_replay_buffer import MultiActionReplayBuffer


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError()

    def __init__(self, buffer: MultiActionReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        action_mode, states, actions, rewards, solved, new_states, goals = self.buffer.sample(self.sample_size)
        for i in range(len(solved)):
            yield action_mode, states[i], actions[i], rewards[i], solved[i], new_states[i], goals[i]

