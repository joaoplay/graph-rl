import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from agents.graph_agent import GraphAgent
from environments.graph_env import DEFAULT_ACTION_MODES, GraphEnv

class A2CLightning(LightningModule):

    def __init__(self, env: GraphEnv = None, graphs=None, batch_size: int = 64, hidden_size: int = 28,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 sync_rate: int = 10000,
                 replay_size: int = 10 ** 6, warm_start_size: int = 100000, eps_last_frame: int = 5 * 10**5,
                 eps_start: float = 1.0,
                 eps_end: float = 0.0, episode_length: int = 200, warm_start_steps: int = 50000,
                 action_modes: tuple[int] = DEFAULT_ACTION_MODES, reward_steps = 4) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.env = env
        self.graphs = graphs
        self.agent = GraphAgent(self.env, self.graphs, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def unpack_batch(self, batch, net):
        states = []
        actions = []
        rewards = []
        not_done_idx = []
        last_states = []
        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))
            actions.append(int(exp.action))
            rewards.append(exp.reward)
            if exp.last_state is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))

        states_v = torch.FloatTensor(np.array(states, copy=False))
        actions_t = torch.LongTensor(actions)

        rewards_np = np.array(rewards, dtype=np.float32)
        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False))
            last_vals_v = net(last_states_v)[1]
            last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
            last_vals_np *= self.hparams.gamma ** self.hparams.reward_steps
            rewards_np[not_done_idx] += last_vals_np

        ref_vals_v = torch.FloatTensor(rewards_np)

        return states_v, actions_t, ref_vals_v

    def training_step(self, *args, **kwargs):
        pass




