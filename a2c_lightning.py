from pytorch_lightning import LightningModule

from agents.graph_agent import GraphAgent
from environments.graph_env import DEFAULT_ACTION_MODES, GraphEnv


class A2CLightning(LightningModule):
    """
    A2C Lightning
    """

    def __init__(self, env: GraphEnv = None, graphs=None, batch_size: int = 64, hidden_size: int = 28,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 sync_rate: int = 10000,
                 replay_size: int = 10 ** 6, warm_start_size: int = 100000, eps_last_frame: int = 5 * 10**5,
                 eps_start: float = 1.0,
                 eps_end: float = 0.0, episode_length: int = 200, warm_start_steps: int = 50000,
                 action_modes: tuple[int] = DEFAULT_ACTION_MODES) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.env = env
        self.graphs = graphs
        self.agent = GraphAgent(self.env, self.graphs, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def training_step(self, batch, nb_batch):
        pass




