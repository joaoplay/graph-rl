from pytorch_lightning import LightningModule

from environments.graph_env import GraphEnv


class PPOLightning(LightningModule):

    def __init__(self, env: GraphEnv = None, graphs=None, batch_size: int = 32, lr: float = 0.00025,
                 gamma: float = 0.99, sync_rate: int = 10000, replay_size: int = 10 ** 6,
                 eps_last_frame: int = 5 * 10 ** 5, eps_start: float = 1.0, eps_end: float = 0.2,
                 warm_start_steps: int = 50000, action_modes: tuple[int] = DEFAULT_ACTION_MODES,
                 multi_action_q_network: dict = None, num_dataloader_workers: int = 1) -> None:
        super().__init__()

        self.save_hyperparameters()

        number_of_nodes = graphs[0].num_nodes
        representation_dim = graphs[0].representation_dim

        self.q_networks = MultiActionModeDQN(action_modes=self.hparams.action_modes,
                                             input_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: representation_dim,
                                                 ACTION_MODE_SELECTING_END_NODE: representation_dim,
                                             },
                                             action_output_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: number_of_nodes,
                                                 ACTION_MODE_SELECTING_END_NODE: number_of_nodes,
                                             }, **self.hparams.multi_action_q_network)
        self.target_q_networks = MultiActionModeDQN(action_modes=self.hparams.action_modes,
                                                    input_dim={
                                                        ACTION_MODE_SELECTING_START_NODE: representation_dim,
                                                        ACTION_MODE_SELECTING_END_NODE: representation_dim,
                                                    },
                                                    action_output_dim={
                                                        ACTION_MODE_SELECTING_START_NODE: number_of_nodes,
                                                        ACTION_MODE_SELECTING_END_NODE: number_of_nodes,
                                                    }, **self.hparams.multi_action_q_network)

        self.env = env
        self.graphs = graphs
        self.buffer = MultiActionReplayBuffer(self.hparams.action_modes)
        self.agent = GraphAgent(self.env, self.graphs, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)