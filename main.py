import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger

from dqn_lightning import DQNLightning
from environments.generator.single_vessel_graph_generator import SingleVesselGraphGenerator
from environments.graph_env import GraphEnv


@hydra.main(config_path="configs", config_name="default_config")
def run_from_config_file(cfg: DictConfig):
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    graph_generator = SingleVesselGraphGenerator(**cfg.environment)

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYmQ4MjE1OC0yNzBhLTQyNzctYjFmZS00YTFhYjYxZTdmMjUifQ==",  # replace with your own
        project="jbsimoes/graph-rl",
        mode=os.getenv("NEPTUNE_MODE", "async")
    )
    environment = GraphEnv(max_steps=3000, irrigation_goal=0.06)
    train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)
    model = DQNLightning(environment, train_graphs, replay_size=10**6)

    trainer = Trainer(
        max_epochs=10**12,
        # max_time={'hours': 2},
        val_check_interval=100,
        logger=neptune_logger,
        progress_bar_refresh_rate=0,
    )

    trainer.fit(model)


if __name__ == '__main__':
    run_from_config_file()
