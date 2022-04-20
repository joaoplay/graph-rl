import os

import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from dqn_lightning import DQNLightning
from environments.generator.single_vessel_graph_generator import SingleVesselGraphGenerator
from environments.graph_env import GraphEnv
from settings import USE_CUDA

os.environ["WANDB_API_KEY"] = '237099249b3c0e91437061c393ab089d03339bc3'

wandb.init(project="graph-rl", entity="jbsimoes", mode=os.getenv("WANDB_UPLOAD_MODE", "online"))


@hydra.main(config_path="configs", config_name="default_config")
def run_from_config_file(cfg: DictConfig):
    seed_everything(cfg.random_seed, workers=True)

    graph_generator = SingleVesselGraphGenerator(**cfg.environment)

    environment = GraphEnv(max_steps=cfg.max_steps, irrigation_goal=cfg.irrigation_goal)
    train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)

    model = DQNLightning(env=environment, graphs=train_graphs, multi_action_q_network=cfg.multi_action_q_network, **cfg.core)

    trainer = Trainer(
        max_time={'hours': cfg.training_duration_in_hours},
        gpus=[0] if USE_CUDA else None,
        progress_bar_refresh_rate=0,
        limit_val_batches=1,
        check_val_every_n_epoch=cfg.validation_interval,
        deterministic=cfg.deterministic
    )

    trainer.fit(model)


if __name__ == '__main__':
    run_from_config_file()
