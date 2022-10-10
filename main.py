import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from dqn_lightning import DQNLightning
from environments.generator.single_vessel_graph_generator import SingleVesselGraphGenerator
from environments.graph_env import GraphEnv
from settings import USE_CUDA

os.environ["WANDB_API_KEY"] = '237099249b3c0e91437061c393ab089d03339bc3'

# FIXME: Move it to ENV variable. I can't do it now because I don't want to deal with the DOCKER rebuild process.
WANDB_PATH = '/data' if USE_CUDA == 1 else '.'

#wandb.init(project="graph-rl", entity="jbsimoes", mode=os.getenv("WANDB_UPLOAD_MODE", "online"), dir=WANDB_PATH)

@hydra.main(config_path="configs", config_name="default_config")
def run_from_config_file(cfg: DictConfig):
    seed_everything(cfg.random_seed, workers=True)

    graph_generator = SingleVesselGraphGenerator(**cfg.environment)

    environment = GraphEnv(max_steps=cfg.max_steps, irrigation_goal=cfg.irrigation_goal)
    train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)

    model = DQNLightning(env=environment, graphs=train_graphs, num_dataloader_workers=cfg.num_dataloader_workers,
                         multi_action_q_network=cfg.multi_action_q_network, **cfg.core)

    trainer = Trainer(
        max_epochs=1000,
        #max_time={'hours': cfg.training_duration_in_hours},
        gpus=[1] if USE_CUDA else None,
        limit_val_batches=1,
        check_val_every_n_epoch=cfg.validation_interval,
        # deterministic=cfg.deterministic
    )

    trainer.fit(model)


if __name__ == '__main__':
    run_from_config_file()
