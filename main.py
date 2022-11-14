import os
import time

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

from dqn_lightning import DQNLightning
from environments.generator.single_vessel_graph_generator import SingleVesselGraphGenerator
from environments.graph_env import GraphEnv
from settings import USE_CUDA, NEPTUNE_INSTANCE

os.environ["WANDB_API_KEY"] = '237099249b3c0e91437061c393ab089d03339bc3'

# FIXME: Move it to ENV variable. I can't do it now because I don't want to deal with the DOCKER rebuild process.
WANDB_PATH = '/data' if USE_CUDA == 1 else '.'

#wandb.init(project="graph-rl", entity="jbsimoes", mode=os.getenv("WANDB_UPLOAD_MODE", "online"))


def check_hierarchical_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

    folder_path = f'{path}{time.time()}'
    os.makedirs(folder_path)

    return folder_path


def run_hierarchical_experiment(cfg: DictConfig):
    folder_path = check_hierarchical_folder(f'{WANDB_PATH}/hierarchical_models/')

    graph_generator = SingleVesselGraphGenerator(**cfg.environment)

    for h in cfg.hierarchical:
        print(f'Running irrigation hierarchical level {h}')

        goal = h.irrigation_goal
        training_steps = h.training_steps

        environment = GraphEnv(max_steps=cfg.max_steps, irrigation_goal=goal, inject_irrigation=cfg.inject_irrigation,
                               irrigation_compression=cfg.irrigation_compression, irrigation_grid_dim=cfg.irrigation_grid_dim,
                               irrigation_grid_cell_size=cfg.irrigation_grid_cell_size)
        train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)

        model = DQNLightning(env=environment, graphs=train_graphs, num_dataloader_workers=cfg.num_dataloader_workers,
                             multi_action_q_network=cfg.multi_action_q_network, **h.core)
        model.load_models(folder_path)
        model.populate(model.hparams.warm_start_steps)

        trainer = Trainer(
            #max_epochs=training_steps,
            # max_time={'hours': cfg.training_duration_in_hours},
            gpus=[cfg.gpu_device] if USE_CUDA else None,
            enable_progress_bar=False,
            limit_val_batches=1,
            check_val_every_n_epoch=cfg.validation_interval,
            # deterministic=cfg.deterministic
            callbacks=[EarlyStopping(monitor='episode-length', patience=cfg.early_stopping_patience, mode='min', min_delta=2)]
        )

        trainer.fit(model)

        model.save_models(folder_path)


def run_experiment(cfg: DictConfig):
    graph_generator = SingleVesselGraphGenerator(**cfg.environment)

    environment = GraphEnv(max_steps=cfg.max_steps, irrigation_goal=cfg.irrigation_goal, inject_irrigation=cfg.inject_irrigation,
                           irrigation_compression=cfg.irrigation_compression, irrigation_grid_dim=cfg.irrigation_grid_dim,
                           irrigation_grid_cell_size=cfg.irrigation_grid_cell_size)
    train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)

    model = DQNLightning(env=environment, graphs=train_graphs, num_dataloader_workers=cfg.num_dataloader_workers,
                         multi_action_q_network=cfg.multi_action_q_network, **cfg.core)
    model.populate(model.hparams.warm_start_steps)

    trainer = Trainer(
        #max_epochs=-1,
        max_time={'hours': 120},
        gpus=[cfg.gpu_device] if USE_CUDA else None,
        enable_progress_bar=False,
        limit_val_batches=1,
        check_val_every_n_epoch=cfg.validation_interval,
        #callbacks=[EarlyStopping(monitor='episode-length', patience=cfg.early_stopping_patience, mode='min', min_delta=1)]
        # deterministic=cfg.deterministic
    )

    trainer.fit(model)


@hydra.main(config_path="configs", config_name="default_config")
def run_from_config_file(cfg: DictConfig):
    NEPTUNE_INSTANCE['config'] = cfg

    seed_everything(cfg.random_seed, workers=True)

    if 'hierarchical' in cfg and cfg.hierarchical is not None:
        run_hierarchical_experiment(cfg)
    else:
        run_experiment(cfg)


if __name__ == '__main__':
    run_from_config_file()
