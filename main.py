import os
import time

import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from dqn import DQN
from environments.generator.single_vessel_graph_generator import SingleVesselGraphGenerator
from environments.generator.vascular_network_from_file_generator import VascularNetworkFromFileGenerator
from environments.graph_env import GraphEnv
from settings import USE_CUDA, WANDB_PATH, BASE_PATH

os.environ["WANDB_API_KEY"] = '237099249b3c0e91437061c393ab089d03339bc3'

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
                               irrigation_compression=cfg.irrigation_compression,
                               irrigation_grid_dim=cfg.irrigation_grid_dim,
                               irrigation_grid_cell_size=cfg.irrigation_grid_cell_size,
                               irrigation_percentage_goal=h.irrigation_percentage_goal,
                               exclude_isolated_from_start_nodes=cfg.exclude_isolated_from_start_nodes)
        train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)

        model = DQN(env=environment, graphs=train_graphs, num_dataloader_workers=cfg.num_dataloader_workers,
                    multi_action_q_network=cfg.multi_action_q_network, **h.core, device='cuda' if USE_CUDA else 'cpu',
                    use_hindsight=cfg.use_hindsight)
        model.load_models(folder_path)
        model.populate(model.hparams.warm_start_steps)

        model.train(training_steps, validation_interval=cfg.validation_interval)

        model.save_models(folder_path)


def run_experiment(cfg: DictConfig):
    if cfg.constant_flow:
        graph_generator = VascularNetworkFromFileGenerator(f"{BASE_PATH}/{cfg.environment.initial_env_file}")
    else:
        graph_generator = SingleVesselGraphGenerator(**cfg.environment)

    print(f'Setting up environment...')
    environment = GraphEnv(max_steps=cfg.max_steps, irrigation_goal=cfg.irrigation_goal,
                           inject_irrigation=cfg.inject_irrigation,
                           irrigation_compression=cfg.irrigation_compression,
                           irrigation_grid_dim=cfg.irrigation_grid_dim,
                           irrigation_grid_cell_size=cfg.irrigation_grid_cell_size,
                           irrigation_percentage_goal=1.0,
                           exclude_isolated_from_start_nodes=cfg.exclude_isolated_from_start_nodes,
                           use_irrigation_improvement=cfg.use_irrigation_improvement,
                           constant_flow=cfg.constant_flow)

    print(f'Generating graphs...')
    train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)


    model = DQN(env=environment, graphs=train_graphs, num_dataloader_workers=cfg.num_dataloader_workers,
                multi_action_q_network=cfg.multi_action_q_network, **cfg.core, device='cuda' if USE_CUDA else 'cpu',
                use_hindsight=cfg.use_hindsight, double_dqn=cfg.double_dqn)
    print(f'Populating replay buffer...')
    model.populate(model.hparams.warm_start_steps)

    print(f'Training...')
    model.train(650001, validation_interval=cfg.validation_interval)


@hydra.main(config_path="configs", config_name="default_config")
def run_from_config_file(cfg: DictConfig):
    wandb.init(project="graph-rl", entity="jbsimoes", mode=os.getenv("WANDB_UPLOAD_MODE", "online"), config=cfg,
               name=os.getenv("NEPTUNE_RUN_NAME", None))

    seed_everything(cfg.random_seed, workers=True)

    if 'hierarchical' in cfg and cfg.hierarchical is not None:
        run_hierarchical_experiment(cfg)
    else:
        run_experiment(cfg)


if __name__ == '__main__':
    run_from_config_file()
