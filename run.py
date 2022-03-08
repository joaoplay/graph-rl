import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from agents.graph_dqn_agent import GraphDQNAgent
from environments.generator.single_vessel_graph_generator import SingleVesselGraphGenerator
from environments.graph_env import GraphEnv
from util import str_to_class

DEFAULT_PARAMS = {
    'batch_size': 50,
    'optimizer': 'Adam',
    'embedding_dim': 50,
    'hidden_output_dim': 50,
    'learning_rate': 0.05
}


def parse_stop_conditions(stop_conditions_list):
    stop_condition_instances = []
    for stop_condition in stop_conditions_list:
        stop_condition_dict = OmegaConf.to_object(stop_condition)
        cls = str_to_class("environments.stop_conditions", stop_condition_dict.pop('name'))
        instance = cls(**stop_condition_dict)
        stop_condition_instances += [instance]

    return stop_condition_instances


@hydra.main(config_path="configs", config_name="default_config")
def run_from_config_file(cfg: DictConfig):
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    print(f"Checking config")

    print("Numpy")
    print(np.show_config())
    print("Torch Config")
    print(torch.__config__.show())
    print("Torch Parallel")
    print(torch.__config__.parallel_info())

    graph_generator = SingleVesselGraphGenerator(**cfg.environment)
    train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)

    validation_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)

    parsed_stop_conditions = parse_stop_conditions(cfg.stop_conditions)

    environment = GraphEnv(stop_conditions=parsed_stop_conditions, stop_after_void_action=cfg.stop_after_void_action,
                           max_edges_percentage=cfg.max_edges_percentage)
    agent = GraphDQNAgent(environment=environment, start_node_selection_dqn_params=cfg.start_node_selection_dqn,
                          end_node_selection_dqn_params=cfg.end_node_selection_dqn, **cfg.core, **cfg.exploratory_actions)

    agent.train(train_graphs, validation_graphs, cfg.max_steps)


if __name__ == '__main__':
    run_from_config_file()
