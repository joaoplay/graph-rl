import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from dqn_lightning import DQNLightning
from environments.generator.single_vessel_graph_generator import SingleVesselGraphGenerator
from environments.graph_env import GraphEnv


@hydra.main(config_path="configs", config_name="default_config")
def run_from_config_file(cfg: DictConfig):
    graph_generator = SingleVesselGraphGenerator(**cfg.environment)

    environment = GraphEnv(max_steps=3000, irrigation_goal=0.8)
    train_graphs = graph_generator.generate_multiple_graphs(cfg.number_of_graphs)
    model = DQNLightning(environment, train_graphs, replay_size=10**6)

    trainer = Trainer(
        gpus=[0],
        max_epochs=1000,
        val_check_interval=100,
    )

    trainer.fit(model)


if __name__ == '__main__':
    run_from_config_file()
