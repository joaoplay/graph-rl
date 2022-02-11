from matplotlib import pyplot as plt

from environments.stop_conditions.all_graphs_exhausted import AllGraphsExhausted
from environments.stop_conditions.max_steps_exceeded import MaxStepsExceeded
from settings import NEPTUNE_INSTANCE
from agents.graph_dqn_agent import GraphDQNAgent
from environments.generator.single_vessel_graph_generator import SingleVesselGraphGenerator
from environments.graph_env import GraphEnv

DEFAULT_PARAMS = {
    'batch_size': 50,
    'optimizer': 'Adam',
    'embedding_dim': 50,
    'hidden_output_dim': 50,
    'learning_rate': 0.05
}

DEFAULT_STOP_CONDITIONS = [
    AllGraphsExhausted(),
    MaxStepsExceeded(max_steps=100)
]


if __name__ == '__main__':
    NEPTUNE_INSTANCE['config/params'] = DEFAULT_PARAMS

    graph_generator = SingleVesselGraphGenerator(size_x=10, size_y=10, interval_between_nodes=1)
    graphs = graph_generator.generate_multiple_graphs(1)

    environment = GraphEnv(stop_conditions=DEFAULT_STOP_CONDITIONS)

    agent = GraphDQNAgent(environment=environment, validation_interval=2)

    agent.train(graphs, graphs, 20)

    plt.show()
