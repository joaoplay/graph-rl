import numpy as np
from matplotlib import pyplot as plt

from environments.generator.vascular_network_from_file_generator import VasculatureNetworkFromFileGenerator
from o2calculator.calculator import calculate_network_irrigation
from settings import BASE_PATH
from util import draw_nx_irrigation_network


def validate_reward():
    gen = VasculatureNetworkFromFileGenerator(BASE_PATH + '/environments/graph_examples/functional2.yml')
    graph = gen.generate()

    prepared_data = graph.prepare_for_reward_evaluation()

    irrigation, sources, pressures, edges_source, edges_list = calculate_network_irrigation(prepared_data[0],
                                                                                            prepared_data[1],
                                                                                            prepared_data[2], [10, 10],
                                                                                            [0.1, 0.1])

    fig_irrigation, ax_irrigation = plt.subplots()
    ax_irrigation.imshow(np.flipud(irrigation), cmap='hot', vmin=0,
                         interpolation='nearest')

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_nx_irrigation_network(prepared_data[3], pressures,
                               edges_source, edges_list, ax)

    plt.show()


if __name__ == '__main__':
    validate_reward()
