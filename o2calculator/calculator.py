import logging
import time
from collections import OrderedDict
from functools import cache

import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.ma import sqrt

from settings import BASE_PATH, USE_CUDA


def remove_duplicated_edges(edges_list):
    """
    Remove duplicated edges in graphs with bi-directional edges.
    :param edges_list: An edge list with shape (2, num_edges). [
                                                                  [edge_1_start_node_id, edge_2_start_node_id]
                                                                  [edge_1_end_node_id, edge_2_end_node_id]
                                                                ]
    :return: Return a new edge list with the same shape as edges_list input. Duplicated edges are excluded.
    """

    duplicated_free_start_nodes = []
    duplicated_free_end_nodes = []

    visited_edges = set()
    for edge_idx in range(len(edges_list[0])):
        start_node_idx = edges_list[0][edge_idx]
        end_node_idx = edges_list[1][edge_idx]

        if (start_node_idx, end_node_idx) in visited_edges:
            # This edge is duplicated. Probably a bidirectional edge
            continue

        # Add this edge to the final output
        duplicated_free_start_nodes += [edges_list[0][edge_idx]]
        duplicated_free_end_nodes += [edges_list[1][edge_idx]]

        # Add both pairs to the visited set.
        visited_edges.add((start_node_idx, end_node_idx))
        visited_edges.add((end_node_idx, start_node_idx))

    return [duplicated_free_start_nodes, duplicated_free_end_nodes]


def convert_edges_list_to_node_neighbourhood_map(edges_list, edges_features):
    """
    Converts an edge list into a
    :param edges_list: An edge list with shape (2, num_edges). [
                                                                    [edge_1_start_node_id, edge_2_start_node_id]
                                                                    [edge_1_end_node_id, edge_2_end_node_id]
                                                                  ]
    :param edges_features: A list with shape (num_edges, num_edge_features)
    :return: Returns an ordered map containing node IDs as keys (by ascending order) and a list of neighbour node IDs
             with corresponding edge features.
             {
                0: [ [1, [0.45, 0.32]], [2, [0.12, 0.34]]]
                1: [ [0, [0.45, 0.32]], [2, [0.12, 0.34]]]
                2: [ [0, [0.45, 0.32]], [1, [0.12, 0.34]]]
             }
    """

    # Make sure that node IDs are in ascending order.
    neighbour_edges_map = {}

    start_nodes = edges_list[0]
    end_nodes = edges_list[1]

    for i in range(len(start_nodes)):
        current_node = start_nodes[i]
        if current_node not in neighbour_edges_map:
            # This is a new node. Init it in the map
            neighbour_edges_map[current_node] = []

        # Add
        neighbour_edges_map[current_node].append([end_nodes[i], edges_features[i]])

    neighbour_edges_map = OrderedDict(sorted(neighbour_edges_map.items()))

    return neighbour_edges_map


def build_pressures_matrix(node_features: list[list[float]], edges: list[list[int]], edges_features: list[list[float]]):
    """
    Builds a matrix o pressures (FIXME: This is not the right name for this matrix. Ask Rui for the proper name)
    :param node_features: A list with shape (num_nodes, num_node_features) containing the the features for each node
    :param edges: An edge list with shape (2, num_edges). [
                                                            [edge_1_start_node_id, edge_2_start_node_id]
                                                            [edge_1_end_node_id, edge_2_end_node_id]
                                                           ]
    :param edges_features:
    :return:
    """
    n_nodes = len(node_features)
    matrix = []

    # Convert edges list to a "nodes neighbourhood" representation
    node_neighbours_map = convert_edges_list_to_node_neighbourhood_map(edges, edges_features)

    for node_idx, neighbour_edges in node_neighbours_map.items():
        node = node_features[node_idx]

        if node[4] != 0:
            # The current node is input or output. Corresponding pressures are constant, as defined in the
            # feature's map
            row = [0] * n_nodes
            # Every element in the row is 0 except the diagonal
            row[node_idx] = 1
            matrix += [row]

            continue

        node_row = [0] * n_nodes
        # Calculate the value in the diagonal
        node_row[node_idx] = sum([-1 / edge_features[1][0] for edge_features in neighbour_edges])

        for edge in neighbour_edges:
            # Calculate value for position (node_idx, edge.end_node.index)
            node_row[edge[0]] = 1 / edge[1][0]

        matrix += [node_row]

    return matrix


def build_edges_source(edges, edges_features, pressures):
    """
    Calculates the "source" of each edge in the edges list. An array with the pressures of each node must be provided.
    :param edges: An edge list with shape (2, num_edges). [
                                                            [edge_1_start_node_id, edge_2_start_node_id]
                                                            [edge_1_end_node_id, edge_2_end_node_id]
                                                           ]
    :param edges_features:
    :param pressures:
    :return: A list with shape (num_edges, 1) with the "source" value for each edge
    """
    edges_source = []

    # Iterate over edges
    for edge_idx in range(len(edges[0])):
        # Select start and end nodes
        start_node_idx = edges[0][edge_idx]
        end_node_idx = edges[1][edge_idx]

        # Get pressures of start and end nodes
        start_node_p = pressures[start_node_idx]
        end_node_p = pressures[end_node_idx]

        # Calculate Q of the current edge
        edge_q = (start_node_p - end_node_p) / edges_features[edge_idx][0]

        edge_source = abs(edge_q) * ((start_node_p + end_node_p) / 2)

        edges_source += [edge_source]

    return edges_source


def build_source(edges, nodes_features, edges_source, n_cells, cell_size):
    """
    Build source 3D map with source values. The number of cells in each dimension is determined by Lx, ly and Lz.
    Note that each cell is not necessarily a cube.
    :param cell_size:
    :param n_cells:
    :param edges: An edge list with shape (2, num_edges). [
                                                            [edge_1_start_node_id, edge_2_start_node_id]
                                                            [edge_1_end_node_id, edge_2_end_node_id]
                                                          ]
    :param nodes_features: A list with shape (num_nodes, num_node_features) containing the feature set for each node
    :param edges_source: A vector with dimension (num_edges, 1). It is expected the values from build_edges_source method
    :return:
    """
    # Init source matrix with zeros
    source = np.zeros(n_cells)

    n_dims = len(n_cells)

    # Iterate over every edge
    for edge_idx in range(len(edges[0])):
        start_node_idx = edges[0][edge_idx]
        end_node_idx = edges[1][edge_idx]

        start_node_features = nodes_features[start_node_idx]
        end_node_features = nodes_features[end_node_idx]

        # Calculate the start and ending cell. It will be useful to iterate over all cells wherein the current edge is
        # travelling
        min_coordinates = np.minimum(start_node_features[0:n_dims], end_node_features[0:n_dims], dtype=float)
        max_coordinates = np.maximum(start_node_features[0:n_dims], end_node_features[0:n_dims], dtype=float)

        min_cell = np.divide(min_coordinates, cell_size, out=np.zeros_like(min_coordinates), where=cell_size != 0) \
            .astype(int)
        max_cell = np.divide(max_coordinates, cell_size, out=np.zeros_like(max_coordinates), where=cell_size != 0) \
            .astype(int)

        max_cell = np.minimum(max_cell + 1, n_cells)

        start_node_coordinates = start_node_features[0:n_dims].astype(float)
        end_node_coordinates = end_node_features[0:n_dims].astype(float)

        start_node_cell = np.divide(start_node_coordinates, cell_size, out=np.zeros_like(start_node_coordinates), where=cell_size != 0).astype(int)
        end_node_cell = np.divide(end_node_coordinates, cell_size, out=np.zeros_like(end_node_coordinates), where=cell_size != 0).astype(int)

        def find_cells_in_dim(dim):
            t = (np.arange(min_cell[dim], max_cell[dim]) - min_cell[dim]) / (max_cell[dim] - min_cell[dim])
            repeated = np.tile(end_node_cell - start_node_cell, (t.size, 1))
            temp = t.reshape(-1, 1) * repeated
            points = np.add(start_node_cell, temp)

            """for i in range(min_cell[dim], max_cell[dim]):
                t = (i - min_cell[dim]) / (max_cell[dim] - min_cell[dim])
                temp = t * (max_cell - min_cell)
                points += [tuple(np.add(min_cell, temp).astype(int))]"""

            return points

        pointsX = find_cells_in_dim(0).astype(int)
        pointsY = find_cells_in_dim(1).astype(int)

        points_final = np.concatenate([pointsX, pointsY])

        source[points_final[:, 0], points_final[:, 1]] = np.maximum(edges_source[edge_idx], source[points_final[:, 0],
                                                                                                   points_final[:, 1]])

    """fig, ax = plt.subplots()
    ax.imshow(np.fliplr(source), cmap='hot', interpolation='nearest')
    fig.savefig(f'{BASE_PATH}/test_images/sources-{time.time()}.png')"""

    return source


@cache
def build_k2(l_x, l_y):
    """
    Build K2
    :param l_x:
    :param l_y:
    :return:
    """
    kx = np.zeros((l_x, l_y))
    ky = np.zeros((l_x, l_y))

    lx_fourier = np.fft.fftfreq(l_x) * (2 * np.pi)
    ly_fourier = np.fft.fftfreq(l_y) * (2 * np.pi)

    kx[:, np.arange(l_x)] = lx_fourier
    ky[np.arange(l_y), :] = ly_fourier

    """for ix in range(l_x):
        for iy in range(l_y):
            kx[:, iy] = np.fft.fftfreq(l_x) * (2 * np.pi)
            ky[ix, :] = np.fft.fftfreq(l_y) * (2 * np.pi)
    """

    k2 = sqrt(kx ** 2 + ky ** 2)

    return k2


def calc_oxygen(source, k2, oxygen_diff_length):
    """
    Calculate Oxygen
    :param oxygen_diff_length:
    :param source:
    :param k2:
    :return:
    """
    source_k = np.fft.fftn(source)
    o2_k = source_k / (k2 + 1 / oxygen_diff_length)
    o2 = np.fft.ifftn(o2_k)
    return o2


def calculate_pressures(matrix, static_pressures):
    """
    Calculate pressures in each node
    :param matrix:
    :param static_pressures:
    :return:
    """
    return np.dot(matrix, static_pressures)


def calculate_network_irrigation(node_features, edges_list, edges_features, environment_dim, cell_size):
    # Build matrix from adjacency matrix and features
    matrix = build_pressures_matrix(node_features, edges_list, edges_features)
    # Convert matrix to numpy

    pressures_tensor = torch.FloatTensor(matrix)
    if USE_CUDA == 1:
        pressures_tensor = pressures_tensor.cuda()

    # Invert matrix
    try:
        reverse_matrix = torch.linalg.inv(pressures_tensor)
    except RuntimeError:
        logging.error("Not invertible. Assigning reward 0")
        return 0

    reverse_matrix = reverse_matrix.cpu().detach().numpy()

    # Calculate static pressures
    static_pressures = np.array([node[3] for node in node_features])
    # Calculate pressure in each node
    pressures = calculate_pressures(reverse_matrix, static_pressures)

    # Remove duplicated edges
    duplicated_free_edges = remove_duplicated_edges(edges_list)
    # Calculate source for each edge
    edges_source = build_edges_source(edges=duplicated_free_edges, edges_features=edges_features,
                                      pressures=pressures)

    np_environment_dim = np.array(environment_dim)
    number_of_cells = np.divide((np_environment_dim - 1), cell_size).astype(int) + 1

    # Build matrix with number of cells in each dimension defined by L_X, L_Y, L_Z
    sources_by_cell = build_source(np.array(duplicated_free_edges), np.array(node_features), edges_source,
                                   number_of_cells, cell_size)

    # Get k2
    k2 = build_k2(number_of_cells[0], number_of_cells[1])
    # Calculate oxygen in each cell
    oxygen = calc_oxygen(sources_by_cell, k2, 15)
    oxygen = np.real(oxygen)

    return oxygen, sources_by_cell, pressures, edges_source, duplicated_free_edges
