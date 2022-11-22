import importlib
import logging

import bezier
import networkx as nx
import numpy as np


def merge_dicts(dict1, dict2):
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [value, dict1[key]]
    return dict3


def str_to_class(module_name, class_name):
    """Return a class instance from a string reference"""
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            logging.error('Class does not exist')
    except ImportError:
        logging.error('Module does not exist')
    return class_ or None


def draw_nx_graph_with_coordinates(networkx_graph, ax):
    node_x = nx.get_node_attributes(networkx_graph, 'x')
    node_y = nx.get_node_attributes(networkx_graph, 'y')
    coordinates = merge_dicts(node_x, node_y)
    nx.draw(networkx_graph, coordinates, ax, connectionstyle="arc3,rad=0.1", arrowsize=1, with_labels=True)


def draw_nx_irrigation_network(networkx_graph, pressures, edges_flow, edges, ax):
    node_x = nx.get_node_attributes(networkx_graph, 'x')
    node_y = nx.get_node_attributes(networkx_graph, 'y')
    coordinates = merge_dicts(node_x, node_y)

    pressures_by_idx = {idx: round(pressure, 3) for idx, pressure in enumerate(pressures)}
    edges_flow = {(edges[0][edge_idx], edges[1][edge_idx]): round(edges_flow[edge_idx], 3) for edge_idx in range(len(edges[0]))}

    options = {"edgecolors": "tab:gray", "node_size": 20, "alpha": 0.5}
    nx.draw_networkx_nodes(networkx_graph, coordinates, ax=ax, node_color="tab:red", **options)

    nx.draw_networkx_edges(networkx_graph, coordinates, width=3, alpha=0.5, edge_color="tab:red", ax=ax)

    nx.draw_networkx_labels(networkx_graph, coordinates, pressures_by_idx, font_size=8, font_color="black", ax=ax)

    nx.draw_networkx_edge_labels(networkx_graph, coordinates, edge_labels=edges_flow, font_color='black', font_size=8,
                                 bbox=dict(alpha=0), label_pos=0.4, ax=ax)


def curved_edges(G, pos, dist_ratio=0.2, bezier_precision=20, polarity='random'):
    # Get nodes into np array
    edges = np.array(G.edges())
    l = edges.shape[0]

    if polarity == 'random':
        # Random polarity of curve
        rnd = np.where(np.random.randint(2, size=l) == 0, -1, 1)
    else:
        # Create a fixed (hashed) polarity column in the case we use fixed polarity
        # This is useful, e.g., for animations
        rnd = np.where(np.mod(np.vectorize(hash)(edges[:, 0]) + np.vectorize(hash)(edges[:, 1]), 2) == 0, -1, 1)

    # Coordinates (x,y) of both nodes for each edge
    # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
    u, inv = np.unique(edges, return_inverse=True)
    coords = np.array([pos[x] for x in u])[inv].reshape([edges.shape[0], 2, edges.shape[1]])
    coords_node1 = coords[:, 0, :]
    coords_node2 = coords[:, 1, :]

    # Swap node1/node2 allocations to make sure the directionality works correctly
    should_swap = coords_node1[:, 0] > coords_node2[:, 0]
    coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]

    # Distance for control points
    dist = dist_ratio * np.sqrt(np.sum((coords_node1 - coords_node2) ** 2, axis=1))

    # Gradients of line connecting node & perpendicular
    m1 = (coords_node2[:, 1] - coords_node1[:, 1]) / (coords_node2[:, 0] - coords_node1[:, 0])
    m2 = -1 / m1

    # Temporary points along the line which connects two nodes
    # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    t1 = dist / np.sqrt(1 + m1 ** 2)
    v1 = np.array([np.ones(l), m1])
    coords_node1_displace = coords_node1 + (v1 * t1).T
    coords_node2_displace = coords_node2 - (v1 * t1).T

    # Control points, same distance but along perpendicular line
    # rnd gives the 'polarity' to determine which side of the line the curve should arc
    t2 = dist / np.sqrt(1 + m2 ** 2)
    v2 = np.array([np.ones(len(edges)), m2])
    coords_node1_ctrl = coords_node1_displace + (rnd * v2 * t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd * v2 * t2).T

    # Combine all these four (x,y) columns into a 'node matrix'
    node_matrix = np.array([coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

    # Create the Bezier curves and store them in a list
    curveplots = []
    for i in range(l):
        nodes = node_matrix[:, i, :].T
        curveplots.append(bezier.Curve(nodes, degree=2).evaluate_multi(np.linspace(0, 1, bezier_precision)).T)

    # Return an array of these curves
    curves = np.array(curveplots)
    return curves
