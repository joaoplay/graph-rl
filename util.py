import importlib
import logging

import networkx as nx


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


def draw_nx_irrigation_network(networkx_graph, pressures, edges_flow, edges, ax, edge_q):
    node_x = nx.get_node_attributes(networkx_graph, 'x')
    node_y = nx.get_node_attributes(networkx_graph, 'y')
    coordinates = merge_dicts(node_x, node_y)

    pressures_by_idx = {idx: round(pressure, 3) for idx, pressure in enumerate(pressures)}
    #edges_flow = {(edges[0][edge_idx], edges[1][edge_idx]): round(edges_flow[edge_idx], 3) for edge_idx in
    #              range(len(edges[0]))}
    edges_q_with_node = {(edges[0][edge_idx], edges[1][edge_idx]): round(edge_q[edge_idx], 3) for edge_idx in
                         range(len(edges[0]))}

    options = {"edgecolors": "tab:gray", "node_size": 20, "alpha": 0.5}
    nx.draw_networkx_nodes(networkx_graph, coordinates, ax=ax, node_color="tab:red", **options)

    nx.draw_networkx_edges(networkx_graph, coordinates, width=3, alpha=0.5, edge_color="tab:red", ax=ax)

    nx.draw_networkx_labels(networkx_graph, coordinates, pressures_by_idx, font_size=6, font_color="black", ax=ax)

    nx.draw_networkx_edge_labels(networkx_graph, coordinates, edge_labels=edges_q_with_node, font_color='black', font_size=6,
                                 bbox=dict(alpha=0), label_pos=0.4, ax=ax)
