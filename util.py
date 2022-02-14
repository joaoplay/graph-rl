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
    nx.draw(networkx_graph, coordinates, ax)
