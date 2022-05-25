import torch


class GraphNode:

    def __init__(self, unique_id: int, coordinates: tuple):
        super(GraphNode, self).__init__()

        self.unique_id = unique_id
        self.coordinates = coordinates

    def get_features_dict(self):
        return {
            'x': self.coordinates[0],  # Coordinate X
            'y': self.coordinates[1],  # Coordinate Y
        }
