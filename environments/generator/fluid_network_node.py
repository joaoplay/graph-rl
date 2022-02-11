from environments.generator.graph_node import GraphNode


class FluidNetworkNode(GraphNode):

    def __init__(self, unique_id: int, coordinates: tuple, z: float, pressure: float, node_type: int):
        super().__init__(unique_id, coordinates)
        self.z = z
        self.pressure = pressure
        self.node_type = node_type

    def get_features_dict(self):
        features_dict = super().get_features_dict()
        features_dict.update({
            'z': self.z,
            'pressure': self.pressure,
            'node_type': self.node_type
        })

        return features_dict



