import numpy as np


class BatchSampler:

    def __init__(self, data, batch_size) -> None:
        super().__init__()
        self.current_index = 0
        self.sample_indices = list(range(len(data)))
        self.batch_size = batch_size
        self.data = data

    def sample(self):
        if (self.current_index + 1) * self.batch_size > len(self.sample_indices):
            self.current_index = 0
            np.random.shuffle(self.sample_indices)

        selected_idx = self.sample_indices[
                       self.current_index * self.batch_size:(self.current_index + 1) * self.batch_size]
        self.current_index += 1
        return selected_idx
