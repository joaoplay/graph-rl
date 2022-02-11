import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class GraphStateDataset(Dataset):

    def __init__(self, graphs_state) -> None:
        super().__init__()
        self.graphs_state = graphs_state

    def __len__(self):
        return len(self.graphs_state)

    def __getitem__(self, index) -> T_co:
        if torch.is_tensor(index):
            index = index.tolist()

        return self.graphs_state[index]
