from collections import deque
from typing import Any

import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT


class CumRewardEarlyStopping(EarlyStopping):

    def __init__(self, window_size=5) -> None:
        super().__init__()
        self.window_size = window_size
        self.last_n_cum_rewards = deque(maxlen=window_size)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                           batch: Any, batch_idx: int) -> None:
        is_episode_end = outputs["is_episode_end"]

        if is_episode_end:
            self.last_n_cum_rewards.append(outputs["cum_reward"])

        if len(self.last_n_cum_rewards) == self.window_size:
            std_dev = np.std(self.last_n_cum_rewards)


