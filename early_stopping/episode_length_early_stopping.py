import torch
from pytorch_lightning import Callback


class EpisodeLengthEarlyStopping(Callback):

    def __init__(self, max_steps_without_improvement=10) -> None:
        super().__init__()
        self.consecutive_validation_steps_without_improvement = 0
        self.previous_score = None
        self.monitor = 'episode-length'
        self.max_steps_without_improvement = max_steps_without_improvement

    def on_validation_end(self, trainer, pl_module):
        if self.previous_score is not None:
            if torch.ge(trainer.callback_metrics[self.monitor], self.previous_score):
                self.consecutive_validation_steps_without_improvement += 1
            else:
                self.consecutive_validation_steps_without_improvement = 0

        if self.consecutive_validation_steps_without_improvement >= self.max_steps_without_improvement:
            trainer.should_stop = True

        self.previous_score = trainer.callback_metrics[self.monitor]
