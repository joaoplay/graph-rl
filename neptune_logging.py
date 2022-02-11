from neptune.new.types import File

from settings import NEPTUNE_INSTANCE


def log_batch_training_result(loss, reward):
    NEPTUNE_INSTANCE['training/batch/loss'].log(loss)
    NEPTUNE_INSTANCE['training/batch/reward'].log(reward)


def log_batch_validation_result(performance):
    NEPTUNE_INSTANCE['validation/batch/peformance'].log(performance)


def upload_graph_plot(plot, iteration):
    NEPTUNE_INSTANCE[f'validation/visualization/graphs/training-step-{iteration}.jpeg'].upload(File.as_image(plot))
