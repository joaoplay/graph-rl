from neptune.new.types import File

from settings import NEPTUNE_INSTANCE


def log_batch_training_result(loss, reward):
    NEPTUNE_INSTANCE['training/batch/loss'].log(loss)
    NEPTUNE_INSTANCE['training/batch/reward'].log(reward)


def log_batch_validation_result(performance):
    NEPTUNE_INSTANCE['validation/batch/peformance'].log(performance)


def log_training_simulation_results(average_reward):
    NEPTUNE_INSTANCE['training/simulation/average_reward'].log(average_reward)


def log_validation_simulation_results(average_reward):
    NEPTUNE_INSTANCE['validation/simulation/average_reward'].log(average_reward)


def upload_graph_plot(plot, iteration):
    NEPTUNE_INSTANCE[f'validation/visualization/graphs/training-step-{iteration}.jpeg'].upload(File.as_image(plot))
