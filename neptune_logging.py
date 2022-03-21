from neptune.new.types import File

from settings import NEPTUNE_INSTANCE


def log_batch_training_result(loss):
    NEPTUNE_INSTANCE['training/batch/loss'].log(loss)


def log_batch_validation_result(performance):
    NEPTUNE_INSTANCE['validation/batch/peformance'].log(performance)


def log_training_simulation_results(average_reward):
    NEPTUNE_INSTANCE['training/simulation/average_reward'].log(average_reward)


def log_validation_simulation_results(average_reward):
    NEPTUNE_INSTANCE['validation/simulation/average_reward'].log(average_reward)


def upload_graph_plot(plot, iteration):
    NEPTUNE_INSTANCE[f'validation/visualization/graphs/training-step-{iteration}.jpeg'].upload(File.as_image(plot))


def upload_action_frequency(plot, iteration, graph_idx):
    NEPTUNE_INSTANCE[f'validation/visualization/action-frequency/hist-it{iteration}-graph{graph_idx}.jpeg'].upload(
        File.as_image(plot))


def upload_irrigation_heatmaps(sources_heatmap, irrigation_heatmap, iteration, stage):
    NEPTUNE_INSTANCE[f'{stage}/visualization/irrigation_heatmap/sources-heatmap-{iteration}.jpeg'].upload(
        File.as_image(sources_heatmap))
    NEPTUNE_INSTANCE[f'{stage}/visualization/irrigation_heatmap/irrigation-heatmap-{iteration}.jpeg'].upload(
        File.as_image(irrigation_heatmap))


def log_training_instant_mean_reward(mean_reward):
    NEPTUNE_INSTANCE[f'training/instant_reward'].log(mean_reward)


def log_training_mean_reward(mean_reward):
    NEPTUNE_INSTANCE[f'training/reward'].log(mean_reward)


def upload_action_selection(node_selection_plot, iteration):
    NEPTUNE_INSTANCE[f'validation/visualization/node-selection/bar-it{iteration}.jpeg'].upload(
        File.as_image(node_selection_plot))