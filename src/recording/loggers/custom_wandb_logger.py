import os
import pandas as pd
import wandb

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.recording.loggers.abstract_logger import AbstractLogger
from util import keys
from util.function import get_from_nested_dict
from util.types import *


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


class CustomWAndBLogger(AbstractLogger):
    """
    Logs (some) recorded results using wandb.ai.
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm):
        super().__init__(config=config, algorithm=algorithm)
        wandb_params = get_from_nested_dict(config, list_of_keys=["recording", "wandb"], raise_error=True)
        self.wandb_plot_log_frequency = wandb_params.get("plot_frequency")
        self.video_frequency = wandb_params.get("video_frequency")
        if self.video_frequency is None:
            self.video_frequency = self.wandb_plot_log_frequency
        self.record_additional_plots = wandb_params.get("additional_plots")
        project_name = wandb_params.get("project_name")
        environment_name = get_from_nested_dict(config, list_of_keys=["environment", "environment"],
                                                default_return=None)

        if wandb_params.get("task_name") is not None:
            project_name = project_name + "_" + wandb_params.get("task_name")
        elif environment_name is not None:
            project_name = project_name + "_" + environment_name

        recording_structure = config.get("_recording_structure")
        groupname = recording_structure.get("_groupname")[-127:]
        runname = recording_structure.get("_runname")[-127:]
        job_name = recording_structure.get("_job_name")

        # Check if the environment variable exists
        if 'WANDB_DIR' in os.environ and wandb_params.get("use_env_wandb_dir", False):
            recording_dir = os.environ['WANDB_DIR']
        else:
            recording_dir = recording_structure.get("_recording_dir")

        tags = wandb_params.get("tags", [])
        if tags is None:
            tags = []
        if get_from_nested_dict(config, list_of_keys=["algorithm", "name"], default_return=None) is not None:
            tags.append(get_from_nested_dict(config, list_of_keys=["algorithm", "name"]))

        reset_wandb_env()
        entity = wandb_params.get("entity")

        start_method = wandb_params.get("start_method")
        settings = wandb.Settings(start_method=start_method) if start_method is not None else None

        for i in range(10):
            try:
                self.wandb_logger = wandb.init(project=project_name,  # name of the whole project
                                               tags=tags,
                                               # tags to search the runs by. Currently, contains algorithm name
                                               job_type=job_name,  # name of your experiment
                                               group=groupname,
                                               # group of identical hyperparameters for different seeds
                                               name=runname,  # individual repetitions
                                               dir=recording_dir,  # local directory for wandb recording
                                               config=config,  # full file config
                                               reinit=False,
                                               entity=entity,
                                               settings=settings
                                               )
                break
            except Exception as e:
                from time import sleep
                from random import random
                import warnings
                # implement a simple randomized exponential backoff if starting a run fails
                waiting_time = ((random() / 50) + 0.01) * (2 ** i)
                # wait between 0.01 and 10.24 seconds depending on the random seed and the iteration of the exponent

                warnings.warn("Problem with starting wandb: {}. Trying again in {} seconds".format(e,
                                                                                                   waiting_time))
                sleep(waiting_time)
                if i == 9:
                    warnings.warn("wandb init failed several times.")
                    raise e

    def log_iteration(self, recorded_values: ValueDict, iteration: int) -> None:
        """
        Parses and logs the given dict of recorder metrics to wandb.
        Args:
            recorded_values: A dictionary of previously recorded things
            iteration: The current iteration of the algorithm
        Returns:

        """
        wandb_log_dict = {}
        # log_scalars
        if keys.SCALARS in recorded_values:
            wandb_log_dict[keys.SCALARS] = recorded_values.get(keys.SCALARS)
            wandb_log_dict['iteration'] = iteration

        log_plots = self.wandb_plot_log_frequency > 0 and iteration % self.wandb_plot_log_frequency == 0
        log_video = self.wandb_plot_log_frequency > 0 and iteration % self.video_frequency == 0
        if log_plots:
            # record plots in this iteration
            if keys.FIGURES in recorded_values:
                figures = recorded_values.get(keys.FIGURES)
                if isinstance(figures, list):  # multiple figures in a list
                    wandb_log_dict[keys.FIGURES] = {f"{position}_{keys.FIGURES}": figure
                                                    for position, figure in enumerate(figures)}
                else:  # single figure
                    wandb_log_dict[keys.FIGURES] = {keys.FIGURES: figures}
            if keys.VIDEO_ARRAYS in recorded_values and log_video:
                video_arrays = recorded_values.get(keys.VIDEO_ARRAYS)
                if isinstance(video_arrays, list):  # multiple figures in a list
                    wandb_log_dict[keys.VIDEO_ARRAYS] = {f"{position}_{keys.VIDEO_ARRAYS}": wandb.Video(video_array,
                                                                                                        format="mp4",
                                                                                                        fps=16)
                                                         for position, video_array in enumerate(video_arrays)}
                else:  # single figure
                    wandb_log_dict[keys.VIDEO_ARRAYS] = {keys.VIDEO_ARRAYS: video_arrays}
            if self.record_additional_plots:
                additional_plots = self._algorithm.additional_plots(iteration=iteration)
                wandb_log_dict[keys.ADDITIONAL_PLOTS] = additional_plots
        if wandb_log_dict:  # logging dictionary is not empty
            self.wandb_logger.log(wandb_log_dict)

    def finalize(self, final_values: ValueDict) -> None:
        """
        Properly close the wandb logger
        Returns:

        """
        if keys.TABLES in final_values:
            tables = final_values.get(keys.TABLES)
            for table_name, table in tables.items():
                if isinstance(table.index, pd.MultiIndex):
                    # Group by the first level of the multi-index and compute the mean
                    # This effectively aggregates over the 2nd axis of the original experiment results

                    self.wandb_logger.log({f"Pareto/{table_name}": table.groupby(level=0).mean()})
                else:
                    self.wandb_logger.log({f"Pareto/{table_name}": table})
        self.wandb_logger.finish()
