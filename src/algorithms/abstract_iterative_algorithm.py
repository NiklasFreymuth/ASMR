from abc import ABC, abstractmethod

import plotly.graph_objects as go

from util.types import *


class AbstractIterativeAlgorithm(ABC):
    def __init__(self, config: ConfigDict) -> None:
        """
        Initializes the iterative algorithm using the full config used for the experiment.
        Args:
            config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.
        Returns:

        """
        self._config = config

    @abstractmethod
    def fit_and_evaluate(self) -> ValueDict:
        """
        Trains the algorithm for a single iteration, evaluates its performance and subsequently organizes and provides
        metrics, losses, plots etc. of the fit and evaluation
        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def fit_iteration(self) -> ValueDict:
        """
        Train your algorithm for a single iteration. This can e.g., be a single epoch of neural network training,
        a policy update step, or something more complex. Just see this as the outermost for-loop of your algorithm.

        Returns: May return an optional dictionary of values produced during the fit. These may e.g., be statistics
        of the fit such as a training loss.

        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> ValueDict:
        """
        Evaluate given input data and potentially auxiliary information to create a dictionary of resulting values.
        What kind of things are scored/evaluated depends on the concrete algorithm.
        Args:

        Returns: A dictionary with different values that are evaluated from the given input data. May e.g., the
        accuracy of the model.

        """
        raise NotImplementedError

    def save_checkpoint(self, directory: str, iteration: Optional[int] = None,
                        is_final_save: bool = False, is_initial_save: bool = False) -> None:
        """
        This method causes the algorithm to checkpoint its current state. A checkpoint consists of all information
         needed to re-instantiate the instance of the algorithm at the given point in time. This can be e.g., the
         architecture, weights and optimizer state for neural networks
        Args:
            directory: Directory to save the checkpoint to
            iteration: Current iteration of the algorithm at the point of checkpointing
            is_final_save: Whether this is the last save for this trial
            is_initial_save: Whether this is the first save for this trial

        Returns:

        """
        raise NotImplementedError("AbstractIterativeAlgorithm does not implement save_checkpoint()")

    def load_from_checkpoint(self, checkpoint_config: ConfigDict):
        """
        Loads the algorithm state from the given checkpoint path/experiment configuration name.
        May be used at the start of the algorithm to resume training.
        Args:
            checkpoint_config: Dictionary containing the configuration of the checkpoint to load. Includes
               checkpoint_path: Path to a checkpoint folder of a previous execution of the same algorithm
                iteration: (iOptional[int]) The iteration to load. If not provided, will load the last available iter
                repetition: (int) The algorithm repetition/seed to load. If not provided, will load the first repetition

        Returns:

        """
        raise NotImplementedError("AbstractIterativeAlgorithm does not implement load_from_checkpoint()")

    @property
    def config(self) -> ConfigDict:
        return self._config

    ####################
    # additional plots #
    ####################

    def additional_plots(self, iteration: int) -> Dict[Key, go.Figure]:
        """
        May provide arbitrary functions here that are used to draw additional plots.
        Args:
            iteration: The algorithm iteration this function was called at
        Returns: A dictionary of {plot_name: plot}, where plot_function is any function that takes
          this algorithm at a current point as an argument, and returns a plotly figure.

        """
        raise NotImplementedError("AbstractIterativeAlgorithm does not implement additional_plots()")

    def get_final_values(self) -> ValueDict:
        """
        Returns a dictionary of values that are to be stored as final values of the algorithm.
        Returns:

        """
        raise NotImplementedError("AbstractIterativeAlgorithm does not implement get_final_values()")