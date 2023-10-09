"""
Basic PyTorch Network. Defines some often-needed utility functions. Can be instantiated by child classes
"""
import abc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from util.save_and_load.save_and_load_keys import NETWORK_KWARGS_FILE, TORCH_SAVE_FILE, ARCHITECTURE, OPTIMIZERS, \
    SCHEDULER
from util.types import *


class AbstractArchitecture(nn.Module, abc.ABC):

    def __init__(self, use_gpu: bool = False, **kwargs):
        """
        Basic Network initialization. Makes no assumption about the kind of network that is created, but handles
          gpu utilization, loading and saving
        Args:
            use_gpu: Whether to use a gpu for this architecture or not.
            kwargs: Other arguments passed by inheriting architectures.
              Will be saved in self._kwargs for save/load utility
        """
        super().__init__()
        self._kwargs = locals()
        self._kwargs["type"] = type(self)
        del self._kwargs["self"]
        del self._kwargs["__class__"]

        if use_gpu:
            self._gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._gpu_device = None

    def _initialize_optimizer_and_scheduler(self, training_config: ConfigDict) -> None:
        """
        Initializes the optimizer and learning rate scheduler for this network/architecture.
        Args:
            training_config: Config describing the details about the training process.
            Should contain the following keys:
                {optimizer - The optimizer to use. Should be a string that can be passed to get_optimizer,
                 learning_rate - The learning rate to use,
                 l2_norm - The l2 norm to use for regularization,
                 lr_scheduling_rate - The rate at which to decay the learning rate. If None, no decay is used.}

        Returns: None

        """
        from util.torch_util.torch_util import get_optimizer
        optimizer = get_optimizer(training_config.get("optimizer", "adam"))
        self._optimizer = optimizer(self.parameters(),
                                    lr=training_config.get("learning_rate"),
                                    weight_decay=training_config.get("l2_norm", 0.0))

        lr_scheduling_rate = training_config.get("lr_scheduling_rate")
        if lr_scheduling_rate is not None and lr_scheduling_rate < 1:
            self._learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self._optimizer,
                                                                             gamma=lr_scheduling_rate)
        else:
            self._learning_rate_scheduler = None

    def to_gpu(self, tensor: InputBatch) -> InputBatch:
        if tensor is not None and self._gpu_device:
            if isinstance(tensor, BaseData):
                return tensor.to(device=self._gpu_device)
            elif isinstance(tensor, Tensor) and not tensor.is_cuda:
                return tensor.to(device=self._gpu_device)
        return tensor

    def save(self, destination_folder: Path, file_index: Union[int, str],
             save_kwargs: bool = False) -> Path:
        """
        Saves this network class.
        This always includes the state_dict of the network, i.e., the model parameters.
        If save_kwargs, then the kwargs of this class will also be saved. These are needed to reconstruct the model
        architecture, and to subsequently load the parameters into it when loading the model.
        Usually, the kwargs will only be needed to save once per trial, while the parameters can be checkpointed in
        regular intervals. This is handled by the recording.
        Args:
            destination_folder: The path to save the architeture and optimizer state_dicts
              (and potentially the kwargs) to
            file_index: Indexing of the file. Can e.g., correspond to the algorithm iteration to save at. Can also be
              "final" for the last checkpoint
            save_kwargs: Whether to save the kwargs of the network class or not. These are used to reconstruct the
              class when loading the network, i.e., to recreate the original class constructor call when loading a
              network in.
        Returns: The path to the saved file as a pathlib.Path object

        """
        assert isinstance(file_index, (int, str)), f"Need to provide a file index, got '{file_index}'"
        from util.save_and_load.save_and_load_utility import save_dict

        if save_kwargs:
            save_dict(path=destination_folder, file_name=NETWORK_KWARGS_FILE,
                      dict_to_save=self.kwargs, save_type="npz")

        if isinstance(file_index, int):
            torch_save_path = destination_folder / f"{TORCH_SAVE_FILE}{file_index:04d}.pt"
        else:  # file index is a string
            torch_save_path = destination_folder / f"{TORCH_SAVE_FILE}_{file_index}.pt"

        torch.save({ARCHITECTURE: self.state_dict(),
                    OPTIMIZERS: [optimizer.state_dict() for optimizer in self.optimizers],
                    SCHEDULER: self._learning_rate_scheduler.state_dict() if self._learning_rate_scheduler else None},
                   torch_save_path)
        return torch_save_path

    def _load(self, load_path: Path) -> None:
        """
        Loads a checkpoint for the architecture. This checkpoint contains the state for the architecture itself, as
        well as the state dicts for all optimizers it uses.
        Args:
            load_path: Path to load the .tar checkpoint from

        Returns:

        """
        if load_path.suffix != ".pt":
            load_path = load_path.with_suffix(".pt")
        checkpoint = torch.load(load_path)
        self.eval()  # needs to be here to have dropout etc. consistent
        for optimizer, optimizer_state_dict in zip(self.optimizers, checkpoint.get(OPTIMIZERS)):
            # load the optimizers
            optimizer.load_state_dict(optimizer_state_dict)

        if self._learning_rate_scheduler is not None and checkpoint.get(SCHEDULER) is not None:
            # load the learning rate scheduler
            self._learning_rate_scheduler.load_state_dict(checkpoint.get(SCHEDULER))
        self.load_state_dict(checkpoint.get(ARCHITECTURE))

    def forward(self, tensor: torch.Tensor, **kwargs):
        raise NotImplemented("Network baseclass does not implement method 'forward'")

    @property
    def kwargs(self) -> ConfigDict:
        return self._kwargs

    @property
    def optimizers(self) -> List[optim.Optimizer]:
        return [self._optimizer]

    @property
    def learning_rate_scheduler(self) -> Optional:
        return self._learning_rate_scheduler

    @property
    def gpu_device(self) -> Optional[torch.device]:
        return self._gpu_device

    @staticmethod
    def load_from_path(state_dict_path: Path, network_kwargs: Optional[Union[ConfigDict, str]] = None,
                       **kwargs) -> "AbstractArchitecture":
        """
        Loads a network from the given specified path using a statedict and provided kwargs.
        Args:
            state_dict_path: Path to the state_dict.
            network_kwargs: (Optional) If provided, this is either a dictionary of the network kwargs,
              or a path pointing towards a dictionary save file of the kwargs. If not, the kwargs file is assumed to be
              located in the same filter as the state dict
            **kwargs: (Optional) If provided, additional keyword arguments may be given to the loaded architecture.
              These can be arbitrary arguments that the architecture expects in its init

        Returns: The loaded architecture

        """
        from util.save_and_load.save_and_load_utility import load_dict, undo_numpy_conversions
        if network_kwargs is None:  # assume that the network kwargs are in the same folder as the state_dict
            from util.save_and_load.save_and_load_keys import NETWORK_KWARGS_FILE
            network_kwargs_path = state_dict_path.parent.joinpath(NETWORK_KWARGS_FILE)
            network_kwargs = str(network_kwargs_path)
        network_kwargs = load_dict(network_kwargs)

        architecture_type = network_kwargs.get("type").item()
        del network_kwargs["type"]
        assert issubclass(architecture_type,
                          AbstractArchitecture), f"Must inherit from AbstractArchitecture base class, " \
                                                 f"given {architecture_type} instead"

        network_kwargs = undo_numpy_conversions(network_kwargs)
        algorithm_kwargs = network_kwargs.get('kwargs')
        del network_kwargs['kwargs']

        # instantiate architecture class of the given type and load the state dict and optimizers for it
        architecture: AbstractArchitecture = architecture_type(**network_kwargs,  # network params like latent dim etc.
                                                               **kwargs,  # use to pass an environment
                                                               **algorithm_kwargs  # algo params
                                                               )
        architecture._load(load_path=state_dict_path)
        return architecture
