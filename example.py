import numpy as np
import torch

from src.algorithms.get_algorithm import get_algorithm
from src.recording.recorder import Recorder
from util.types import *


def get_environment_config(num_refinements: int = 6) -> ConfigDict:
    """
    The task config defines the task/environment. More specifically, it defines the mesh refinement environment,
    including the FEM parameters and the mesh refinement parameters such as the element penalty and the number of
    refinements.
    """
    environment_config = {"environment": "mesh_refinement",
                          "mesh_refinement":
                              {"num_timesteps": num_refinements,
                               "element_penalty":
                                   {"value": 3.0e-2},
                               "element_limit_penalty": 1000,
                               "maximum_elements": 20000,
                               "refinement_strategy": "absolute_discrete",  # "discrete" or "single_agent"
                               "reward_type": "spatial_area",  # use our area scaling

                               # define the (class of) PDE(s) to solve with the FEM
                               "fem":
                                   {"pde_type": "poisson",
                                    "num_pdes": 100,  # use 100 PDE for the training buffer
                                    "domain":
                                        {"fixed_domain": False,
                                         "domain_type": "lshape",
                                         "num_integration_refinements": num_refinements,
                                         # how many times to refine the initial mesh to get the reference mesh
                                         },
                                    "poisson":
                                        {"fixed_load": False,
                                         "num_components": 3,  # use a Gaussian mixture of 3 components
                                         "element_features":  # poisson-specific element features
                                             {"load_function": True, }
                                         }

                                    },

                               # features to use for the MPN
                               "element_features":
                                   {"x_position": False,
                                    "y_position": False,
                                    "area": True,
                                    "solution_mean": True,
                                    "solution_std": True,
                                    },
                               "edge_features":
                                   {
                                       "distance_vector": False,
                                       "euclidean_distance": True,
                                   }
                               },

                          }
    return environment_config


def get_algorithm_config() -> ConfigDict:
    """
    The algorithm config defines the algorithm, i.e., the RL backbone and the MPN it uses to predict the refinement
    actions.
    """
    algorithm_config = {"name": "ppo",
                        "use_mixed_reward": True,
                        "verbose": True,
                        "batch_size": 32,
                        "discount_factor": 0.99,
                        "use_gpu": False,
                        "sample_buffer_on_gpu": False,
                        "ppo":
                            {"num_rollout_steps": 256,
                             "normalize_observations": True,
                             "normalize_rewards": False,
                             "epochs_per_iteration": 5,
                             "value_function_scope": "spatial"
                             },
                        "network":
                            {"latent_dimension": 64,
                             "base":
                                 {"scatter_reduce": "mean",
                                  "stack":
                                      {"use_layer_norm": True,
                                       "num_steps": 2,
                                       "use_residual_connections": True,
                                       "mlp":
                                           {"activation_function": "leakyrelu",
                                            "num_layers": 2, }
                                       }
                                  },
                             "actor":
                                 {"mlp":
                                      {"activation_function": "tanh",
                                       "num_layers": 2, }
                                  },
                             "critic":
                                 {"mlp":
                                      {"activation_function": "tanh",
                                       "num_layers": 2, }
                                  },
                             "training":
                                 {"learning_rate": 3.0e-4},
                             }
                        }

    return algorithm_config


def get_recording_config() -> ConfigDict:
    """
    The recording config defines how the results are recorded.
    """
    recording_config = {"wandb":
                            {"enabled": True,
                             "plot_frequency": 5,
                             "additional_plots": True,
                             "project_name": "ASMR",
                             "task_name": "Example",
                             "tags": ["asmr"],
                             "start_method": "thread",
                             },
                        "checkpoint": True,
                        "checkpoint_frequency": 5}
    return recording_config


def get_recording_structure():
    return {"_groupname": "asmr_examples",
            "_runname": "PoissonExample",
            "_recording_dir": "reports/example/",
            "_job_name": "asmr_example"}


def get_config(num_refinements: int = 6):
    """
    Params:
        num_refinements: The number of mesh refinements to perform. This is the number of timesteps in the mesh
            refinement environment, and the number of uniform refinements of the reference
    The config consists of 3 parts:
    1. The algorithm section, which defines the algorithm
    2. The task section, which defines the task/environment
    3. The recording section, which defines how the results are recorded. This contains a recording structure, which
         defines the directory structure of the recording.
    Returns:

    """
    config = {"environment": get_environment_config(num_refinements=num_refinements),
              "algorithm": get_algorithm_config(),
              "recording": get_recording_config(),
              "_recording_structure": get_recording_structure()}
    return config


def main(iterations: int = 100, num_refinements: int = 6, seed: int = 123):
    """
    An example of how to run the ASMR algorithm. This is a simple example, which uses default configurations for
    a PPO RL backbone and the Poisson task.
    For more advanced examples, see configs/asmr/tests.yaml, which you can execute with
        python main.py configs/asmr/tests.yaml -e $experiment_name -o
    where $experiment_name is any of the names of the experiments in the yaml file.

    The experiments of the paper, including baselines and ablations,
    are all listed in the configs/asmr/ folder and sorted by their respective task.
    Args:
        iterations: The number of iterations to run the algorithm for
        num_refinements: How many refinement steps to use. We use 6 in the paper,
            but fewer are possible for faster convergence and testing.
        seed: The random seed to use for the experiment

    Returns:

    """
    config = get_config(num_refinements=num_refinements)

    # initialize random seeds
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    # initialize the actual algorithm, which includes the environment, and the recording
    algorithm = get_algorithm(config=config,
                              seed=seed)
    recorder = Recorder(config=config, algorithm=algorithm)

    for current_iteration in range(iterations):
        # fit and evaluate the algorithm for a single episode
        recorded_values = algorithm.fit_and_evaluate()

        # record the results of this iteration
        scalars = recorder.record_iteration(iteration=current_iteration, recorded_values=recorded_values)
        # the scalars are returned here, but automatically logged to wandb. See the console outputs


if __name__ == '__main__':
    main(num_refinements=4)  # we use 4 refinements for the example to make it faster. The paper experiments use 6.
