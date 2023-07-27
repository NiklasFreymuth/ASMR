# ASMR

## Abstract
The Finite Element Method, an important technique in engineering, is aided by Adaptive Mesh Refinement (AMR), which dynamically refines mesh regions to allow for a favorable trade-off between computational speed and simulation accuracy.
Classical methods for AMR depend on task-specific heuristics or expensive error estimators, hindering their use for complex simulations.
Recent _learned_ AMR methods tackle these problems, but so far scale only to simple toy examples. 
We formulate AMR as a novel Adaptive Swarm Markov Decision Process in which a mesh is modeled as a system of simple collaborating agents that may split into multiple new agents.
This framework allows for a spatial reward formulation that simplifies the credit assignment problem, which we combine with Message Passing Networks to propagate information between neighboring mesh elements.
We experimentally validate the effectiveness of our approach, Adaptive Swarm Mesh Refinement (ASMR), showing that it learns reliable, scalable, and efficient refinement strategies on a set of challenging problems.
Our approach improves computation speed by more than tenfold compared to uniform refinements in complex simulations. 
Additionally, we outperform learned baselines and achieve a refinement quality that is on par with a traditional error-based AMR strategy without expensive oracle information about the error signal. 
Swarm Reinforcement Learning for Adaptive Mesh Refinement

# Getting Started

## Setting up the environment

This project uses conda (https://docs.conda.io/en/latest/) and pip for handling packages and dependencies.

To get faster package management performance, you can also use mamba instead of conda (https://github.com/conda-forge/miniforge#mambaforge). Make sure you dont use mamba that is installed as a package using conda.

To install mamba on Linux-like OSes use one of the commands below. 

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

or

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

For Windows please see the documentation in the link above or use (not recommended).
```
conda install -c conda-forge mamba
```

You should be able to install all requirements using the commands below:

```
# for cpu use
mamba env create -f ./env/environment-cpu.yaml
mamba activate ASMR_cpu

# for gpu use
mamba env create -f ./env/environment-cuda.yaml
mamba activate ASMR_cuda

# dont forget to login into wandb
wandb login
```


# Experiments

## Tests
Test if everything works by running the test experiments, which will test all methods and tasks 
for a single simplified training episode:

```bash
python main.py configs/asmr/tests.yaml -o
```

## Examples
We give examples for environment generation and the general training interface in `example.py`. Here,
ASMR is trained on the Poisson equation using the PPO algorithm.

## Provided Experiments
We provide all experiments used in the paper in the `configs/asmr` folder. The experiments are organized by task/pde.
For example, `configs/asmr/poisson.yaml` contains all Poisson experiments, and ASMR can be trained on this task via the
provided `poisson_asmr` config by executing `python main.py configs/poisson.yaml -e poisson_asmr -o`

## Creating and running experiments

Experiments are configured and distributed via cw2 (https://www.github.com/ALRhub/cw2.git).
For this, the folder `configs` contains a number of `.yaml` files that describe the configuration of the task to run. 
The configs are composed of individual
_experiments_, each of which is separated by three dashes and uniquely identified with an `$EXPERIMENT_NAME$`.

To run an experiment from any of these files on a local machine, type
`python main.py configs/$FILE_NAME$.yaml -e $EXPERIMENT_NAME$ -o`.

To start an experiment on a cluster that uses Slurm
(https://slurm.schedmd.com/documentation.html), run
`python main.py configs/$FILE_NAME$.yaml -e $EXPERIMENT_NAME$ -o -s --nocodecopy`.

Running an experiments provides a (potentially nested) config dictionary to main.py.
For more information on how to use this, refer to the cw2 docs.

# Project Structure

## Reports

After running the first experiment, a folder `reports` will be created.
This folder contains everything that the loggers pick up, organized by the name of the experiment and the repetition.

## Source

The `src` folder contains the source code of this project. It is organized into the following subfolders:

### Algorithms
This folder includes all the iterative training algorithms, that are used by [cw2](https://www.github.com/ALRhub/cw2).
The `rl` directory implements common Reinforcement Learning algorithms, such as `PPO` and `DQN`.

### Environments
The environments include a `Mesh Refinement` and a `Sweep Mesh Refinement` environment.
Both deal with geometric graphs of varying size that represent meshes over a fixed boundary 
and are used for the finite element method.

### Modules
Building blocks for the Message Passing Network architecture.

### Recording

We provide a logger for all console outputs, different scalar metrics and task-dependent visualizations per iteration.
The scalar metrics may optionally be logged to the [wandb dashboard](https://wandb.ai).
The metrics and plot that are recorded depend on both the task and the algorithm being run.
The loggers can (and should) be extended
to suite your individual needs, e.g., by plotting additional metrics or animations whenever necessary.

All locally recorded entities are saved the `reports` directory, where they are organized by their experiment and repetition.

## Util

Common utilities used by the entire source-code can be found here. 
This includes additions torch code, common definitions and functions, and save and load functionality.

## ASMR Evaluations

The folder `asmr_evaluations` contains the code for the evaluation of the ASMR algorithm and all baselines. It uses
checkpoint files from the training of the algorithms and evaluates them on the separate evaluation PDE.
