# script to gather the evaluations made on the cluster experiments. Does nothing except going into the files and copying
# the data into a big npz
import argparse
import os
import sys

import numpy as np
import yaml
from tqdm import tqdm

# path hacking for scripts from top level
current_directory = os.getcwd()
sys.path.append(current_directory)


def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_folder', help="exp folder in /reports/ to search for checkpoints in")
    args = parser.parse_args()
    exp_folder = args.exp_folder

    root = "reports"

    exp_root_path = os.path.join(root, exp_folder)
    if not os.path.exists(exp_root_path):
        raise FileNotFoundError(f"Path '{exp_root_path}' not found."
                                f"Make sure to start this script from the root folder 'DAVIS' of the project")

    valid_exps = sorted([x for x in os.listdir(exp_root_path) if os.path.isdir(os.path.join(exp_root_path, x))])

    num_repetitions = 10
    num_pdes = 100

    pde_type = None
    data = {}
    for exp in valid_exps:
        print(f"Copying results for: {exp}")
        exp_path = os.path.join(exp_root_path, exp, "log")

        experiment_results = np.full((num_repetitions, num_pdes, 4), np.nan)

        approach_idx = None
        element_penalty = None
        for rep_id, rep in tqdm(enumerate(sorted(os.listdir(exp_path)))):
            rep_in_path = os.path.join(exp_path, rep)

            # load config yaml file to get the name of the environment and the idx of the approach
            try:
                config = yaml.load(open(os.path.join(rep_in_path, "config.yaml"), "r"), Loader=yaml.FullLoader)
                if pde_type is None:
                    pde_type = config.get("environment").get("mesh_refinement").get("fem").get("pde_type")
                if approach_idx is None:
                    approach_idx = int(config["recording"]["idx"])
                if approach_idx in [2200, 2201, 2210]:  # "sweep". The 'penalty' is maximum number of elements
                    element_penalty = config["environment"]["mesh_refinement"]["maximum_elements"]
                elif approach_idx in [2000, 2001, 2010]:  # "single agent".
                    # # Here, the 'penalty' is the number of timesteps
                    element_penalty = config["environment"]["mesh_refinement"]["num_timesteps"]
                else:  # everything else
                    element_penalty = config["environment"]["mesh_refinement"]["element_penalty"]["value"]
            except FileNotFoundError:
                print(f"Could not find config file in {rep_in_path}. Skipping...")
                continue

            # load .npz of final values
            try:
                final_values = np.load(os.path.join(rep_in_path, "final_values.npz"), allow_pickle=True)
                final_values = final_values["TABLES"]
            except FileNotFoundError:
                print(f"Could not find final values file in {rep_in_path}. Skipping...")
                continue

            # assume a single table and element penalty per "final values" for now
            final_values = final_values.item()["final_0"]
            final_values = final_values.reset_index().drop(columns=['penalty'])
            np_array = np.full((num_pdes, 4), np.nan)
            np_array[:, 0] = final_values["num_agents"].values
            np_array[:, 1] = final_values["squared_error"].values
            np_array[:, 2] = final_values["mean_error"].values
            np_array[:, 3] = final_values["top0.1_error"].values

            experiment_results[rep_id] = np_array

        data[f"idx={approach_idx}_penalty={element_penalty}"] = experiment_results

    os.makedirs(f"evaluation_results/iclr2023/{pde_type}", exist_ok=True)
    save_path = f"evaluation_results/iclr2023/{pde_type}/{exp_folder}.npz"
    np.savez_compressed(save_path, **data)
    print(f"Saved data with keys '{data.keys()}' in path '{save_path}'! 🎉")


if __name__ == '__main__':
    main()
