from importlib import import_module
from os import path

import yaml


class EnvironmentLoader:
    """
    Can dynamically load environment classes that have been registered in environments.yaml.
    """

    def __init__(self):
        self._environments = {}

        config_file_path = path.join(path.dirname(__file__), "environments.yaml")
        self._register_from_config_file(config_file_path)

    def _register_from_config_file(self, yaml_file_path: str):
        """
        Interprets each top-level key as the ID of a new environment.
        Properties:
            module: path to the module containing the class, separated by dots.

        Args:
            yaml_file_path: absolute path to a yaml file
        """
        with open(yaml_file_path) as f:
            from yaml.loader import SafeLoader
            data = yaml.load(f, Loader=SafeLoader)

            for env_id in data.keys():
                env = data.get(env_id)
                self._register(
                    env_id=env_id,
                    env_module_name=env.get("module"),
                )

    def _register(self, env_id: str, env_module_name: str):
        self._environments[env_id] = {
            "module": env_module_name,
        }

    def load(self, env_id: str):
        """
        Tries to dynamically load the environment class from the module registered under the given ID.
        Assumes that the module under the given path contains a class
        with the same name as the containing file.

        Returns:
            The environment class.
        """
        if env_id not in self._environments:
            raise ValueError(f"Unknown environment '{env_id}'")

        module_name = self._environments.get(env_id).get("module")
        class_name = module_name.split(".")[-1].title().replace("_", "")
        # assume class name is a Camel-cased version of file name, i.e., file_name.py -> FileName
        module = import_module(module_name)
        env_class = getattr(module, class_name)
        return env_class
