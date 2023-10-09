# Import modules
from modules.swarm_environments.mesh.mesh_refinement.mesh_refinement import MeshRefinement
from modules.swarm_environments.mesh.mesh_refinement.sweep.sweep_mesh_refinement import SweepMeshRefinement


class EnvironmentLoader:
    def __init__(self):
        # Register classes into the dictionary
        self._environments = {
            'mesh_refinement': MeshRefinement,
            'sweep_mesh_refinement': SweepMeshRefinement,
        }

    def create(self, key, **kwargs):
        creator = self._environments.get(key)
        if not creator:
            raise ValueError(key + ' is not a valid environment.')
        return creator(**kwargs)
