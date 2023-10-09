from dataclasses import dataclass, replace
from typing import Type, List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from skfem import MeshTri1 as MeshTri1

from modules.swarm_environments.mesh.mesh_refinement.domains.mesh_creation_util import polygon_facets


def remove_elements_by_indices(elements: np.array, indices: np.array) -> np.array:
    """
    Removes elements from a mesh by their index
    Args:
        elements: Triangular elements of shape (#elements, 3))
        indices: Indices of vertices to be removed

    Returns: All elements that do not contain the chosen index, with ids adapted accordingly

    """
    # filter rows that contain index
    for index in reversed(sorted(indices)):
        non_containing_indices, = np.where(~np.any(elements == index, axis=0))
        elements = elements[:, non_containing_indices]
        # re-adjust larger indices
        elements[elements >= index] -= 1
    return elements


def meshpy_from_geometry(cls: Type, points: Union[np.array, List[Tuple[float, float]]],
                         facets: List[Tuple[int, int]], max_element_volume: float,
                         holes: Optional[List[Tuple[float, float]]] = None) -> "ExtendedMeshTri1":
    """
    Creates a mesh from a polygon defined by its boundary nodes
    Args:
        cls:
        points:
        facets:
        max_element_volume: Maximum volume/area of a triangle in the mesh
        holes: (Optional) List of points that define the position of holes in the mesh

    Returns: A mesh of the given class

    """
    from meshpy.triangle import MeshInfo, build
    info = MeshInfo()
    info.set_points(points)  # set geometry points

    if holes is not None:
        # set hole position. This is a list of points, one for each hole. For each point,
        # the algorithm will try to find a polygon that is completely inside the hole and then consider this hole
        # to be outside the domain. The algorithm will then try to triangulate the domain outside the hole.
        info.set_holes(holes)

    info.set_facets(facets)  # set facets/edges between points. This is a list of tuples, one for each facet.

    mesh = build(info, max_volume=max_element_volume)  # max_volume is the maximum area of a triangle in the mesh

    mesh_points = np.ascontiguousarray(np.array(mesh.points).T)
    mesh_tris = np.ascontiguousarray(np.array(mesh.elements).T)
    skfem_mesh = cls(mesh_points, mesh_tris)
    return skfem_mesh

@dataclass(repr=False)
class ExtendedMeshTri1(MeshTri1):
    """
    A wrapper/extension of the Scikit FEM MeshTri1 that allows for more flexible mesh initialization.
    This class allows for arbitrary sizes and centers of the initial meshes, and offers utility for different initial
    mesh types.

    """

    @classmethod
    def init_symmetric(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy",
                       center: np.ndarray = np.array([0.5, 0.5]),
                       size: float = 1.0) -> MeshTri1:
        r"""Initialize a symmetric mesh of the unit square.
        The mesh topology is as follows::
            *------------*
            |\          /|
            |  \      /  |
            |    \  /    |
            |     *      |
            |    /  \    |
            |  /      \  |
            |/          \|
            O------------*
            ---l---
            ----size------
        Parameters
        ----------
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        center
            Numpy array of shape (2,0) with the x and y coordinates of the square center, typically
            the coordinates are in the range between 0.0 and 1.0 with a default position of 0.5, 0.5.
        size
            The size of the square. The square is centered around the center point and has a side length of size.

        """
        c_x, c_y = center[0], center[1]
        length = size / 2

        if initial_meshing_method == "meshpy":
            points = np.array([[c_x - length, c_x + length, c_x + length, c_x - length],
                               [c_y - length, c_y - length, c_y + length, c_y + length]], dtype=np.float64).T

            facets = polygon_facets(start=0, end=len(points) - 1)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)

        elif initial_meshing_method == "custom":
            nrefs = int(np.log2(size / (np.sqrt(max_element_volume) * np.sqrt(2))))

            p = np.array([[c_x - length, c_x + length, c_x + length, c_x - length, c_x],
                          [c_y - length, c_y - length, c_y + length, c_y + length, c_y]], dtype=np.float64)

            t = np.array([[0, 1, 4],
                          [1, 2, 4],
                          [2, 3, 4],
                          [0, 3, 4]], dtype=np.int64).T

            mesh = cls(p, t)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")

        return mesh

    @classmethod
    def init_big_domain(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy",
                        center: np.ndarray = np.array([0.5, 0.5]),
                        size: float = 1.0, scale: float = 1.0, domain_type="square") -> MeshTri1:
        r"""Initialize a really big square as a domain
        """
        c_x, c_y = center[0], center[1]
        length = size / 2
        if initial_meshing_method == "meshpy":
            # create a spiral of points
            if domain_type == "small_spiral":
                points = np.array([[0, 1, 1, 0, 0, 3 / 5, 3 / 5, 2 / 5, 2 / 5, 1 / 5, 1 / 5, 4 / 5, 4 / 5, 0],
                                   [0, 0, 1, 1, 2 / 6, 2 / 6, 4 / 6, 4 / 6, 3 / 6, 3 / 6, 5 / 6, 5 / 6, 1 / 6, 1 / 6]],
                                  dtype=np.float64).T
            elif domain_type in ["big_spiral", "large_spiral"]:
                points = np.array([[0, 1, 1, 0, 0, 5 / 7, 5 / 7, 2 / 7, 2 / 7, 3 / 7, 3 / 7, 4 / 7, 4 / 7, 1 / 7, 1 / 7,
                                    6 / 7, 6 / 7, 0],
                                   [0, 0, 1, 1, 2 / 8, 2 / 8, 6 / 8, 6 / 8, 4 / 8, 4 / 8, 5 / 8, 5 / 8, 3 / 8, 3 / 8,
                                    7 / 8, 7 / 8, 1 / 8, 1 / 8]],
                                  dtype=np.float64).T
            elif domain_type == "big_square":
                points = np.array([[0, 1, 1, 0],
                                   [0, 0, 1, 1]],
                                  dtype=np.float64).T
            else:
                raise ValueError(f"Unknown domain type '{domain_type}'")
            # make len(hole_centers) rectangular holes with a size of 0.1
            facets = polygon_facets(start=0, end=len(points) - 1)

            points = points * scale
            # max_element_volume = max_element_volume * 16

            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)

        elif initial_meshing_method == "custom":
            nrefs = int(np.log2(size / (np.sqrt(max_element_volume) * np.sqrt(2))))

            p = np.array([[c_x - length, c_x + length, c_x + length, c_x - length, c_x],
                          [c_y - length, c_y - length, c_y + length, c_y + length, c_y]], dtype=np.float64)

            t = np.array([[0, 1, 4],
                          [1, 2, 4],
                          [2, 3, 4],
                          [0, 3, 4]], dtype=np.int64).T

            mesh = cls(p, t)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")

        return mesh

    @classmethod
    def init_sqsymmetric(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy",
                         center: np.ndarray = np.array([0.5, 0.5]), size: float = 1.0) -> MeshTri1:
        r"""Initialize a symmetric mesh of the unit square.
        The mesh topology is as follows::
            *------*------*
            |\     |     /|
            |  \   |   /  |
            |    \ | /    |
            *------*------*
            |    / | \    |
            |  /   |   \  |
            |/     |     \|
            O------*------*
            ---l---
            ----size-------
        Parameters
        ----------
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy" or. Recommended is "meshpy" as it allows
            for multiprocessing.
        center
            Numpy array of shape (2,0) with the x and y coordinates of the square center, typically
            the coordinates are in the range between 0.0 and 1.0 with a default position of 0.5, 0.5.
        size
            of mesh in the range between 0.0 and 1.0.
        """

        c_x, c_y = center[0], center[1]
        length = size / 2

        if initial_meshing_method == "meshpy":
            points = np.array([[c_x - length, c_x + length, c_x + length, c_x - length],
                               [c_y - length, c_y - length, c_y + length, c_y + length]], dtype=np.float64).T

            facets = polygon_facets(start=0, end=len(points) - 1)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)
        elif initial_meshing_method == "custom":
            nrefs = int(np.log2(size / (np.sqrt(max_element_volume) * np.sqrt(2))))

            p = np.array([[c_x - length, c_x, c_x + length, c_x - length, c_x,
                           c_x + length, c_x - length, c_x, c_x + length],
                          [c_y - length, c_y - length, c_y - length, c_y, c_y,
                           c_y, c_y + length, c_y + length, c_y + length]],
                         dtype=np.float64)
            t = np.array([[0, 1, 4],
                          [1, 2, 4],
                          [2, 4, 5],
                          [0, 3, 4],
                          [3, 4, 6],
                          [4, 6, 7],
                          [4, 7, 8],
                          [4, 5, 8]], dtype=np.int64).T

            mesh = cls(p, t)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")

        return mesh

    @classmethod
    def init_lshaped(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy",
                     hole_position=np.array([0.5, 0.5]),
                     *args, **kwargs) -> MeshTri1:
        r"""Initialize a mesh for the L-shaped domain.
        The mesh topology is as follows::
            *-------*       1
            | \     |
            |   \   |
            |     \ |
            *-------*-------*
            |     / | \     |
            |   /   |   \   |
            | /     |     \ |
            0-------*-------*
        Parameters
        ----------
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        hole_position
            is the position of the hole in the lshaped domain np.array of shape (2,), range between 0, 1.
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        """
        assert not kwargs, f"No keyword arguments allowed, given '{kwargs}'"
        assert not args, f"No positional arguments allowed, given '{args}'"
        hp_x, hp_y = hole_position[0], hole_position[1]

        if initial_meshing_method == "meshpy":
            points = np.array([[0., 1., 1., hp_x, hp_x, 0.],
                               [0., 0., hp_y, hp_y, 1., 1.]], dtype=np.float64).T
            facets = polygon_facets(start=0, end=len(points) - 1)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)
        elif initial_meshing_method == "custom":
            desired_element_size = (np.sqrt(max_element_volume) * np.sqrt(2))
            # rough approximation of the desired size of a mesh edge based on its volume
            nrefs = int(np.log2(1 / desired_element_size))

            points = np.array([[hp_x, 1., hp_x, 0., hp_x, 0., 0., 1.],
                               [hp_y, hp_y, 1., hp_y, 0., 0., 1., 0.]], dtype=np.float64)

            elements = np.array([[0, 1, 7],
                                 [0, 2, 6],
                                 [0, 6, 3],
                                 [0, 7, 4],
                                 [0, 4, 5],
                                 [0, 3, 5]], dtype=np.int64).T

            mesh = cls(points, elements)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")
        return mesh

    @classmethod
    def init_square_hole(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy",
                         hole_position: np.ndarray = np.array([0.25, 0.25]),
                         hole_size: np.ndarray = np.array([0.5, 0.5])) -> MeshTri1:
        r"""Initialize a mesh for the symmetric domain with a hole.
        The mesh topology is as follows::
            *-------*-------*
            | \   /    \  / |
            |  *---------*  |
            | /|         | \|
            *  |         |  *
            | \|         |/ |
            |  *---------*  |
            | /   \    /  \ |
            0-------*-------*
        Parameters
        ----------
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        hole_position
            Numpy array with hole position in the range between 0.0 and 1.0 for x and y coordinates.
            Position of the lower left corner of the hole.
        hole_size
            size of the hole in the range between 0.0 and 1.0. Size of the hole in x and y direction from the lower
            left corner of the hole.
        """
        hp_x, hp_y = hole_position[0], hole_position[1]  # position of the lower left corner of the hole

        hole_size_x, hole_size_y = hole_size[0], hole_size[1]

        vertices = np.array([[0., .5, 1., 1., 1., .5, 0., 0., hp_x, hp_x + hole_size_x, hp_x + hole_size_x, hp_x],
                             [0., 0., 0., .5, 1., 1., 1., .5, hp_y, hp_y, hp_y + hole_size_y, hp_y + hole_size_y]],
                            dtype=np.float64)

        if initial_meshing_method == "meshpy":
            points = vertices.T
            facets = polygon_facets(start=0, end=7) + polygon_facets(start=8, end=11)

            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=[(float(np.clip(hp_x + hole_size_x / 2, 0, 1)),
                                                float(np.clip(hp_y + hole_size_y / 2, 0, 1)))])
        elif initial_meshing_method == "custom":
            desired_element_size = (np.sqrt(max_element_volume) * np.sqrt(2))
            # rough approximation of the desired size of a mesh edge based on its volume
            nrefs = int(np.log2(1 / desired_element_size))

            # delete vertices that are not needed as they are in the hole
            vertex_deletion_indices = []
            if hp_y + hole_size_y >= 1. and hp_x <= 0.:
                vertex_deletion_indices.append(6)
                vertex_deletion_indices.append(11)
            if hp_y + hole_size_y >= 1. and hp_x + hole_size_x >= 1.:
                vertex_deletion_indices.append(4)
                vertex_deletion_indices.append(10)
            if hp_y <= 0. and hp_x + hole_size_x >= 1.:
                vertex_deletion_indices.append(2)
                vertex_deletion_indices.append(9)
            if hp_y <= 0. and hp_x <= 0.:
                vertex_deletion_indices.append(0)
                vertex_deletion_indices.append(8)

            if hp_x <= 0.:
                vertex_deletion_indices.append(7)
            if hp_y + hole_size_y >= 1.:
                vertex_deletion_indices.append(5)
            if hp_x + hole_size_x >= 1.:
                vertex_deletion_indices.append(3)

            if hp_y <= 0.:
                vertex_deletion_indices.append(1)

            vertices = np.delete(vertices, vertex_deletion_indices, axis=1)

            elements = np.array([[0, 1, 8],
                                 [1, 2, 9],
                                 [2, 3, 9],
                                 [3, 4, 10],
                                 [4, 5, 10],
                                 [5, 6, 11],
                                 [6, 7, 11],
                                 [7, 0, 8],
                                 [8, 9, 1],
                                 [9, 10, 3],
                                 [10, 11, 5],
                                 [11, 8, 7]
                                 ], dtype=np.int64).T

            elements = remove_elements_by_indices(elements=elements, indices=vertex_deletion_indices)

            assert elements.shape[1] >= 6, f"Mesh is not closed. Given vertices '{vertices.shape}' " \
                                           f"and elements '{elements.shape}' for parameters '{hole_position}' and '{hole_size}'."
            mesh = cls(vertices, elements)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")
        return mesh

    @classmethod
    def init_trapezoid_hole(cls: Type, boundary_nodes: np.array, holes: list,
                            max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy") -> MeshTri1:
        r"""
        Parameters
        ----------
        boundary_nodes
            Numpy array of shape (2, n) with the x and y coordinates of the boundary nodes. Must be in the correct
            order, i.e. the nodes must be connected by edges in the correct order.
        holes
            List of numpy arrays of shape (2, n) with the x and y coordinates of the hole nodes. Must be in the correct
            order, i.e. the nodes must be connected by edges in the correct order.
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.

        """

        if initial_meshing_method == "meshpy":
            points = boundary_nodes
            facets = polygon_facets(start=0, end=len(points)-1)

            hole_start_idx = len(points)
            for hole in holes:
                hole_end_idx = hole_start_idx + len(hole) - 1
                facets += polygon_facets(start=hole_start_idx, end=hole_end_idx)
                hole_start_idx = hole_end_idx + 1

            hole_centers = [tuple(np.mean(hole, axis=0)) for hole in holes]
            # concatenate all holes to the points
            points = np.concatenate([points] + holes, axis=0)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=hole_centers
                                        )
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")
        return mesh


    @classmethod
    def init_circle(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy", *args,
                    **kwargs) -> MeshTri1:
        r"""Initialize a circle mesh.
        Works by repeatedly refining the following mesh and moving
        new nodes to the boundary::
                   -
                 / | \
               /   |   \
             /     |     \
            |------*------|
             \     |     /
               \   |   /
                 \ | /
            0      -
        Parameters
        ----------
        max_element_volume
            desired element size in the range between 0.0 and 1.0
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        """
        assert not kwargs, f"No keyword arguments allowed, given '{kwargs}'"
        assert not args, f"No positional arguments allowed, given '{args}'"
        if initial_meshing_method == "meshpy":
            points = [((np.cos(angle) + 1) / 2, (np.sin(angle) + 1 / 2))
                      for angle in np.linspace(0, 2 * np.pi, int(1 / np.sqrt(max_element_volume)), endpoint=False)]
            facets = polygon_facets(start=0, end=len(points) - 1)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)
        elif initial_meshing_method == "custom":
            desired_element_size = (np.sqrt(max_element_volume) * np.sqrt(2))
            # rough approximation of the desired size of a mesh edge based on its volume
            nrefs = int(np.log2(1 / desired_element_size))

            p = np.array([[0.5, 0.5],
                          [1., 0.5],
                          [0.5, 1.],
                          [0., 0.5],
                          [0.5, 0.]], dtype=np.float64).T

            t = np.array([[0, 1, 2],
                          [0, 1, 4],
                          [0, 2, 3],
                          [0, 3, 4]], dtype=np.int64).T
            m = cls(p, t)

            c_x, c_y = 0.5, 0.5  # center of the circle
            radius = 0.5

            for _ in range(nrefs):
                m = m.refined()
                m = m.smoothed()
                D = m.boundary_nodes()
                tmp = m.p
                tmp[:, D] = tmp[:, D] / np.linalg.norm(tmp[:, D], axis=0)
                m = replace(m, doflocs=tmp)

                p = m.p * radius
                p[0], p[1] = p[0] + c_x, p[1] + c_y
                t = m.t
            mesh = cls(p, t)
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")

        return mesh

    @classmethod
    def init_octagon(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy") -> MeshTri1:
        r"""Initialize an octagonal mesh 
               *--------*
             /  \     /  \
            *    \   /    *
            |  \  \ /  /  |
            |      *      |
            |  / /   \ \  |
            *   /     \   *
             \ /       \ /
              *---------*
        Parameters
        ----------
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method :
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        """
        center: np.ndarray = np.array([0.5, 0.5])
        #  Numpy array of shape (2,0) with the x and y coordinates of the hexagon center
        size: int = 1

        c_x, c_y = center[0], center[1]
        length = size / 2

        boundary_nodes = np.array([[c_x, c_y],
                                   [c_x + length, c_y + length / 2],
                                   [c_x + length / 2, c_y + length],
                                   [c_x - length / 2, c_y + length],
                                   [c_x - length, c_y + length / 2],
                                   [c_x - length, c_y - length / 2],
                                   [c_x - length / 2, c_y - length],
                                   [c_x + length / 2, c_y - length],
                                   [c_x + length, c_y - length / 2]
                                   ], dtype=np.float64)

        if initial_meshing_method == "meshpy":
            points = boundary_nodes[:, 1:]
            facets = polygon_facets(start=0, end=len(points) - 1)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)
        elif initial_meshing_method == "custom":
            boundary_nodes = boundary_nodes.T
            desired_element_size = (np.sqrt(max_element_volume) * np.sqrt(2))
            # rough approximation of the desired size of a mesh edge based on its volume
            nrefs = int(np.log2(1 / desired_element_size))

            t = np.array([[1, 2, 0],
                          [2, 3, 0],
                          [3, 4, 0],
                          [4, 5, 0],
                          [5, 6, 0],
                          [6, 7, 0],
                          [7, 8, 0],
                          [8, 1, 0]
                          ], dtype=np.int64).T

            mesh = cls(boundary_nodes, t)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")

        return mesh

    @classmethod
    def init_hexagon(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy") -> MeshTri1:
        r"""Initialize a mesh for the hexagon domain.
                *------*
              /   \   /  \
            *------*------*
              \  /   \  /
                *------*
        Parameters
        ----------
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        """

        center: np.ndarray = np.array([0.5, 0.5])
        #  Numpy array of shape (2,0) with the x and y coordinates of the hexagon center
        size: int = 1

        c_x, c_y = center[0], center[1]
        length = size / 2

        boundary_nodes = np.array([[c_x, c_y],
                                   [c_x + length, c_y],
                                   [c_x + length / 2, c_y + length],
                                   [c_x - length / 2, c_y + length],
                                   [c_x - length, c_y],
                                   [c_x - length / 2, c_y - length],
                                   [c_x + length / 2, c_y - length]
                                   ], dtype=np.float64)

        if initial_meshing_method == "meshpy":
            points = boundary_nodes[:, 1:]
            facets = polygon_facets(start=0, end=len(points) - 1)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)
        elif initial_meshing_method == "custom":
            boundary_nodes = boundary_nodes.T
            desired_element_size = (np.sqrt(max_element_volume) * np.sqrt(2))
            # rough approximation of the desired size of a mesh edge based on its volume
            nrefs = int(np.log2(1 / desired_element_size))

            t = np.array([[1, 2, 0],
                          [2, 3, 0],
                          [3, 4, 0],
                          [4, 5, 0],
                          [5, 6, 0],
                          [6, 1, 0],
                          ], dtype=np.int64).T

            mesh = cls(boundary_nodes, t)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")

        return mesh

    @classmethod
    def init_trapezoid(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy",
                       boundary_nodes: np.array = np.array([[0.0, 0.0], [0.0, 1.0],
                                                            [1.0, 0.6], [1.0, 0.2]])) -> MeshTri1:
        r"""Initialize some trapezoid mesh.
        Parameters
        ----------
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        boundary_nodes : np.array
            Numpy array of shape (4,2) with the x and y coordinates of the boundary nodes of the trapezoid.
        """
        assert boundary_nodes.shape[0] == 4, "Boundary nodes must span a trapezoid"
        assert boundary_nodes.shape[1] == 2, "Boundary nodes must be 2D"
        assert boundary_nodes[0, 0] == boundary_nodes[1, 0], "Boundary nodes must span a trapezoid"
        assert boundary_nodes[2, 0] == boundary_nodes[3, 0], "Boundary nodes must span a trapezoid"

        if initial_meshing_method == "meshpy":
            points = boundary_nodes
            facets = polygon_facets(start=0, end=len(points) - 1)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)
        elif initial_meshing_method == "custom":
            desired_element_size = (np.sqrt(max_element_volume) * np.sqrt(2))
            # rough approximation of the desired size of a mesh edge based on its volume
            nrefs = int(np.log2(1 / desired_element_size))

            boundary_nodes = np.concatenate((boundary_nodes, np.mean(boundary_nodes, axis=0)[None, :]),
                                            axis=0).T

            t = np.array([[0, 1, 4],
                          [1, 2, 4],
                          [2, 3, 4],
                          [0, 3, 4]], dtype=np.int64).T

            mesh = cls(boundary_nodes, t)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")

        return mesh

    @classmethod
    def init_convex_polygon(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy",
                            boundary_nodes: np.array = np.array([[1., 0.], [.5, 1.], [-.5, 1.],
                                                                 [-1., 0.], [-.5, -1], [.5, -1.]])) -> MeshTri1:
        r"""Initialize a mesh from some boundary nodes by spanning a convex polygon.
                *------*
              /   \   /  \
            *------*------*
              \  /   \  /
                *------*
        Parameters
        ----------
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        boundary_nodes
            Numpy array of points that define the boundary of the polygon - shape Nx2
        """
        # get convex hull of boundary nodes
        from scipy.spatial import ConvexHull
        boundary_nodes = (boundary_nodes - np.min(boundary_nodes)) / (
                np.max(boundary_nodes) - np.min(boundary_nodes))  # normalize to [0,1]
        boundary_nodes = boundary_nodes[ConvexHull(boundary_nodes).vertices]

        if initial_meshing_method == "meshpy":
            points = boundary_nodes
            facets = polygon_facets(start=0, end=len(points) - 1)
            mesh = meshpy_from_geometry(cls=cls,
                                        points=points,
                                        facets=facets,
                                        max_element_volume=max_element_volume,
                                        holes=None)
        elif initial_meshing_method == "custom":
            desired_element_size = (np.sqrt(max_element_volume) * np.sqrt(2))
            # rough approximation of the desired size of a mesh edge based on its volume
            nrefs = int(np.log2(1 / desired_element_size))

            center = (np.min(boundary_nodes, axis=0) + np.max(boundary_nodes, axis=0)) / 2
            center = center[:, None]

            nodes = np.concatenate((center, boundary_nodes.T),
                                   axis=1)  # "key" nodes are polygon center + boundary nodes

            elements = np.zeros((len(boundary_nodes), 3), dtype=np.int64)

            for index in range(1, len(boundary_nodes)):
                elements[index - 1] = [index, index + 1,
                                       0]  # connect boundary nodes to center and the next boundary clockwise
            elements[len(boundary_nodes) - 1] = [len(boundary_nodes), 1, 0]

            mesh = cls(nodes, elements.T)

            for _ in range(nrefs):
                mesh = mesh.refined()
        else:
            raise ValueError(f"Unknown initial meshing method '{initial_meshing_method}'")

        return mesh

    @classmethod
    def init_polygon(cls: Type, max_element_volume: float = 0.01, initial_meshing_method: str = "meshpy",
                     boundary_nodes: np.array = np.array([[1., 0.], [.5, 1.], [-.5, 1.],
                                                          [-1., 0.], [-.5, -1], [.5, -1.]])) -> MeshTri1:
        r"""Initialize a mesh
                *------*
              /   \   /  \
            *------*------*
              \  /   \  /
                *------*
        Parameters
        ----------
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        initial_meshing_method : str
            Which meshing method to use. Either "custom", "meshpy". Recommended is "meshpy" as it allows
            for multiprocessing.
        boundary_nodes
            Numpy array of points that define the boundary of the polygon - shape Nx2
        """
        assert initial_meshing_method == "meshpy", "Only meshpy is supported for arbitrary polygon meshes"
        points = boundary_nodes
        facets = polygon_facets(start=0, end=len(points) - 1)
        mesh = meshpy_from_geometry(cls=cls,
                                    points=points,
                                    facets=facets,
                                    max_element_volume=max_element_volume,
                                    holes=None)

        return mesh

    def element_finder(self, mapping=None):
        """
        Find the element that contains the point (x, y). Returns -1 if the point is in no element
        Args:
            mapping: A mapping from the global node indices to the local node indices. Currently not used

        Returns:

        """

        nelems = self.t.shape[1]
        elements = self.p[:, self.t].T

        from modules.swarm_environments.util.point_in_geometry import points_in_triangles, fast_points_in_triangles

        def finder(x, y, _use_point_in_triangle=False):
            """
            Find the element that contains the point (x, y). Returns -1 if the point is in no element
            Args:
                x: Array of point x coordinates of shape (num_points, )
                y: Array of point y coordinates of shape (num_points, )
                _use_point_in_triangle: If True, use the point_in_triangle algorithm. If False, use the KDTree.
                Internal parameter that is used to switch between the two algorithms as a backup if the KDTree fails.

            Returns:

            """
            if _use_point_in_triangle:
                # brute force approach - check all elements
                element_indices = points_in_triangles(points=np.array([x, y]).T, triangles=elements)
            else:
                # find candidate elements
                from pykdtree.kdtree import KDTree
                tree = KDTree(np.mean(self.p[:, self.t], axis=1).T)
                num_elements = min(5, nelems)
                # usually (distance, index), but we do not care about the distance
                _, candidate_indices = tree.query(np.array([x, y]).T, num_elements)

                # try to find the element that contains the point using the KDTree
                element_indices = fast_points_in_triangles(points=np.array([x, y]).T, triangles=elements,
                                                           candidate_indices=candidate_indices)
                invalid_elements = element_indices == -1

                if invalid_elements.any():  # fallback to brute force search
                    element_indices[invalid_elements] = finder(x=x[invalid_elements],
                                                               y=y[invalid_elements],
                                                               _use_point_in_triangle=True)

            return element_indices

        return finder

    def __post_init__(self):
        """Support node orders used in external formats.

        We expect ``self.doflocs`` to be ordered based on the
        degrees-of-freedom in :class:`skfem.assembly.Dofs`.  External formats
        for high order meshes commonly use a less strict ordering scheme and
        the extra nodes are described as additional rows in ``self.t``.  This
        method attempts to accommodate external formas by reordering
        ``self.doflocs`` and changing the indices in ``self.t``.

        """
        from skfem.element import Element
        import logging
        logger = logging.getLogger(__name__)
        if self.sort_t:
            self.t = np.sort(self.t, axis=0)

        self.doflocs = np.asarray(self.doflocs, dtype=np.float64, order="K")
        self.t = np.asarray(self.t, dtype=np.int64, order="K")

        M = self.elem.refdom.nnodes

        if self.nnodes > M and self.elem is not Element:
            # reorder DOFs to the expected format: vertex DOFs are first
            # note: not run if elem is not set
            p, t = self.doflocs, self.t
            t_nodes = t[:M]
            uniq, ix = np.unique(t_nodes, return_inverse=True)
            self.t = (np.arange(len(uniq), dtype=np.int64)[ix]
                      .reshape(t_nodes.shape))
            doflocs = np.hstack((
                p[:, uniq],
                np.zeros((p.shape[0], np.max(t) + 1 - len(uniq))),
            ))
            doflocs[:, self.dofs.element_dofs[M:].flatten('F')] = \
                p[:, t[M:].flatten('F')]
            self.doflocs = doflocs

        # C_CONTIGUOUS is more performant in dimension-based slices
        if not self.doflocs.flags['C_CONTIGUOUS']:
            if self.doflocs.shape[1] > 1e3:
                logger.warning("Transforming over 1000 vertices "
                               "to C_CONTIGUOUS.")
            self.doflocs = np.ascontiguousarray(self.doflocs)

        if not self.t.flags['C_CONTIGUOUS']:
            if self.t.shape[1] > 1e3:
                logger.warning("Transforming over 1000 elements "
                               "to C_CONTIGUOUS.")
            self.t = np.ascontiguousarray(self.t)

        # run validation
        if self.validate and logger.getEffectiveLevel() <= logging.DEBUG:
            self.is_valid()


###############################################################################
#                                                                             #
#                              Test Cases                                     #
#                                                                             #
###############################################################################

def visualize_current_geometries():
    """Visualize current geometries"""
    max_element_volume = 0.01
    mesh_sym = ExtendedMeshTri1.init_symmetric(max_element_volume)
    mesh_sym1 = ExtendedMeshTri1.init_symmetric(max_element_volume, center=np.array([0.4, 0.4]), size=0.8)
    mesh_sym_sq = ExtendedMeshTri1.init_sqsymmetric(max_element_volume)
    mesh_sym_sq1 = ExtendedMeshTri1.init_sqsymmetric(max_element_volume, center=np.array([0.6, 0.6]), size=0.6)
    mesh_symetric_hole = ExtendedMeshTri1.init_square_hole(max_element_volume)
    mesh_symetric_hole1 = ExtendedMeshTri1.init_square_hole(max_element_volume=max_element_volume,
                                                            hole_position=np.array([0.5, 0.5]),
                                                            hole_size=np.array([0.3, 0.3]))
    mesh_lshaped = ExtendedMeshTri1.init_lshaped(max_element_volume=max_element_volume,
                                                 hole_position=np.array([0.5, 0.5]))
    mesh_circle = ExtendedMeshTri1.init_circle(max_element_volume)
    mesh_octagon = ExtendedMeshTri1.init_octagon(max_element_volume)
    mesh_hexagon = ExtendedMeshTri1.init_hexagon(max_element_volume)
    mesh_polygon = ExtendedMeshTri1.init_convex_polygon(boundary_nodes=np.array([[1., 0.], [-.5, 1.], [-1., 0.],
                                                                                 [-.5, -1], [.5, -1.]]), )

    meshes = [mesh_sym, mesh_sym1, mesh_sym_sq, mesh_sym_sq1, mesh_symetric_hole, mesh_symetric_hole1, mesh_lshaped,
              mesh_circle, mesh_octagon, mesh_hexagon, mesh_polygon]

    x, y = [], []

    fig = plt.figure(1, figsize=(100, 100))

    i = 1
    for mesh in meshes:
        fig.add_subplot(4, 5, i)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.scatter(mesh.p[0], mesh.p[1])
        i += 1

    plt.show()


if __name__ == '__main__':
    visualize_current_geometries()
