import math

import numpy as np
from skfem import Mesh
from skfem.visuals.matplotlib import draw, plot


def element_midpoints(mesh: Mesh) -> np.array:
    """
    Get the midpoint of each element
    Args:
        mesh: The mesh as a skfem.Mesh object

    Returns: Array of shape (num_elements, 2) containing the midpoint of each element

    """
    return np.mean(mesh.p[:, mesh.t], axis=1).T


def visualize_solution(mesh: Mesh, solution: np.array, draw_mesh: bool = True):
    if draw_mesh:
        ax = draw(mesh)
        return plot(mesh, solution, ax=ax, shading='gouraud', colorbar=True)
    else:
        return plot(mesh, solution, shading='gouraud', colorbar=True)


def convert_error(error_estimate, element_indices, num_vertices: int):
    """
    get error per vertex from error per element

    Args:
        error_estimate: Error estimate per element
        element_indices: Elements of the mesh. Array of shape (num_elements, 3)
        num_vertices: Number of vertices of the mesh
    Returns:

    """
    error_per_node = np.zeros(shape=(num_vertices,))
    np.add.at(error_per_node, element_indices, error_estimate[:, None])  # inplace operation
    return error_per_node


def get_aggregation_per_element(solution: np.array, element_indices: np.array,
                                aggregation_function_str: str = 'mean') -> np.array:
    """
    get aggregation of solution per element from solution per vertex by adding all spanning vertices for each element

    Args:
        solution: Error estimate per element of shape (num_elements, solution_dimension)
        element_indices: Elements of the mesh. Array of shape (num_elements, vertices_per_element),
        where vertices_per_element is 3 triangular meshes
        aggregation_function_str: The aggregation function to use. Can be 'mean', 'std', 'min', 'max', 'median'
    Returns: An array of shape (num_elements, ) containing the solution per element

    """
    if aggregation_function_str == 'mean':
        solution_per_element = solution[element_indices].mean(axis=1)
    elif aggregation_function_str == 'std':
        solution_per_element = solution[element_indices].std(axis=1)
    elif aggregation_function_str == 'min':
        solution_per_element = solution[element_indices].min(axis=1)
    elif aggregation_function_str == 'max':
        solution_per_element = solution[element_indices].max(axis=1)
    elif aggregation_function_str == 'median':
        solution_per_element = np.median(solution[element_indices], axis=1)
    else:
        raise ValueError(f'Aggregation function {aggregation_function_str} not supported')
    return solution_per_element


def angle_to(point: np.ndarray, center_point: np.ndarray = np.array([0.5, 0.5])):
    """
    get angle to a center point for a given point
    center point is mapped to the origin

    Args:
        point: An array of shape [2, ]
        center_point: An array of shape [2, ]
    Returns:
        returns a scalar value - the angle to the center point
    """

    x = point[0] - center_point[0]
    y = point[1] - center_point[1]

    return math.atan2(y, x)


def get_line_segment_distances(points: np.array, projection_segments: np.array,
                               return_minimum: bool = False, return_tangent_points: bool = False) -> np.array:
    """
    Calculates the distances of an array of points to an array of line segments.
    Vectorized for any number of points and line segments
    Args:
        points: An array of shape [num_points, 2], i.e., an array of points to project towards the projection segments
        projection_segments: An array of shape [num_segments, 4], i.e., an array of line segments/point pairs
        return_minimum: If True, the minimum distance is returned. If False, an array of all distances is returned
        return_tangent_points: If True, distances and tangent points of the projections to all segments are returned

    Returns: An array of shape [num_points, {num_segments, 1}] containing the distance of each point to each segment
        or the minimum segment, depending on return_minimum

    """
    segment_distances = projection_segments[:, :2] - projection_segments[:, 2:]
    tangent_positions = np.sum(projection_segments[:, :2] * segment_distances, axis=1) - points @ segment_distances.T
    segment_lengths = np.linalg.norm(segment_distances, axis=1)

    # the normalized tangent position is in [0,1] if the projection to the line segment is directly possible
    normalized_tangent_positions = tangent_positions / segment_lengths ** 2

    # it gets clipped to [0,1] otherwise, i.e., clips projections to the boundary of the line segment.
    # this is necessary since line segments may describe an internal part of the mesh domain, meaning
    # that we always want the distance to the segment rather than the distance to the line it belongs to
    normalized_tangent_positions[normalized_tangent_positions > 1] = 1  # clip too big values
    normalized_tangent_positions[normalized_tangent_positions < 0] = 0  # clip too small values
    tangent_points = projection_segments[:, :2] - normalized_tangent_positions[..., None] * segment_distances
    projection_vectors = points[:, None, :] - tangent_points

    distances = np.linalg.norm(projection_vectors, axis=2)
    if return_minimum:
        distances = np.min(distances, axis=1)
    if return_tangent_points:
        return distances, tangent_points
    return distances
