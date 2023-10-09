import numpy as np
from numba import njit
from typing import Optional


@njit(fastmath=True)
def points_on_edges(points: np.array, edges: np.array, candidate_indices: Optional[np.array] = None):
    """
    Determines which edge each point lies on. If the point does not lie on any edge, the index is -1.
    Args:
        points: An array of points with shape (2, N)
        edges: An array of edges with shape (2, 2, M), where each edge is defined by its start and end points.
        candidate_indices: An array of triangle indices with shape (N, K) containing K candidate edges per queried point
    Returns: An array of edge indices with shape (num_points, ) containing the index of the edge that contains the
        point. If the point does not lie on any edge, the index is -1.

    """
    points = points.T
    edges = edges.T
    points_on_edges = np.empty(points.shape[0], dtype=np.int64)
    points_on_edges.fill(-1)  # -1 means no triangle found for this point
    for point_index, point in enumerate(points):
        if candidate_indices is not None:
            edge_indices = candidate_indices[point_index].astype(np.int64)
        else:
            edge_indices = np.arange(edges.shape[0], dtype=np.int64)
        for edge_index in edge_indices:
            current_edge = edges[edge_index]
            # the commented out code is equivalent to the following, but roughly 100x slower due to numpy array accesses
            # edge_vector = current_edge[1] - current_edge[0]
            # point_vector = point - current_edge[0]
            # cross_product = point_vector[1] * edge_vector[0] - point_vector[0] * edge_vector[1]
            # dot_product = point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]
            # squared_length = edge_vector[0] * edge_vector[0] + edge_vector[1] * edge_vector[1]
            # edge_vector = current_edge[1] - current_edge[0]
            # point_vector = point - current_edge[0]

            abs_cross_product = np.abs(
                np.subtract(
                    np.multiply(
                        np.subtract(point[1], current_edge[0, 1]),
                        np.subtract(current_edge[1, 0], current_edge[0, 0])
                    ),
                    np.multiply(
                        np.subtract(point[0], current_edge[0, 0]),
                        np.subtract(current_edge[1, 1], current_edge[0, 1])
                    )
                )
            )
            dot_product = np.add(
                np.multiply(
                    np.subtract(point[0], current_edge[0, 0]),
                    np.subtract(current_edge[1, 0], current_edge[0, 0])
                ),
                np.multiply(
                    np.subtract(point[1], current_edge[0, 1]),
                    np.subtract(current_edge[1, 1], current_edge[0, 1])
                )
            )
            squared_length = np.add(
                np.multiply(
                    np.subtract(current_edge[1, 0], current_edge[0, 0]),
                    np.subtract(current_edge[1, 0], current_edge[0, 0])
                ),
                np.multiply(
                    np.subtract(current_edge[1, 1], current_edge[0, 1]),
                    np.subtract(current_edge[1, 1], current_edge[0, 1])
                )
            )
            if (abs_cross_product < 1e-9) + (0 <= dot_product) + (dot_product <= squared_length) == 3:
                points_on_edges[point_index] = edge_index
                break
    return points_on_edges


@njit(fastmath=True)
def fast_points_in_triangles(points: np.array, triangles: np.array,
                             candidate_indices: Optional[np.array] = None) -> np.array:
    """
    Args:
        points (np.ndarray): array of points with shape (N, 2)
        triangles (np.ndarray): array of triangles. Shape (M, 3, 2)
        candidate_indices (np.ndarray): array of candidate triangle indices for each point. Shape (N, K), where
        K is the number of candidate triangles for each point. If None, all triangles are checked for each point.

    Returns:
        point_in (np.ndarray): Array with triangle index for each point. Shape (N, ) containing the index of the
        triangle that contains the point. If the point is not in any triangle, the index is -1.
    """
    point_in = np.empty(points.shape[0], dtype=np.int64)
    point_in.fill(-1)  # -1 means no triangle found for this point

    # for each point, check if it lies inside any of the triangles by checking whether the point is on the same side of
    # the triangle's edges as the triangle's vertices
    for point_index, point in enumerate(points):
        if candidate_indices is not None:
            triangle_indices = candidate_indices[point_index].astype(np.int64)
        else:
            triangle_indices = np.arange(triangles.shape[0], dtype=np.int64)
        for triangle_index in triangle_indices:
            current_triangle = triangles[triangle_index]
            b1 = np.sign(
                np.subtract(
                    np.multiply(
                        np.subtract(point[0], current_triangle[1, 0]),
                        np.subtract(current_triangle[0, 1], current_triangle[1, 1]),
                    ),
                    np.multiply(
                        np.subtract(current_triangle[0, 0], current_triangle[1, 0]),
                        np.subtract(point[1], current_triangle[1, 1]),
                    ),
                )
            )

            b2 = np.sign(
                np.subtract(
                    np.multiply(
                        np.subtract(point[0], current_triangle[2, 0]),
                        np.subtract(current_triangle[1, 1], current_triangle[2, 1]),
                    ),
                    np.multiply(
                        np.subtract(current_triangle[1, 0], current_triangle[2, 0]),
                        np.subtract(point[1], current_triangle[2, 1]),
                    ),
                )
            )

            b3 = np.sign(
                np.subtract(
                    np.multiply(
                        np.subtract(point[0], current_triangle[0, 0]),
                        np.subtract(current_triangle[2, 1], current_triangle[0, 1]),
                    ),
                    np.multiply(
                        np.subtract(current_triangle[2, 0], current_triangle[0, 0]),
                        np.subtract(point[1], current_triangle[0, 1]),
                    ),
                )
            )
            if (np.abs(b1 + b2 + b3) + ((b1 == 0) + (b2 == 0) + (b3 == 0))) == 3:
                point_in[point_index] = triangle_index
                break

    return point_in


@njit
def points_in_triangles(points: np.array, triangles: np.array,
                        candidate_indices: Optional[np.array] = None) -> np.array:
    """
    Args:
        points (np.ndarray): array of points with shape (N, 2)
        triangles (np.ndarray): array of triangles. Shape (M, 3, 2)
        candidate_indices (np.ndarray): array of candidate triangle indices for each point. Shape (N, K), where
        K is the number of candidate triangles for each point. If None, all triangles are checked for each point.

    Returns:
        point_in (np.ndarray): Array with triangle index for each point. Shape (N, ) containing the index of the
        triangle that contains the point. If the point is not in any triangle, the index is -1.
    """
    point_in = np.empty(points.shape[0], dtype=np.int64)
    point_in.fill(-1)  # -1 means no triangle found for this point

    # for each point, check if it lies inside any of the triangles by checking whether the point is on the same side of
    # the triangle's edges as the triangle's vertices
    for point_index, point in enumerate(points):
        if candidate_indices is not None:
            triangle_indices = candidate_indices[point_index].astype(np.int64)
        else:
            triangle_indices = np.arange(triangles.shape[0], dtype=np.int64)
        for triangle_index in triangle_indices:
            current_triangle = triangles[triangle_index]
            b1 = np.sign(
                np.subtract(
                    np.multiply(
                        np.subtract(point[0], current_triangle[1, 0]),
                        np.subtract(current_triangle[0, 1], current_triangle[1, 1]),
                    ),
                    np.multiply(
                        np.subtract(current_triangle[0, 0], current_triangle[1, 0]),
                        np.subtract(point[1], current_triangle[1, 1]),
                    ),
                )
            )

            b2 = np.sign(
                np.subtract(
                    np.multiply(
                        np.subtract(point[0], current_triangle[2, 0]),
                        np.subtract(current_triangle[1, 1], current_triangle[2, 1]),
                    ),
                    np.multiply(
                        np.subtract(current_triangle[1, 0], current_triangle[2, 0]),
                        np.subtract(point[1], current_triangle[2, 1]),
                    ),
                )
            )

            b3 = np.sign(
                np.subtract(
                    np.multiply(
                        np.subtract(point[0], current_triangle[0, 0]),
                        np.subtract(current_triangle[2, 1], current_triangle[0, 1]),
                    ),
                    np.multiply(
                        np.subtract(current_triangle[2, 0], current_triangle[0, 0]),
                        np.subtract(point[1], current_triangle[0, 1]),
                    ),
                )
            )
            if (np.abs(b1 + b2 + b3) + ((b1 == 0) + (b2 == 0) + (b3 == 0))) == 3:
                point_in[point_index] = triangle_index
                break

    return point_in


##################
# Test functions #
##################

def main():
    import time
    np.random.seed(0)
    points = np.random.uniform((0.0, 0.0), (1.0, 1.0), size=(10000, 2))
    triangles = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0], [0.5, 0.5]],
            [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]],
            [[1.0, 0.0], [1.0, 1.0], [0.5, 0.5]],
            [[1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
        ],
    )

    first_call = time.perf_counter()
    point_in = points_in_triangles(points, triangles)
    second_call = time.perf_counter()
    point_in = points_in_triangles(points, triangles)
    done = time.perf_counter()

    print(f"First call with jit compile: {second_call - first_call}\n Second call: {done - second_call}")

    from skfem import MeshTri

    for initial_resolution in np.arange(10) + 1:
        print(f"Initial Resolution: {initial_resolution}")
        # higher numbers correspond to more mesh refinement steps
        mesh = MeshTri.init_symmetric().refined(int(initial_resolution))
        triangles = mesh.p[:, mesh.t].T

        print(f"Larger mesh: {mesh}")
        first_call = time.perf_counter()
        point_in = points_in_triangles(points, triangles)
        second_call = time.perf_counter()
        point_in = points_in_triangles(points, triangles)
        done = time.perf_counter()

        print(f"First call with jit compile: {second_call - first_call}\n Second call: {done - second_call}")


if __name__ == "__main__":
    main()
