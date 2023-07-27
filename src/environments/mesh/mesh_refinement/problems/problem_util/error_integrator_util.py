from typing import Optional

import numpy as np
import torch
import torch_scatter
from numpy import ndarray
from skfem import Basis

from util.function import get_triangle_areas_from_indices
from util.point_in_geometry import points_on_edges


def get_integration_weights(basis: Basis) -> np.array:
    """
    Calculates the integration weights for the given mesh.
    Scale the integration weights by the area of the corresponding reference elements
    Args:
        basis: Basis. Contains the (triangular) mesh to calculate the integration weights for.

    Returns: The integration weights for the given mesh.

    """
    integration_weights = get_triangle_areas_from_indices(positions=basis.mesh.p.T,
                                                          triangle_indices=basis.mesh.t.T)
    integration_weights = integration_weights / np.sum(integration_weights)  # normalize
    return integration_weights


def find_integration_points_on_edges(coarse_basis, integration_points: np.array,
                                     corresponding_elements: Optional[np.array] = None) -> np.array:
    """
    Finds for each integration point whether it lies on an edge of the coarse mesh and if so, which elements
    correspond to the edge.
    Args:
        coarse_basis: The basis of the coarse mesh to calculate the error estimate for based on the fine integration
            mesh.
        integration_points: The integration points of the fine integration mesh.
            An array of shape (num_integration_points, 2)
        corresponding_elements: (Optional) The elements of the coarse mesh that contain the integration points. Used to
            determine the elements that contain the edges that contain the integration points without having to do
            a full search. An array of shape (num_integration_points,) containing the indices of the elements.
    Returns: An array of edge indices with shape (num_points, ) containing the index of the edge that contains the
    point. If the point does not lie on any edge, the index is -1.

    """
    edges = coarse_basis.mesh.facets
    edge_positions = coarse_basis.mesh.p[:, edges]

    if corresponding_elements is not None:
        candidate_indices = coarse_basis.mesh.t2f.T[corresponding_elements]
    else:
        candidate_indices = None
    edge_membership = points_on_edges(points=integration_points, edges=edge_positions,
                                      candidate_indices=candidate_indices)
    # array of edge indices with shape (num_points, ) containing the index of the edge that contains the point.
    return edge_membership


def get_integrated_differences(pointwise_differences: np.array,
                               integration_weights: np.array,
                               corresponding_edge_elements: np.array,
                               corresponding_full_elements: np.array,
                               valid_edge_indices: np.array,
                               num_coarse_elements: int, error_metric: str) -> np.array:
    """
    Calculates the integrated differences between the coarse evaluation and the reference evaluation.
    For this, the differences are summed up over the corresponding coarse elements by summing over
    integration points on edges and integration points inside elements.
    Args:
        pointwise_differences: The absolute error/difference per integration point.
            Shape (#integration points, #solution dimensions)
        integration_weights: The weights of the integration points. Shape (#integration points,)
        corresponding_edge_elements: The elements of the coarse mesh which have edges with integration points on
        them. Array of shape (#integration_points_on_edges, 2)
        corresponding_full_elements: The elements of the coarse mesh which have integration points on them.
            Array of shape (#integration_points_on_elements,)
        valid_edge_indices: The indices of the integration points that lie on edges. Array of shape
            (#integration_points_on_edges,). The negation of this array can be used to get the indices of the
            integration points that lie inside elements.
        num_coarse_elements: The number of faces/elements of the coarse mesh.
        error_metric: The metric to use for calculating the differences between the coarse evaluation and
            the reference evaluation. Can be "mean" or "squared".

    Returns: elementwise_differences, which are the integrated differences between the coarse evaluation and the
        reference evaluation per *coarse* element as an array of shape (#elements, #solution dimensions).

    """
    assert error_metric in ["mean", "squared", "maximum"], f"Unknown error metric: {error_metric}"

    if "squared" in error_metric:
        pointwise_differences = pointwise_differences ** 2

    if "maximum" in error_metric:
        scatter_operation = lambda *args, **kwargs: torch_scatter.scatter_max(*args, **kwargs)[0]
    else:
        scatter_operation = torch_scatter.scatter_sum
        pointwise_differences = pointwise_differences * integration_weights[:, None]

    # sum-aggregate over corresponding faces by adding up the differences of the integration points
    # differences when mapping the current_basis to the reference_basis' integration points for each point
    # we do this in torch because it is a lot faster than numpy
    pointwise_differences = torch.tensor(pointwise_differences)
    corresponding_edge_elements = torch.tensor(corresponding_edge_elements)
    corresponding_full_elements = torch.tensor(corresponding_full_elements)
    edge_differences = pointwise_differences[valid_edge_indices]
    in_element_differences = pointwise_differences[~valid_edge_indices]
    left_edge_differences = scatter_operation(edge_differences, corresponding_edge_elements[:, 0],
                                              dim=0, dim_size=num_coarse_elements)
    right_edge_differences = scatter_operation(edge_differences, corresponding_edge_elements[:, 1],
                                               dim=0, dim_size=num_coarse_elements)
    element_differences = scatter_operation(in_element_differences, corresponding_full_elements,
                                            dim=0, dim_size=num_coarse_elements)

    if "maximum" in error_metric:
        elementwise_difference = torch.max(left_edge_differences, right_edge_differences)
        elementwise_difference = torch.max(elementwise_difference, element_differences)
    else:
        elementwise_difference = ((left_edge_differences + right_edge_differences) / 2) + element_differences
    elementwise_difference = elementwise_difference.numpy()

    return elementwise_difference


def probes_from_elements(basis: Basis, x: ndarray, cells: ndarray):
    """
    Return matrix which acts on a solution vector to find its values. Uses pre-computed cell indices.
    on points `x`.
    Args:
        basis: The basis to use for the interpolation
        x: The points to interpolate to
        cells: Cell indices per point. A cell index of -1 means that the point is not in any cell.

    Returns:

    """
    import sys
    if "pyodide" in sys.modules:
        from scipy.sparse.coo import coo_matrix
    else:
        from scipy.sparse import coo_matrix
    pts = basis.mapping.invF(x[:, :, np.newaxis], tind=cells)
    phis = np.array(
        [
            basis.elem.gbasis(basis.mapping, pts, k, tind=cells)[0]
            for k in range(basis.Nbfun)
        ]
    ).flatten()
    return coo_matrix(
        (
            phis,
            (
                np.tile(np.arange(x.shape[1]), basis.Nbfun),
                basis.element_dofs[:, cells].flatten(),
            ),
        ),
        shape=(x.shape[1], basis.N),
    )
