import numpy as np
from plotly import graph_objects as go
from plotly.basedatatypes import BaseTraceType
from skfem import Basis, Mesh

from modules.swarm_environments.mesh.mesh_refinement.mesh_refinement_util import angle_to
from modules.swarm_environments.util.function import get_triangle_areas_from_indices
from typing import List, Callable
from modules.swarm_environments.util.visualization import get_layout

"""
This class handles the visualization for the MeshRefinement environment.
While there is built-in visualization utility in scikit-FEM, it uses matplotlib as a backend and is thus
unfortunately not compatible with our framework and wandb.
We instead re-write the important parts of it here.

"""


def contour_trace_from_element_values(mesh: Mesh, element_evaluations: np.array,
                                      trace_name: str = "Value per Agent") -> List[BaseTraceType]:
    """
        Creates a list of plotly traces for the elements of a mesh
        Args:
            mesh: A scikit-fem mesh
            element_evaluations: A numpy array of shape (mesh.nelements,) containing the evaluations of the elements
            trace_name: Name/Title of the trace
        Returns:
            A list of plotly traces

    """
    vertex_positions = mesh.p.T
    element_indices = mesh.t.T
    element_positions = vertex_positions[element_indices]
    element_midpoints = np.mean(element_positions, axis=1)

    marker_trace = (go.Scatter(
        x=element_midpoints[:, 0], y=element_midpoints[:, 1],
        text=element_evaluations,
        marker=dict(
            size=10,
            cmin=np.nanmin(element_evaluations),
            cmax=np.nanmax(element_evaluations),
            color=element_evaluations,
            colorbar=dict(yanchor="top",
                          y=1,
                          x=0,  # put colorbar on the left
                          ticks="outside"),
            colorscale="Viridis",
        ),
        name=trace_name,
        mode="markers"))

    traces = [marker_trace]
    return traces


def get_evaluation_heatmap_from_basis(basis: Basis,
                                      evaluation_function: Callable[[np.array], np.array],
                                      normalize_by_element_area: bool = False,
                                      resolution: int = 101) -> List[BaseTraceType]:
    """
    Draws a plotly.graph_objects.Contour()-trace of the solution interpolated by the mesh at the given positions.
    For every element, the values inside the element are some interpolation of the solution values of its nodes/vertices,
     where the concrete interpolation method depends on the kind of element that is used as a basis.
    Args:
        basis: A scikit-FEM basis. Consists of a mesh and an element used for each of the elements
        evaluation_function: A function that takes a numpy array of shape (n,2) and returns a numpy array of shape (n,)
        resolution: Resolution at which to interpolate. Will interpolate a grid of resolution x resolution points
        normalize_by_element_area: If True, the values will be normalized by the area of each mesh element.

    Returns: A plotly trace consisting of the outlines of the mesh

    """
    mesh = basis.mesh
    x_coordinates = mesh.p[0]
    y_coordinates = mesh.p[1]

    # Define a regular grid over the data
    # this grid is a big list of shape (2, resolution**2) of (x,y)-coordinates
    x_resolution = np.linspace(x_coordinates.min(), x_coordinates.max(), resolution)
    y_resolution = np.linspace(y_coordinates.min(), y_coordinates.max(), resolution)
    x_resolution, y_resolution = np.meshgrid(x_resolution, y_resolution)

    flattened_points = np.vstack((x_resolution.flatten(), y_resolution.flatten()))

    plot_evaluation_grid = evaluation_function(flattened_points)

    # find corresponding elements for all elements

    element_areas = get_triangle_areas_from_indices(positions=mesh.p.T, triangle_indices=mesh.t.T)

    # find corresponding elements and filter out grid points that are outside the boundary
    corresponding_elements = mesh.element_finder()(x=flattened_points[0], y=flattened_points[1])
    plot_evaluation_grid[corresponding_elements == -1] = None

    if normalize_by_element_area:
        plot_evaluation_grid = plot_evaluation_grid / element_areas[corresponding_elements]

    plot_evaluation_grid = plot_evaluation_grid.reshape(resolution, resolution)

    # we then draw the evaluated grid
    contour_trace = go.Contour(x=x_resolution[1], y=y_resolution[:, 0],
                               z=plot_evaluation_grid,
                               colorscale="Jet",  # nice colors
                               connectgaps=False,
                               zmin=np.nanmin(plot_evaluation_grid),
                               zmax=np.nanmax(plot_evaluation_grid),
                               line={"width": 0.0},  # do not show contour lines, as we want continous contours
                               contours_coloring='heatmap',  # use continous contours
                               colorbar=dict(yanchor="top", y=1, x=0,  # put colorbar on the left
                                             ticks="outside")
                               )
    return [contour_trace]


def get_points_at_boundary(flattened_points: np.ndarray, in_boundary_mask: np.array, resolution: int):
    """
        Calculates all points of a grid that are outside the mesh but have a direct neighbor in the mesh
        Args:
            flattened_points: Grid points - Numpy array of shape (resolution*resolution, 2)
            in_boundary_mask: Mask of points that are inside the mesh
            resolution: Grid resolution
        Returns: Numpy array of all points at boundary and numpy array of all indexes of these points in flattened
                 points

        """
    points_at_boundary, index_points_at_boundary = [], []

    for index in range(len(flattened_points.T)):
        # continue if point is on the left or right boundary of the grid
        if index == 0 or index == (len(flattened_points.T) - 1):
            continue

        horizontal_neighbor_inside_mesh = \
            not in_boundary_mask[index] and \
            (in_boundary_mask[index + 1] or in_boundary_mask[index - 1])

        # continue if point is on the upside or downside boundary of the grid
        if index < resolution or (len(flattened_points.T) - (resolution + 1)) < index:
            continue

        vertical_neighbor_inside_mesh = \
            not in_boundary_mask[index] and \
            (in_boundary_mask[index + resolution] or in_boundary_mask[index - resolution])

        if horizontal_neighbor_inside_mesh or vertical_neighbor_inside_mesh:
            points_at_boundary.append(flattened_points.T[index])
            index_points_at_boundary.append(index)

    return np.array(points_at_boundary), np.array(index_points_at_boundary)


def get_boundary_plot(boundary_points: np.ndarray):
    """
        Plots the boundary points
        Args:
            boundary_points: Boundary poits
        Returns:
            A plotly figure

        """
    # sort points based on angle to center (0.5, 0.5) - to construct Polygon
    sorted_boundary_points = np.array(sorted(boundary_points, key=angle_to, reverse=True))

    # construct polygon path
    x, y = sorted_boundary_points[:, 0], sorted_boundary_points[:, 1]

    path = f'M {x[0]}, {y[0]}'
    for index in range(1, len(sorted_boundary_points)):
        path += f'L{x[index]},{y[index]} '
    path += ' Z'

    # build figure
    fig = go.Figure()

    # Update axes properties
    fig.update_xaxes(range=[0, 1], zeroline=False)
    fig.update_yaxes(range=[0, 1], zeroline=False)

    # Add shapes
    fig.update_layout(
        shapes=[
            # Polygon
            dict(
                type="path",
                path=path,
                line_color="black",
            ),
        ]
    )

    return fig


def get_mesh_traces(mesh: Mesh, color: str = "black", showlegend: bool = True) -> List[BaseTraceType]:
    """
    Draws a plotly trace depicting the edges/facets of a scikit fem triangle mesh
    Args:
        mesh: A scikit basis. Contains a basis.mesh attribute that has properties
         * mesh.facets of shape (2, num_edges) that lists indices of edges between the mesh, and
         * mesh.p of shape (2, num_nodes) for coordinates between those indices
        color: Color of scatter plot
        showlegend: Whether to show the legend

    Returns: A list of plotly traces [mesh_trace, node_trace], where mesh_trace consists of the outlines of the mesh
        and node_trace consists of an overlay of all nodes

    """
    facets = mesh.facets
    vertices = mesh.p

    node_trace = go.Scatter(x=vertices[0], y=vertices[1],
                            mode="markers", marker={"size": 2, "color": color},
                            # "cmid": np.mean(colors)},
                            name="Nodes",
                            showlegend=showlegend)

    num_edges = facets.shape[-1]
    edge_x_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_y_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_x_positions[0::3] = vertices[0, facets[0]]
    edge_x_positions[1::3] = vertices[0, facets[1]]
    edge_y_positions[0::3] = vertices[1, facets[0]]
    edge_y_positions[1::3] = vertices[1, facets[1]]
    edge_trace = go.Scatter(x=edge_x_positions, y=edge_y_positions,
                            mode="lines",
                            line=dict(color=color, width=1),
                            name="Mesh",
                            showlegend=showlegend)
    return [edge_trace, node_trace]


def scalar_per_element_plot(mesh: Mesh, scalar_per_element: np.array, title: str = "Displacement") -> go.Figure:
    """
    Show the scalar quantity per element of a mesh
    Args:
        mesh: The mesh to plot
        scalar_per_element: A scalar quantity of the mesh per mesh element. Shape (mesh.nelements,)
        title: Title of the plot

    Returns: A plotly figure displaying the quantity as a scatter plot with a colorbar and a mesh outline

    """
    contour_trace = contour_trace_from_element_values(mesh=mesh, element_evaluations=scalar_per_element,
                                                      trace_name=title)
    mesh_trace = get_mesh_traces(mesh)
    traces = contour_trace + mesh_trace

    boundary = np.concatenate((mesh.p.min(axis=1), mesh.p.max(axis=1)), axis=0)
    layout = get_layout(boundary=boundary,  # min, max of deformed mesh
                        title=title)
    displacement_plot = go.Figure(data=traces,
                                  layout=layout)
    return displacement_plot
