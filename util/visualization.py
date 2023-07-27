import numpy as np
from plotly import graph_objects as go
from plotly.basedatatypes import BaseTraceType

from util.types import *


def get_layout(boundary: np.array, title: Optional[str] = None,
               extra_str: Optional[str] = None) -> go.Layout:
    """
    Get a layout for a plotly figure.
    Args:
        boundary: The boundary of the plot. The format is [x_min, y_min, x_max, y_max]
        title:
        extra_str:

    Returns:

    """
    if extra_str is not None:
        return go.Layout(yaxis=dict(range=boundary[1::2],
                                    scaleanchor="x",
                                    scaleratio=1),
                         xaxis=dict(range=boundary[0::2]),
                         title=title,
                         font={"size": 9},
                         annotations=[
                             go.layout.Annotation(
                                 text=extra_str,
                                 align='right',
                                 valign='bottom',
                                 showarrow=False,
                                 xref='paper',
                                 yref='paper',
                                 x=0.90,
                                 y=0.02,
                             )
                         ]
                         )
    else:
        return go.Layout(yaxis=dict(range=boundary[1::2],
                                    scaleanchor="x",
                                    scaleratio=1),
                         xaxis=dict(range=boundary[0::2]),
                         title=title
                         )


def ellipse(x_center=0, y_center=0, ax1=[1, 0], ax2=[0, 1], a=1, b=1, N=100):
    # x_center, y_center the coordinates of ellipse center
    # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
    # a, b the ellipse parameters
    if np.linalg.norm(ax1) != 1 or np.linalg.norm(ax2) != 1:
        raise ValueError('ax1, ax2 must be unit vectors')
    if abs(np.dot(ax1, ax2)) > 1e-06:
        raise ValueError('ax1, ax2 must be orthogonal vectors')
    t = np.linspace(0, 2 * np.pi, N)
    # ellipse parameterization with respect to a system of axes of directions a1, a2
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    # rotation matrix
    R = np.array([ax1, ax2]).T
    # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
    xp, yp = np.dot(R, [xs, ys])
    x = xp + x_center
    y = yp + y_center
    return x, y


def plot_2d_covariance(*, mean, covariance_matrix, chisquare_val=2.4477, **kwargs) -> List[BaseTraceType]:
    """
    Draw the covariance matrix for a 2d Gaussian as an ellipsoid.
    Args:
        mean: Mean of the gaussian.
        covariance_matrix: Covariance matrix to draw
        chisquare_val: Scaling of the shown covariance ellipsoid. The default value of 2.4477 corresponds to
        a 95% confidene interval
        **kwargs: Additional arguments to give to plotly
    Returns: A List containing a single go.Scatter trace that contains the ellipsoid
    """
    (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(covariance_matrix)
    phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])

    a = chisquare_val * np.sqrt(largest_eigval)
    b = chisquare_val * np.sqrt(smallest_eigval)

    ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi))
    ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi))

    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R

    kwargs.setdefault("showlegend", False)  # disable legend by default
    ellipsoid_trace = go.Scatter(
        x=mean[0] + r_ellipse[:, 0],
        y=mean[1] + r_ellipse[:, 1],
        mode='lines', **kwargs)
    return [ellipsoid_trace]


def plot_circle(center: np.ndarray, radius: float, N: int, **kwargs) -> BaseTraceType:
    angles = np.array([2 * np.pi * i / float(N) for i in range(N + 1)])
    points = np.array([np.cos(angles),
                       np.sin(angles)]).T
    points *= radius
    points += center
    xs, ys = points.T
    return go.Scatter(
        x=xs, y=ys,
        mode='lines',
        **kwargs
    )


def plotly_figure_to_array(figure):
    import io
    from PIL import Image
    #  convert Plotly figure to numpy array
    fig_bytes = figure.to_image(format="jpg")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def matplotlib_figure_to_image(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    from PIL import Image
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
