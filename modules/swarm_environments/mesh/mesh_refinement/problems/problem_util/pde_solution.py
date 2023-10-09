from dataclasses import dataclass

import numpy as np


@dataclass
class PDESolution:
    """
    A class that stores the solution of a PDE as
    * integration_points, the points at which the solution is evaluated as a vector of shape (2, num_integration_points)
    * integration_weights, the weights of the integration points as a vector of shape (num_integration_points,)
    * reference_evaluation, the solution evaluated at the integration points as a vector of shape
        (num_integration_points, solution_dimension)
    """
    integration_points: np.array
    integration_weights: np.array
    reference_evaluation: np.array
