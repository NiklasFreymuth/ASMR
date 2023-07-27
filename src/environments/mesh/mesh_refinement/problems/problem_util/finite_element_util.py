from skfem import Basis, ElementTriP1


def scalar_linear_basis(basis) -> Basis:
    """
    Returns: The scalar basis used for the error estimation via integration.

    """
    scalar_linear_basis = basis.with_element(ElementTriP1())
    return scalar_linear_basis
