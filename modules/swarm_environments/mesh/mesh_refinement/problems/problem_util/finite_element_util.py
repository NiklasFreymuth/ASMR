from skfem import Basis, ElementTriP1


def to_scalar_basis(basis, linear: bool) -> Basis:
    """
    Returns: The scalar basis used for the error estimation via integration.

    """
    if linear:
        element = ElementTriP1()
    else:
        element = basis.elem
        while hasattr(element, "elem"):
            element = element.elem
    scalar_basis = basis.with_element(element)
    return scalar_basis
