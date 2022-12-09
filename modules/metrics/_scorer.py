from ._regression import mean_square_error

def get_scorer(scoring: str):
    """Get a scorer from a string."""
    try:
        scorer = _SCORER[scoring]
    except KeyError:
        raise ValueError(
            f"{scoring} is not a valid scorer name."
        )
    return scorer


_SCORER = dict(
    mean_square_error = mean_square_error
)