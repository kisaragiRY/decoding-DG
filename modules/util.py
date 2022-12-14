import numpy as np
from numpy.typing import NDArray

# ---------- kernel ---------
def gauss(xx: NDArray, mu: float = 0, sigma: float = .2):
    """A Gaussian kernel."""
    return 1 / ((2 * np.pi) ** 2 * sigma) * np.exp(- (xx - mu) ** 2 / (2 * sigma ** 2))
