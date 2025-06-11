import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r: float, x0: float, n_steps: int, discard: int = 0) -> np.ndarray:
    """
    Generate a logistic-map time series.

    Parameters
    ----------
    r : float
        Control parameter (0 < r ≤ 4).
    x0 : float
        Initial condition (0 < x0 < 1).
    n_steps : int
        Number of points to return (after optional transient discard).
    discard : int, optional
        Iterations to skip at the beginning (useful for flushing transients).

    Returns
    -------
    np.ndarray
        Array of length `n_steps` containing the logistic-map trajectory.
    """
    if not (0 < r <= 4):
        raise ValueError("r must be in (0, 4].")
    if not (0 < x0 < 1):
        raise ValueError("x0 must be in (0, 1).")
    total = n_steps + discard
    xs = np.empty(total, dtype=float)
    xs[0] = x0
    for i in range(1, total):
        xs[i] = r * xs[i - 1] * (1 - xs[i - 1])
    return xs[discard:]           # drop transient

# Example
if __name__ == "__main__":
    r      = 3.9        # chaotic regime
    x0     = 0.5        # initial state
    n      = 25        # length of trajectory
    burn   = 5        # throw away first 100 points

    rs = np.linspace(0, 4, 401)[1:-1]

    for r in rs:
        series = logistic_map(r, x0, n, discard=burn)

        plt.scatter(r * np.ones_like(series), series, s=2)
    
    plt.tight_layout()
    plt.show()

