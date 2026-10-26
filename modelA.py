""" Model 1:
Assumptions:
- A >> x(t) => A + x(t) ~ A
- x(0) > 0
- x(t) > 0 for all t
- K and A are positive constants

Parameters:
- K = 1.0
- A = 6
- x(0) = 0.0025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

def MichelisMentenExplicitModelA(K: float, A: float, x_start: float, t) -> float:
    """
    Computes the Michelis-Menten equation value for given K, A, and x_start,
    assuming that A + x(t) ~ A.
    
    Parameters:
    - K: maximum reaction rate
    - A: Michaelis constant
    - x_start: initial concentration of substrate
    - t: time
    """
    t = np.asarray(t, dtype=float)
    return x_start * np.exp(-K * t / A)

def MichelisMentenGeneralizedModel(K: float, A: float, x_start: float, t: float) -> float:
    """
    Computes the Michelis-Menten equation value for given K, A, and x_start.
    
    Parameters:
    - K: maximum reaction rate
    - A: Michaelis constant
    - x_start: initial concentration of substrate
    - t: time

    We work in log space because Lambert W function can quickly overflow when x_start is large relative to A.
    Since x(t) > 0 and A > 0, we use the positive branch of Lambert W function.
    """
    t = np.asarray(t, dtype=float)
    L = np.log(x_start / A) + (x_start - K * t) / A
    out = np.empty_like(L)

    small = L < 50
    out[small] = np.real(lambertw(np.exp(L[small])))

    w = L[~small] - np.log(L[~small])
    for _ in range(50):
        w = w - (w + np.log(w) - L[~small]) / (1 + 1 / w)
    out[~small] = w

    return A * out

if __name__ == "__main__":
    K, A, x0 = 1.0, 6.0, 0.0025
    t = np.linspace(0, 60, 400)

    x_exact = MichelisMentenGeneralizedModel(K, A, x0, t)
    x_approx = MichelisMentenExplicitModelA(K, A, x0, t)
    residual = x_exact - x_approx

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    fig.suptitle("Comparison of Michelis-Menten Models ($K = 1.0, A = 6, x(0) = 0.0025$)")

    ax1.plot(t, x_exact, label="Exact")
    ax1.plot(t, x_approx, "--", label=r"$A + x(t) \sim A$ Approximation")
    ax1.set_ylabel("$x(t)$")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(t, residual, color="C3")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("$t$"); ax2.set_ylabel("Residual: Exact $-$ Approximate")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()