""" Model 2:
Assumptions:
- A << x(t) => A + x(t) ~ x(t)
- x(0) > 0
- x(t) > 0 for all t
- K and A are positive constants

Parameters:
- K = 1
- A = 0.005
- x(0) = 0.025
"""

import numpy as np
import matplotlib.pyplot as plt
from modelA import MichelisMentenGeneralizedModel

def MichelisMentenExplicitModelB(K: float, x_start: float, t) -> float:
    """
    Computes the Michelis-Menten equation value for given K and x_start,
    assuming that A + x(t) ~ x(t).
    
    Parameters:
    - K: maximum reaction rate
    - x_start: initial concentration of substrate
    - t: time
    """
    t = np.asarray(t, dtype=float)
    return x_start - K * t

if __name__ == "__main__":
    K, A, x0 = 1.0, 0.005, 0.025
    t_empty = x0 / K 
    t = np.linspace(0, 1.5 * t_empty, 400)

    x_exact = MichelisMentenGeneralizedModel(K, A, x0, t)
    x_approx = MichelisMentenExplicitModelB(K, x0, t)
    residual = x_exact - x_approx

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    fig.suptitle("Comparison of Michelis-Menten Models ($K = 1.0, A = 0.005, x(0) = 0.025$)")

    ax1.plot(t, x_exact, label="Exact")
    ax1.plot(t, x_approx, "--", label=r"$A + x(t) \sim x(t)$ Approximation")
    ax1.axvline(t_empty, color="black", lw=2, ls=":", label=r"$t = x_0/K$")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("$x(t)$")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(t, residual, color="C3")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("$t$"); ax2.set_ylabel("Residual: Exact $-$ Approximate")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()