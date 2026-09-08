"""Statistical primitives shared by outer-loop applications."""

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class WilsonCI:
    lower: float
    upper: float
    center: float
    n: int
    k: int  # successes

    def __repr__(self) -> str:
        return (
            f"WilsonCI(p={self.center:.3f}, "
            f"[{self.lower:.3f}, {self.upper:.3f}], n={self.n})"
        )


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> WilsonCI:
    """
    Compute the Wilson score confidence interval.

    Args:
        k: number of successes
        n: number of trials
        alpha: significance level (1-alpha is the confidence level)

    Returns:
        WilsonCI with lower/upper bounds
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError("expected 0 <= k <= n")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if n == 0:
        return WilsonCI(0.0, 1.0, 0.0, 0, 0)

    z = _z_score(1 - alpha / 2)
    p_hat = k / n
    z2 = z * z
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    margin = (
        z
        * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
        / (1 + z2 / n)
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return WilsonCI(lower=lower, upper=upper, center=center, n=n, k=k)


def gated_accept(ci_candidate: WilsonCI, ci_incumbent: WilsonCI) -> bool:
    """
    Gated acceptance test: candidate is certified better than incumbent iff
    the lower bound of candidate exceeds the upper bound of incumbent.

    This conservative rule accepts only when the intervals do not overlap.
    """
    return ci_candidate.lower > ci_incumbent.upper


def illusion_probability(n: int, epsilon: float, q: float) -> float:
    """
    Theoretical probability of ratchet illusion (Theorem 3.1).
    P(F_hat(h') > F_hat(h)) when F(h') = q - epsilon.

    Args:
        n: number of evaluation episodes
        epsilon: true performance gap (h' is epsilon worse)
        q: baseline completion rate

    Returns:
        Illusion probability
    """
    if n < 1:
        raise ValueError("n must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1")
    mean_diff = -epsilon  # h' is epsilon worse
    std_diff = math.sqrt(2 * q * (1 - q) / n)
    # P(difference > 0) = P(Z > -mean_diff/std_diff) = 1 - Phi(-mean_diff/std_diff)
    z = -mean_diff / std_diff
    return 1 - NormalDist().cdf(z)


def _z_score(p: float) -> float:
    """Inverse normal CDF (percent-point function)."""
    return NormalDist().inv_cdf(p)


def sample_size_for_delta(delta: float, q0: float, alpha: float = 0.05) -> int:
    """
    Minimum N such that Wilson CI half-width < delta/2 at completion rate q0.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if not 0 < q0 < 1:
        raise ValueError("q0 must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    z = _z_score(1 - alpha / 2)
    # Wilson margin ≈ z * sqrt(q(1-q)/n) / (1 + z^2/n)
    # For large n: margin ≈ z * sqrt(q(1-q)/n)
    # Set margin = delta/2 and solve for n:
    n = math.ceil((z / (delta / 2)) ** 2 * q0 * (1 - q0))
    return n
