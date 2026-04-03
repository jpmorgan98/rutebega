"""fixed_point_gmres.py

Reusable fixed-point acceleration for transport/source-iteration style solvers.

NOT matrix GMRES.
Small-history residual-minimization accelerator where you only have:

    x_old  -> previous iterate
    x_new  -> raw next iterate from your transport sweep / fixed-point map

Typical use
-----------
from fixed_point_gmres import make_next_gmres

next_gmres = make_next_gmres(m=5, damping=1.0)

xo = initial_guess
while not converged:
    xn = transport_loop(xo)   # raw fixed-point / transport sweep
    xn = next_gmres(xn, xo)   # accelerated iterate
    xo = xn

If you can compute a better residual than (xn - xo), you can pass it as:

    xn = next_gmres(xn, xo, residual=my_residual)

The accelerator stores a short history internally, so create one instance per
solve or call reset() before reusing it for a different problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray | Sequence[float]


@dataclass
class FixedPointGMRES:
    """Small-history residual-minimization accelerator for fixed-point solvers.

    Parameters
    ----------
    m : int, default=5
        Maximum number of history vectors to keep.
    damping : float, default=1.0
        Successive over Relaxation factor between the raw iterate and the accelerated iterate:
            x_out = (1 - damping) * x_raw + damping * x_acc
        Start with 1.0; reduce to 0.5-0.8 if the iteration gets noisy.
    regularization : float, default=1e-12
        Small diagonal shift added to the residual Gram matrix to improve
        robustness when the history becomes nearly linearly dependent.
    max_weight_norm : float | None, default=1e6
        Optional safety check. If the DIIS/Anderson weights become extremely
        large, the accelerator falls back to the raw iterate.
    """

    m: int = 5
    damping: float = 1.0
    regularization: float = 1.0e-12
    max_weight_norm: Optional[float] = 1.0e6
    _x_hist: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _r_hist: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _shape: Optional[Tuple[int, ...]] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Clear all stored history."""
        self._x_hist.clear()
        self._r_hist.clear()
        self._shape = None

    def __call__(
        self,
        xn: ArrayLike,
        xo: ArrayLike,
        residual: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Return an accelerated iterate.

        Parameters
        ----------
        xn : array_like
            Raw next iterate from the transport loop / fixed-point map.
        xo : array_like
            Previous iterate.
        residual : array_like, optional
            Residual associated with the current iterate. If omitted, the method
            uses (xn - xo), which is the fixed-point residual available from the
            iteration itself.

        Returns
        -------
        numpy.ndarray
            Accelerated iterate with the same shape as xn.
        """

        x_raw = np.asarray(xn, dtype=float)
        x_old = np.asarray(xo, dtype=float)

        if x_raw.shape != x_old.shape:
            raise ValueError("xn and xo must have the same shape")

        if self._shape is None:
            self._shape = x_raw.shape
        elif x_raw.shape != self._shape:
            raise ValueError(
                "Input shape changed. Call reset() before reusing the accelerator "
                "on a different problem."
            )

        x_raw_vec = x_raw.reshape(-1).copy()
        x_old_vec = x_old.reshape(-1)

        if residual is None:
            r_vec = x_raw_vec - x_old_vec
        else:
            r_arr = np.asarray(residual, dtype=float)
            if r_arr.shape != x_raw.shape:
                raise ValueError("residual must have the same shape as xn")
            r_vec = r_arr.reshape(-1).copy()

        # Store current raw iterate and current residual-like vector.
        self._x_hist.append(x_raw_vec)
        self._r_hist.append(r_vec)

        if len(self._x_hist) > self.m:
            self._x_hist.pop(0)
            self._r_hist.pop(0)

        # Need at least two history entries before acceleration does anything.
        k = len(self._x_hist)
        if k < 2:
            return x_raw.copy()

        # Residual Gram matrix G_ij = <r_i, r_j>.
        R = np.column_stack(self._r_hist)          # shape (n, k)
        G = R.T @ R                                # shape (k, k)
        if self.regularization > 0.0:
            G = G + self.regularization * np.eye(k)

        # Constrained least-squares / DIIS system:
        #   minimize || sum_i c_i r_i ||  subject to sum_i c_i = 1
        K = np.empty((k + 1, k + 1), dtype=float)
        K[:k, :k] = G
        K[:k, k] = 1.0
        K[k, :k] = 1.0
        K[k, k] = 0.0

        rhs = np.zeros(k + 1, dtype=float)
        rhs[k] = 1.0

        try:
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
            coeffs = sol[:k]
        except np.linalg.LinAlgError:
            return x_raw.copy()

        if not np.all(np.isfinite(coeffs)):
            return x_raw.copy()

        if self.max_weight_norm is not None:
            if np.linalg.norm(coeffs, ord=1) > self.max_weight_norm:
                return x_raw.copy()

        X = np.column_stack(self._x_hist)          # shape (n, k)
        x_acc_vec = X @ coeffs

        x_out_vec = (1.0 - self.damping) * x_raw_vec + self.damping * x_acc_vec
        return x_out_vec.reshape(self._shape)


def make_next_gmres(
    m: int = 5,
    damping: float = 1.0,
    regularization: float = 1.0e-12,
    max_weight_norm: Optional[float] = 1.0e6,
) -> FixedPointGMRES:
    """Factory returning a stateful callable for fixed-point acceleration.

    This lets you use the exact style:

        next_gmres = make_next_gmres(m=5)
        while not converged:
            xn = transport_loop(xo)
            xn = next_gmres(xn, xo)
            xo = xn
    """
    return FixedPointGMRES(
        m=m,
        damping=damping,
        regularization=regularization,
        max_weight_norm=max_weight_norm,
    )


__all__ = ["FixedPointGMRES", "make_next_gmres"]


def transport_loop(xo: np.ndarray) -> np.ndarray:
    """Replace this with your actual transport/source iteration sweep."""
    return 0.8 * xo + 0.2


if __name__ == "__main__":
    np.random.seed(1)

    xo = 100.0 * np.random.random(100)
    next_gmres = make_next_gmres(m=5, damping=1.0)

    tol = 1.0e-10
    max_its = 200

    for it in range(max_its):
        xn = transport_loop(xo)
        xn = next_gmres(xn, xo)

        # The most natural convergence check here is the fixed-point residual.
        # If you can evaluate a more physical transport residual, use that instead.
        res = np.linalg.norm(transport_loop(xn) - xn)
        print(f"it={it:3d}, residual={res:.3e}")

        if res < tol:
            print(f"CONVERGED in {it + 1} iterations")
            xo = xn
            break

        xo = xn