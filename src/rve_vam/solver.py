from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearSolveResult:
    x: np.ndarray
    residual_norm: float
    info: int | None = None


def solve_linear_system(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    method: str = "spsolve",
    context: str = "",
) -> LinearSolveResult:
    a = matrix.tocsr()
    b = np.asarray(rhs, dtype=float)
    label = f" [{context}]" if context else ""
    if a.shape[0] == 0:
        logger.info("Skipping empty linear solve%s: shape=%s, nnz=%d", label, a.shape, a.nnz)
        return LinearSolveResult(x=np.zeros(0, dtype=float), residual_norm=0.0, info=0)

    logger.info("Starting linear solve%s: method=%s, shape=%s, nnz=%d, rhs_norm=%.6e", label, method, a.shape, a.nnz, float(np.linalg.norm(b)))
    start = time.perf_counter()
    if method == "spsolve":
        x = spla.spsolve(a, b)
        info = None
    elif method == "cg":
        x, info = spla.cg(a, b, rtol=1e-10, atol=0.0, maxiter=max(1000, 2 * a.shape[0]))
        if info != 0:
            logger.error("CG solver failed%s: info=%s, elapsed=%.2fs", label, info, time.perf_counter() - start)
            raise RuntimeError(f"CG solver did not converge, info={info}.")
    else:
        raise ValueError(f"Unknown linear solver {method!r}.")

    residual = a @ x - b
    denom = max(1.0, float(np.linalg.norm(b)))
    residual_norm = float(np.linalg.norm(residual) / denom)
    logger.info(
        "Linear solve completed%s: method=%s, residual=%.6e, info=%s, elapsed=%.2fs",
        label,
        method,
        residual_norm,
        info,
        time.perf_counter() - start,
    )
    return LinearSolveResult(x=np.asarray(x, dtype=float), residual_norm=residual_norm, info=info)
