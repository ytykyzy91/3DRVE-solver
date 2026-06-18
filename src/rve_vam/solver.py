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
    rtol: float = 1e-8,
) -> LinearSolveResult:
    a = matrix.tocsr()
    b = np.asarray(rhs, dtype=float)
    label = f" [{context}]" if context else ""
    if a.shape[0] == 0:
        logger.info("Skipping empty linear solve%s: shape=%s, nnz=%d", label, a.shape, a.nnz)
        return LinearSolveResult(x=np.zeros(0, dtype=float), residual_norm=0.0, info=0)

    # Handle multi-RHS: shape (n, nrhs)
    is_multi_rhs = b.ndim == 2 and b.shape[1] > 1
    if is_multi_rhs:
        b_norm = float(np.linalg.norm(b, axis=0).max())
    else:
        b_norm = float(np.linalg.norm(b))

    logger.info("Starting linear solve%s: method=%s, shape=%s, nnz=%d, rhs_norm=%.6e, multi_rhs=%s",
                label, method, a.shape, a.nnz, b_norm, is_multi_rhs)
    start = time.perf_counter()

    if method == "spsolve":
        x = spla.spsolve(a, b)
        info = None
    elif method == "splu":
        # Sparse LU factorization + solve for single or multiple RHS
        lu_start = time.perf_counter()
        logger.info("Factorizing Sparse LU...")
        lu = spla.splu(a.tocsc())
        lu_time = time.perf_counter() - lu_start
        logger.info("Sparse LU factorization completed in %.2fs", lu_time)
        solve_start = time.perf_counter()
        x = lu.solve(b)
        solve_time = time.perf_counter() - solve_start
        logger.info("Backward/forward substitution completed in %.2fs", solve_time)
        info = None
    elif method == "cg":
        # Use relaxed tolerance for homogenization
        x, info = spla.cg(a, b, rtol=rtol, atol=0.0, maxiter=10000)
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
