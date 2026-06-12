from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class PeriodicDofMap:
    node_representatives: np.ndarray
    full_to_reduced: np.ndarray
    transformation: sp.csr_matrix
    pinned_reduced_dofs: np.ndarray
    free_reduced_dofs: np.ndarray


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = int(self.parent[x])
        if parent != x:
            self.parent[x] = self.find(parent)
        return int(self.parent[x])

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _coord_key(values: np.ndarray, tol: float) -> tuple[int, ...]:
    return tuple(np.round(values / tol).astype(np.int64).tolist())


def build_periodic_node_representatives(points: np.ndarray, bounds: np.ndarray, tol: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mins, maxs = np.asarray(bounds, dtype=float)
    lengths = maxs - mins
    effective_tol = max(float(tol), 1e-10 * float(np.max(lengths)))
    uf = _UnionFind(points.shape[0])

    axes = [(0, (1, 2)), (1, (0, 2)), (2, (0, 1))]
    for axis, other_axes in axes:
        lo = np.flatnonzero(np.isclose(points[:, axis], mins[axis], atol=effective_tol, rtol=0.0))
        hi = np.flatnonzero(np.isclose(points[:, axis], maxs[axis], atol=effective_tol, rtol=0.0))
        lo_map = {_coord_key(points[idx, other_axes], effective_tol): int(idx) for idx in lo}
        for idx in hi:
            key = _coord_key(points[idx, other_axes], effective_tol)
            partner = lo_map.get(key)
            if partner is None:
                raise ValueError(
                    f"Failed to pair periodic node {idx} on axis {axis} with key {key} "
                    f"using tolerance {effective_tol}."
                )
            uf.union(partner, int(idx))

    reps = np.array([uf.find(i) for i in range(points.shape[0])], dtype=np.int64)
    canonical: dict[int, int] = {}
    for i, rep in enumerate(reps):
        current = canonical.get(int(rep))
        if current is None or i < current:
            canonical[int(rep)] = i
    return np.array([canonical[int(rep)] for rep in reps], dtype=np.int64)


def build_periodic_dof_map(node_representatives: np.ndarray, n_nodes: int) -> PeriodicDofMap:
    reps = np.asarray(node_representatives, dtype=np.int64)
    unique_reps = sorted(int(v) for v in np.unique(reps))
    rep_to_base = {rep: 3 * i for i, rep in enumerate(unique_reps)}
    full_to_reduced = np.empty(3 * n_nodes, dtype=np.int64)
    rows = np.arange(3 * n_nodes, dtype=np.int64)
    cols = np.empty(3 * n_nodes, dtype=np.int64)
    data = np.ones(3 * n_nodes, dtype=float)

    for node in range(n_nodes):
        base = rep_to_base[int(reps[node])]
        for comp in range(3):
            full = 3 * node + comp
            red = base + comp
            full_to_reduced[full] = red
            cols[full] = red

    n_reduced = 3 * len(unique_reps)
    transformation = sp.coo_matrix((data, (rows, cols)), shape=(3 * n_nodes, n_reduced)).tocsr()
    pinned = np.array([0, 1, 2], dtype=np.int64)
    free = np.setdiff1d(np.arange(n_reduced, dtype=np.int64), pinned, assume_unique=True)
    return PeriodicDofMap(
        node_representatives=reps,
        full_to_reduced=full_to_reduced,
        transformation=transformation,
        pinned_reduced_dofs=pinned,
        free_reduced_dofs=free,
    )


def build_periodic_dof_map_from_points(points: np.ndarray, bounds: np.ndarray, tol: float) -> PeriodicDofMap:
    reps = build_periodic_node_representatives(points, bounds, tol)
    return build_periodic_dof_map(reps, points.shape[0])


def macro_strain_vector_to_tensor(voigt: np.ndarray, convention: str = "engineering_shear") -> np.ndarray:
    e = np.asarray(voigt, dtype=float).reshape(6)
    if convention == "engineering_shear":
        shear_factor = 0.5
    elif convention == "tensor_shear":
        shear_factor = 1.0
    else:
        raise ValueError("convention must be 'engineering_shear' or 'tensor_shear'.")
    return np.array(
        [
            [e[0], shear_factor * e[3], shear_factor * e[5]],
            [shear_factor * e[3], e[1], shear_factor * e[4]],
            [shear_factor * e[5], shear_factor * e[4], e[2]],
        ],
        dtype=float,
    )


def affine_reference_point(points: np.ndarray, bounds: np.ndarray, origin: str = "zero") -> np.ndarray:
    origin = str(origin).strip().lower()
    if origin == "zero":
        return np.zeros(3, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if origin == "min":
        return bounds[0].copy()
    if origin == "center":
        return 0.5 * (bounds[0] + bounds[1])
    raise ValueError("affine origin must be 'zero', 'min', or 'center'.")


def affine_displacement(
    points: np.ndarray,
    macro_strain_voigt: np.ndarray,
    convention: str = "engineering_shear",
    reference_point: np.ndarray | None = None,
) -> np.ndarray:
    strain = macro_strain_vector_to_tensor(macro_strain_voigt, convention=convention)
    coords = np.asarray(points, dtype=float)
    if reference_point is not None:
        coords = coords - np.asarray(reference_point, dtype=float).reshape(1, 3)
    return (coords @ strain.T).reshape(-1)
