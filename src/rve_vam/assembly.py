from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import logging
import time

import numpy as np
import scipy.sparse as sp

from .elements.hex8 import stiffness_hex8
from .materials import PhaseMapping
from .mesh import Mesh
from .pbc import PeriodicDofMap, affine_displacement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyResult:
    K: sp.csr_matrix
    element_volumes: np.ndarray
    total_element_volume: float
    phase_volumes: dict[str, float]
    phase_volume_fractions: dict[str, float]


@dataclass(frozen=True)
class ReducedAssemblyResult:
    K: sp.csr_matrix
    macro_rhs: np.ndarray
    element_volumes: np.ndarray
    total_element_volume: float
    phase_volumes: dict[str, float]
    phase_volume_fractions: dict[str, float]
    cache_hits: int
    cache_misses: int


def element_dofs(cell: np.ndarray) -> np.ndarray:
    dofs = np.empty(3 * len(cell), dtype=np.int64)
    dofs[0::3] = 3 * cell
    dofs[1::3] = 3 * cell + 1
    dofs[2::3] = 3 * cell + 2
    return dofs


def _phase_fractions(phase_volumes: dict[str, float], total: float) -> dict[str, float]:
    return {phase: float(volume / total) for phase, volume in phase_volumes.items()} if total else {}


def assemble_global_stiffness(
    mesh: Mesh,
    phase_mapping: PhaseMapping,
    chunk_size: int = 5000,
) -> AssemblyResult:
    n_dofs = 3 * mesh.n_points
    global_k = sp.csr_matrix((n_dofs, n_dofs), dtype=float)
    element_volumes = np.zeros(mesh.n_cells, dtype=float)
    phase_volumes: dict[str, float] = defaultdict(float)

    logger.info(
        "Starting full stiffness assembly: cells=%d, dofs=%d, chunk_size=%d",
        mesh.n_cells,
        n_dofs,
        chunk_size,
    )
    start_time = time.perf_counter()
    index_dtype = np.int32 if n_dofs <= np.iinfo(np.int32).max else np.int64
    for start in range(0, mesh.n_cells, chunk_size):
        chunk_time = time.perf_counter()
        stop = min(start + chunk_size, mesh.n_cells)
        n_entries = (stop - start) * 24 * 24
        rows = np.empty(n_entries, dtype=index_dtype)
        cols = np.empty(n_entries, dtype=index_dtype)
        data = np.empty(n_entries, dtype=float)
        cursor = 0

        for eidx in range(start, stop):
            cell = mesh.cells[eidx]
            material_id = int(mesh.material_ids[eidx])
            material = phase_mapping.material_id_to_material[material_id]
            phase = phase_mapping.material_id_to_phase[material_id]
            ke, volume = stiffness_hex8(mesh.points[cell], material.stiffness)
            dofs = element_dofs(cell)
            block_size = 24 * 24
            rows[cursor : cursor + block_size] = np.repeat(dofs, 24)
            cols[cursor : cursor + block_size] = np.tile(dofs, 24)
            data[cursor : cursor + block_size] = ke.ravel()
            cursor += block_size
            element_volumes[eidx] = volume
            phase_volumes[phase] += volume

        chunk = sp.coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
        global_k = global_k + chunk
        logger.info(
            "Full assembly chunk %d-%d/%d completed: chunk_nnz=%d, global_nnz=%d, elapsed=%.2fs",
            start,
            stop,
            mesh.n_cells,
            chunk.nnz,
            global_k.nnz,
            time.perf_counter() - chunk_time,
        )

    total = float(element_volumes.sum())
    logger.info(
        "Full stiffness assembly completed: nnz=%d, volume=%.8g, elapsed=%.2fs",
        global_k.nnz,
        total,
        time.perf_counter() - start_time,
    )
    return AssemblyResult(
        K=global_k.tocsr(),
        element_volumes=element_volumes,
        total_element_volume=total,
        phase_volumes=dict(phase_volumes),
        phase_volume_fractions=_phase_fractions(phase_volumes, total),
    )


class _ElementStiffnessCache:
    def __init__(self, enabled: bool, max_size: int, decimals: int) -> None:
        self.enabled = enabled
        self.max_size = max_size
        self.decimals = decimals
        self._data: OrderedDict[tuple, tuple[np.ndarray, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key(self, coords: np.ndarray, material_id: int) -> tuple:
        relative = coords - coords[0]
        return (material_id, tuple(np.round(relative.ravel(), self.decimals)))

    def get(self, coords: np.ndarray, material_id: int, stiffness: np.ndarray) -> tuple[np.ndarray, float]:
        if not self.enabled or self.max_size <= 0:
            self.misses += 1
            return stiffness_hex8(coords, stiffness)

        key = self.key(coords, material_id)
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
            self.hits += 1
            return value

        self.misses += 1
        value = stiffness_hex8(coords, stiffness)
        self._data[key] = value
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)
        return value


def assemble_reduced_system(
    mesh: Mesh,
    phase_mapping: PhaseMapping,
    pbc_map: PeriodicDofMap,
    macro_strains: np.ndarray,
    *,
    chunk_size: int = 5000,
    use_stiffness_cache: bool = True,
    stiffness_cache_size: int = 4096,
    stiffness_cache_decimals: int = 12,
    affine_reference_point: np.ndarray | None = None,
) -> ReducedAssemblyResult:
    """Assemble directly in the free periodic DOF space.

    This avoids materializing the full 3N-by-3N stiffness matrix and avoids the
    expensive sparse products T.T @ K @ T and T.T @ K @ u_macro. For large RVEs
    this is the preferred path.
    """
    free = pbc_map.free_reduced_dofs
    n_free = int(len(free))
    reduced_to_free = np.full(pbc_map.transformation.shape[1], -1, dtype=np.int64)
    reduced_to_free[free] = np.arange(n_free, dtype=np.int64)

    macro_strains = np.asarray(macro_strains, dtype=float)
    n_cases = int(macro_strains.shape[0])
    macro_displacements = [affine_displacement(mesh.points, macro_strains[i], reference_point=affine_reference_point)
                           for i in range(n_cases)]
    macro_rhs = np.zeros((n_free, n_cases), dtype=float)

    reduced_k = sp.csr_matrix((n_free, n_free), dtype=float)
    element_volumes = np.zeros(mesh.n_cells, dtype=float)
    phase_volumes: dict[str, float] = defaultdict(float)
    cache = _ElementStiffnessCache(use_stiffness_cache, stiffness_cache_size, stiffness_cache_decimals)

    logger.info(
        "Starting reduced stiffness assembly: cells=%d, free_dofs=%d, cases=%d, chunk_size=%d, cache=%s(size=%d)",
        mesh.n_cells,
        n_free,
        n_cases,
        chunk_size,
        use_stiffness_cache,
        stiffness_cache_size,
    )
    start_time = time.perf_counter()
    index_dtype = np.int32 if n_free <= np.iinfo(np.int32).max else np.int64
    for start in range(0, mesh.n_cells, chunk_size):
        chunk_time = time.perf_counter()
        stop = min(start + chunk_size, mesh.n_cells)
        max_entries = (stop - start) * 24 * 24
        rows = np.empty(max_entries, dtype=index_dtype)
        cols = np.empty(max_entries, dtype=index_dtype)
        data = np.empty(max_entries, dtype=float)
        cursor = 0

        for eidx in range(start, stop):
            cell = mesh.cells[eidx]
            material_id = int(mesh.material_ids[eidx])
            material = phase_mapping.material_id_to_material[material_id]
            phase = phase_mapping.material_id_to_phase[material_id]
            coords = mesh.points[cell]
            ke, volume = cache.get(coords, material_id, material.stiffness)
            full_dofs = element_dofs(cell)
            reduced_dofs = pbc_map.full_to_reduced[full_dofs]
            free_dofs = reduced_to_free[reduced_dofs]
            active = free_dofs >= 0
            active_free_dofs = free_dofs[active]
            active_ke = ke[np.ix_(active, active)]
            block_size = active_ke.size
            if block_size:
                rows[cursor : cursor + block_size] = np.repeat(active_free_dofs, active_free_dofs.size)
                cols[cursor : cursor + block_size] = np.tile(active_free_dofs, active_free_dofs.size)
                data[cursor : cursor + block_size] = active_ke.ravel()
                cursor += block_size

            for case_idx, u_macro in enumerate(macro_displacements):
                elem_rhs = -(ke @ u_macro[full_dofs])
                np.add.at(macro_rhs[:, case_idx], active_free_dofs, elem_rhs[active])

            element_volumes[eidx] = volume
            phase_volumes[phase] += volume

        if cursor:
            chunk = sp.coo_matrix(
                (data[:cursor], (rows[:cursor], cols[:cursor])), shape=(n_free, n_free)
            ).tocsr()
            reduced_k = reduced_k + chunk
        logger.info(
            "Reduced assembly chunk %d-%d/%d completed: entries=%d, reduced_nnz=%d, cache_hits=%d, cache_misses=%d, elapsed=%.2fs",
            start,
            stop,
            mesh.n_cells,
            cursor,
            reduced_k.nnz,
            cache.hits,
            cache.misses,
            time.perf_counter() - chunk_time,
        )

    total = float(element_volumes.sum())
    logger.info(
        "Reduced stiffness assembly completed: free_dofs=%d, nnz=%d, volume=%.8g, cache_hits=%d, cache_misses=%d, elapsed=%.2fs",
        n_free,
        reduced_k.nnz,
        total,
        cache.hits,
        cache.misses,
        time.perf_counter() - start_time,
    )
    return ReducedAssemblyResult(
        K=reduced_k.tocsr(),
        macro_rhs=macro_rhs,
        element_volumes=element_volumes,
        total_element_volume=total,
        phase_volumes=dict(phase_volumes),
        phase_volume_fractions=_phase_fractions(phase_volumes, total),
        cache_hits=cache.hits,
        cache_misses=cache.misses,
    )
