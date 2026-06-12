from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np

from .assembly import element_dofs
from .elements.hex8 import element_average_strain_stress_mises_hex8
from .materials import PhaseMapping
from .mesh import Mesh

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalFieldResult:
    displacement: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    mises: np.ndarray
    volume: float
    average_stress: np.ndarray


def recover_local_fields(
    mesh: Mesh,
    phase_mapping: PhaseMapping,
    displacement_flat: np.ndarray,
) -> LocalFieldResult:
    logger.info("Starting local field recovery: cells=%d, points=%d", mesh.n_cells, mesh.n_points)
    start_time = time.perf_counter()
    report_interval = max(1, mesh.n_cells // 20)
    displacement = np.asarray(displacement_flat, dtype=float).reshape(mesh.n_points, 3)
    flat = displacement.reshape(-1)
    strain = np.zeros((mesh.n_cells, 6), dtype=float)
    stress = np.zeros((mesh.n_cells, 6), dtype=float)
    mises = np.zeros(mesh.n_cells, dtype=float)
    stress_integral = np.zeros(6, dtype=float)
    total_volume = 0.0

    for eidx, cell in enumerate(mesh.cells):
        material_id = int(mesh.material_ids[eidx])
        material = phase_mapping.material_id_to_material[material_id]
        dofs = element_dofs(cell)
        elem_strain, elem_stress, elem_mises, elem_volume = element_average_strain_stress_mises_hex8(
            mesh.points[cell], flat[dofs], material.stiffness
        )
        strain[eidx] = elem_strain
        stress[eidx] = elem_stress
        mises[eidx] = elem_mises
        stress_integral += elem_stress * elem_volume
        total_volume += elem_volume
        if (eidx + 1) % report_interval == 0 or eidx + 1 == mesh.n_cells:
            logger.info(
                "Local field recovery progress: %d/%d cells (%.1f%%), elapsed=%.2fs",
                eidx + 1,
                mesh.n_cells,
                100.0 * (eidx + 1) / mesh.n_cells,
                time.perf_counter() - start_time,
            )

    average_stress = stress_integral / total_volume
    logger.info("Local field recovery completed: volume=%.8g, elapsed=%.2fs", total_volume, time.perf_counter() - start_time)
    return LocalFieldResult(
        displacement=displacement,
        strain=strain,
        stress=stress,
        mises=mises,
        volume=float(total_volume),
        average_stress=average_stress,
    )
