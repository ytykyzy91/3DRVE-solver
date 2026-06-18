from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sp

from .assembly import (
    AssemblyResult,
    ReducedAssemblyResult,
    assemble_global_stiffness,
    assemble_reduced_system,
    element_dofs,
)
from .config import MacroStrainAnalysisOptions, SolverOptions
from .elements.hex8 import element_strain_stress_hex8
from .fields import recover_local_fields
from .macro_strain import analysis_with_internal_engineering_strain
from .materials import PhaseMapping, build_phase_mapping, load_json
from .mesh import Mesh, read_vtu_mesh
from .pbc import PeriodicDofMap, affine_displacement, affine_reference_point, build_periodic_dof_map_from_points
from .solver import solve_linear_system

logger = logging.getLogger(__name__)

MACRO_STRAINS = np.eye(6, dtype=float)
MACRO_STRAIN_LABELS = ["epsilon_xx", "epsilon_yy", "epsilon_zz", "gamma_xy", "gamma_yz", "gamma_xz"]


@dataclass(frozen=True)
class HomogenizationResult:
    stiffness: np.ndarray
    stiffness_unsymmetrized: np.ndarray
    volume: float
    phase_volume_fractions: dict[str, float]
    solver_residuals: list[float]
    material_mapping: dict[str, dict[str, str]]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class MacroStrainAnalysisResult:
    field_outputs: list[dict[str, object]]
    summary_path: Path
    volume: float
    phase_volume_fractions: dict[str, float]
    material_mapping: dict[str, dict[str, str]]
    diagnostics: dict[str, object]


def average_stress(mesh: Mesh, phase_mapping: PhaseMapping, displacement: np.ndarray) -> tuple[np.ndarray, float]:
    stress_integral = np.zeros(6, dtype=float)
    volume = 0.0
    for eidx, cell in enumerate(mesh.cells):
        material_id = int(mesh.material_ids[eidx])
        material = phase_mapping.material_id_to_material[material_id]
        dofs = element_dofs(cell)
        elem_stress_integral, elem_volume = element_strain_stress_hex8(
            mesh.points[cell], displacement[dofs], material.stiffness
        )
        stress_integral += elem_stress_integral
        volume += elem_volume
    return stress_integral / volume, float(volume)


def solve_macro_case(
    K: sp.csr_matrix,
    pbc_map: PeriodicDofMap,
    mesh: Mesh,
    phase_mapping: PhaseMapping,
    macro_strain: np.ndarray,
    solver: str,
    affine_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    T = pbc_map.transformation
    u_macro = affine_displacement(mesh.points, macro_strain, reference_point=affine_reference)
    K_reduced = T.T @ K @ T
    rhs_reduced = -(T.T @ (K @ u_macro))
    free = pbc_map.free_reduced_dofs

    q = np.zeros(K_reduced.shape[0], dtype=float)
    K_free = K_reduced[free][:, free].tocsr()
    rhs_free = np.asarray(rhs_reduced[free], dtype=float)
    solve = solve_linear_system(K_free, rhs_free, method=solver, context="homogenization/full")
    q[free] = solve.x
    displacement = u_macro + T @ q
    sigma_avg, volume = average_stress(mesh, phase_mapping, displacement)
    return sigma_avg, solve.residual_norm, volume


def _solve_single_case(
    K: sp.csr_matrix,
    rhs: np.ndarray,
    case_idx: int,
    label: str,
    T: sp.csr_matrix,
    free: np.ndarray,
    mesh: Mesh,
    phase_mapping: PhaseMapping,
    macro_strain: np.ndarray,
    solver: str,
    affine_reference: np.ndarray | None = None,
) -> tuple[int, np.ndarray, float, float]:
    """Solve a single homogenization case (for parallel execution)."""
    solve = solve_linear_system(
        K,
        rhs,
        method=solver,
        context=f"homogenization {case_idx + 1}/6 {label}",
    )
    q = np.zeros(T.shape[1], dtype=float)
    q[free] = solve.x
    displacement = affine_displacement(mesh.points, macro_strain, reference_point=affine_reference) + T @ q
    sigma_avg, volume = average_stress(mesh, phase_mapping, displacement)
    return case_idx, sigma_avg, solve.residual_norm, volume


def _solve_multiple_cases(
    K: sp.csr_matrix,
    rhs_multi: np.ndarray,
    T: sp.csr_matrix,
    free: np.ndarray,
    mesh: Mesh,
    phase_mapping: PhaseMapping,
    affine_reference: np.ndarray | None,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Solve multiple RHS at once using direct solver (multi-RHS optimization)."""
    columns: list[np.ndarray] = []
    residuals: list[float] = []
    volumes: list[float] = []

    # Solve all 6 RHS at once
    solve = solve_linear_system(K, rhs_multi, method="splu", context="homogenization multi-RHS")
    q_all = np.zeros((T.shape[1], 6), dtype=float)
    q_all[free, :] = solve.x

    # Post-process each case
    for case_idx, macro_strain in enumerate(MACRO_STRAINS):
        label = MACRO_STRAIN_LABELS[case_idx]
        displacement = affine_displacement(mesh.points, macro_strain, reference_point=affine_reference) + T @ q_all[:, case_idx]
        sigma_avg, volume = average_stress(mesh, phase_mapping, displacement)

        # Calculate per-case residual
        residual = np.linalg.norm(K @ q_all[free, case_idx] - rhs_multi[:, case_idx]) / max(1.0, np.linalg.norm(rhs_multi[:, case_idx]))
        columns.append(sigma_avg)
        residuals.append(float(residual))
        volumes.append(volume)
        logger.info("Completed homogenization reduced solve %d/6: %s, residual=%.6e", case_idx + 1, label, residual)

    return columns, residuals, volumes


def _solve_from_reduced_assembly(
    reduced: ReducedAssemblyResult,
    pbc_map: PeriodicDofMap,
    mesh: Mesh,
    phase_mapping: PhaseMapping,
    solver: str,
    affine_reference: np.ndarray | None = None,
    parallel: bool = True,
    max_workers: int = 6,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    columns: list[np.ndarray] = [None] * 6  # type: ignore[list-item]
    residuals: list[float] = [0.0] * 6
    volumes: list[float] = [0.0] * 6
    T = pbc_map.transformation
    free = pbc_map.free_reduced_dofs

    # For direct solvers: use multi-RHS optimization (fastest)
    if solver == "splu":
        logger.info("Using Sparse LU multi-RHS solve for all 6 cases")
        return _solve_multiple_cases(reduced.K, reduced.macro_rhs, T, free, mesh, phase_mapping, affine_reference)

    # For iterative solvers: use parallel or serial
    if parallel and max_workers > 1:
        logger.info("Starting PARALLEL homogenization solves with %d workers", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for case_idx, macro_strain in enumerate(MACRO_STRAINS):
                label = MACRO_STRAIN_LABELS[case_idx]
                future = executor.submit(
                    _solve_single_case,
                    reduced.K,
                    reduced.macro_rhs[:, case_idx],
                    case_idx,
                    label,
                    T,
                    free,
                    mesh,
                    phase_mapping,
                    macro_strain,
                    solver,
                    affine_reference,
                )
                futures[future] = case_idx

            for future in as_completed(futures):
                case_idx = futures[future]
                label = MACRO_STRAIN_LABELS[case_idx]
                try:
                    idx, sigma_avg, residual, volume = future.result()
                    columns[idx] = sigma_avg
                    residuals[idx] = residual
                    volumes[idx] = volume
                    logger.info("Completed homogenization reduced solve %d/6: %s, residual=%.6e", idx + 1, label, residual)
                except Exception as e:
                    logger.error("Case %s failed: %s", label, e)
                    raise
    else:
        logger.info("Starting SERIAL homogenization solves")
        for case_idx, macro_strain in enumerate(MACRO_STRAINS):
            label = MACRO_STRAIN_LABELS[case_idx]
            logger.info("Starting homogenization reduced solve %d/6: %s", case_idx + 1, label)
            _, sigma_avg, residual, volume = _solve_single_case(
                reduced.K,
                reduced.macro_rhs[:, case_idx],
                case_idx,
                label,
                T,
                free,
                mesh,
                phase_mapping,
                macro_strain,
                solver,
                affine_reference,
            )
            columns[case_idx] = sigma_avg
            residuals[case_idx] = residual
            volumes[case_idx] = volume
            logger.info("Completed homogenization reduced solve %d/6: %s, residual=%.6e", case_idx + 1, label, residual)

    return columns, residuals, volumes


def homogenize_mesh(
    mesh: Mesh,
    phase_mapping: PhaseMapping,
    *,
    pbc_tolerance: float = 1e-8,
    solver: str = "spsolve",
    symmetrize: bool = True,
    assembly_chunk_size: int = 20000,
    assembly_mode: str = "reduced",
    use_stiffness_cache: bool = True,
    stiffness_cache_size: int = 4096,
    stiffness_cache_decimals: int = 12,
    affine_origin: str = "zero",
    parallel: bool = True,
    parallel_workers: int = 6,
) -> HomogenizationResult:
    mode_desc = "PARALLEL" if parallel and parallel_workers > 1 else "SERIAL"
    logger.info("Starting homogenization: cells=%d, points=%d, assembly_mode=%s, solver=%s, mode=%s, workers=%d",
                mesh.n_cells, mesh.n_points, assembly_mode, solver, mode_desc, parallel_workers if parallel else 1)
    total_start = time.perf_counter()
    logger.info("Building periodic DOF map")
    pbc_start = time.perf_counter()
    pbc_map = build_periodic_dof_map_from_points(mesh.points, mesh.bounds, pbc_tolerance)
    logger.info(
        "Periodic DOF map built: reduced_dofs=%d, free_reduced_dofs=%d, elapsed=%.2fs",
        pbc_map.transformation.shape[1],
        len(pbc_map.free_reduced_dofs),
        time.perf_counter() - pbc_start,
    )
    affine_ref = affine_reference_point(mesh.points, mesh.bounds, affine_origin)
    logger.info("Using affine origin '%s': reference_point=%s", affine_origin, affine_ref.tolist())

    assembly_volume = 0.0
    phase_volume_fractions: dict[str, float] = {}
    extra_diagnostics: dict[str, object] = {"assembly_mode": assembly_mode}

    if assembly_mode == "reduced":
        reduced = assemble_reduced_system(
            mesh,
            phase_mapping,
            pbc_map,
            MACRO_STRAINS,
            chunk_size=assembly_chunk_size,
            use_stiffness_cache=use_stiffness_cache,
            stiffness_cache_size=stiffness_cache_size,
            stiffness_cache_decimals=stiffness_cache_decimals,
            affine_reference_point=affine_ref,
        )
        columns, residuals, case_volumes = _solve_from_reduced_assembly(
            reduced, pbc_map, mesh, phase_mapping, solver, affine_reference=affine_ref,
            parallel=parallel, max_workers=parallel_workers
        )
        assembly_volume = reduced.total_element_volume
        phase_volume_fractions = reduced.phase_volume_fractions
        extra_diagnostics.update(
            {
                "n_free_reduced_dofs": int(reduced.K.shape[0]),
                "reduced_nnz": int(reduced.K.nnz),
                "stiffness_cache_hits": int(reduced.cache_hits),
                "stiffness_cache_misses": int(reduced.cache_misses),
            }
        )
    elif assembly_mode == "full":
        assembly = assemble_global_stiffness(mesh, phase_mapping, chunk_size=assembly_chunk_size)
        columns = []
        residuals = []
        case_volumes = []
        for case_idx, macro_strain in enumerate(MACRO_STRAINS):
            label = MACRO_STRAIN_LABELS[case_idx]
            logger.info("Starting homogenization full solve %d/6: %s", case_idx + 1, label)
            sigma_avg, residual, volume = solve_macro_case(
                assembly.K, pbc_map, mesh, phase_mapping, macro_strain, solver, affine_reference=affine_ref
            )
            logger.info("Completed homogenization full solve %d/6: %s, residual=%.6e", case_idx + 1, label, residual)
            columns.append(sigma_avg)
            residuals.append(residual)
            case_volumes.append(volume)
        assembly_volume = assembly.total_element_volume
        phase_volume_fractions = assembly.phase_volume_fractions
        extra_diagnostics.update({"full_nnz": int(assembly.K.nnz)})
    else:
        raise ValueError("assembly_mode must be 'reduced' or 'full'.")

    logger.info("Homogenization solve loops completed; assembling output stiffness matrix")
    c_unsym = np.column_stack(columns)
    c_out = 0.5 * (c_unsym + c_unsym.T) if symmetrize else c_unsym.copy()
    norm = max(1.0, float(np.linalg.norm(c_unsym)))
    symmetry_error = float(np.linalg.norm(c_unsym - c_unsym.T) / norm)

    diagnostics = {
        "n_points": mesh.n_points,
        "n_cells": mesh.n_cells,
        "cell_type": mesh.cell_type,
        "material_counts": mesh.material_counts,
        "assembly_volume": assembly_volume,
        "case_volumes": case_volumes,
        "geometric_volume": mesh.volume,
        "n_full_dofs": int(3 * mesh.n_points),
        "n_reduced_dofs": int(pbc_map.transformation.shape[1]),
        "n_free_reduced_dofs": int(len(pbc_map.free_reduced_dofs)),
        "symmetry_relative_error": symmetry_error,
        "affine_origin": affine_origin,
        "affine_reference_point": affine_ref.tolist(),
        "elapsed_seconds": time.perf_counter() - total_start,
        **extra_diagnostics,
    }
    logger.info("Homogenization completed: symmetry_error=%.6e, elapsed=%.2fs", symmetry_error, diagnostics["elapsed_seconds"])
    return HomogenizationResult(
        stiffness=c_out,
        stiffness_unsymmetrized=c_unsym,
        volume=float(assembly_volume),
        phase_volume_fractions=phase_volume_fractions,
        solver_residuals=residuals,
        material_mapping=phase_mapping.as_metadata(),
        diagnostics=diagnostics,
    )


def run_homogenization(options: SolverOptions) -> HomogenizationResult:
    mesh = read_vtu_mesh(options.mesh_path)
    config = load_json(options.material_json_path)
    phase_mapping = build_phase_mapping(config, mesh.material_ids, options.material_id_map)
    return homogenize_mesh(
        mesh,
        phase_mapping,
        pbc_tolerance=options.pbc_tolerance,
        solver=options.solver,
        symmetrize=options.symmetrize,
        assembly_chunk_size=options.assembly_chunk_size,
        assembly_mode=options.assembly_mode,
        use_stiffness_cache=options.use_stiffness_cache,
        stiffness_cache_size=options.stiffness_cache_size,
        stiffness_cache_decimals=options.stiffness_cache_decimals,
        affine_origin=options.affine_origin,
        parallel=options.parallel,
        parallel_workers=options.parallel_workers,
    )


def run_macro_strain_analysis(options: SolverOptions) -> MacroStrainAnalysisResult:
    if options.macro_strain_analysis is None:
        raise ValueError("SolverOptions.macro_strain_analysis is required for macro-strain analysis.")

    from .postprocess import write_macro_strain_summary, write_result_vtu

    total_start = time.perf_counter()
    input_analysis = options.macro_strain_analysis
    analysis = analysis_with_internal_engineering_strain(input_analysis)
    logger.info(
        "Starting macro-strain analysis: cases=%d, load_steps=%d, solver=%s, input_strain_convention=%s",
        len(analysis.cases),
        analysis.load_steps,
        options.solver,
        input_analysis.strain_convention,
    )
    mesh = read_vtu_mesh(options.mesh_path)
    config = load_json(options.material_json_path)
    phase_mapping = build_phase_mapping(config, mesh.material_ids, options.material_id_map)
    logger.info("Building periodic DOF map for macro-strain analysis")
    pbc_start = time.perf_counter()
    pbc_map = build_periodic_dof_map_from_points(mesh.points, mesh.bounds, options.pbc_tolerance)
    logger.info(
        "Periodic DOF map built: reduced_dofs=%d, free_reduced_dofs=%d, elapsed=%.2fs",
        pbc_map.transformation.shape[1],
        len(pbc_map.free_reduced_dofs),
        time.perf_counter() - pbc_start,
    )
    affine_ref = affine_reference_point(mesh.points, mesh.bounds, options.affine_origin)
    logger.info("Using affine origin '%s': reference_point=%s", options.affine_origin, affine_ref.tolist())
    macro_strains = np.asarray([case.strain for case in analysis.cases], dtype=float)

    reduced = assemble_reduced_system(
        mesh,
        phase_mapping,
        pbc_map,
        macro_strains,
        chunk_size=options.assembly_chunk_size,
        use_stiffness_cache=options.use_stiffness_cache,
        stiffness_cache_size=options.stiffness_cache_size,
        stiffness_cache_decimals=options.stiffness_cache_decimals,
        affine_reference_point=affine_ref,
    )

    output_dir = analysis.field_output.output_dir or (options.output_dir / "fields")
    output_dir = Path(output_dir)
    summary_outputs: list[dict[str, object]] = []
    T = pbc_map.transformation
    free = pbc_map.free_reduced_dofs

    for case_idx, case in enumerate(analysis.cases):
        final_strain = np.asarray(case.strain, dtype=float)
        input_strain = np.asarray(input_analysis.cases[case_idx].strain, dtype=float)
        logger.info(
            "Starting macro-strain case %d/%d: %s, input_strain=%s (%s), internal_engineering_strain=%s",
            case_idx + 1,
            len(analysis.cases),
            case.name,
            input_strain.tolist(),
            input_analysis.strain_convention,
            final_strain.tolist(),
        )
        for step in range(1, analysis.load_steps + 1):
            step_start = time.perf_counter()
            fraction = step / analysis.load_steps
            rhs = fraction * reduced.macro_rhs[:, case_idx]
            logger.info("Solving case=%s step=%d/%d load_fraction=%.6g", case.name, step, analysis.load_steps, fraction)
            solve = solve_linear_system(
                reduced.K,
                rhs,
                method=options.solver,
                context=f"fields case={case.name} step={step}/{analysis.load_steps}",
            )
            q = np.zeros(T.shape[1], dtype=float)
            q[free] = solve.x
            displacement = affine_displacement(mesh.points, fraction * final_strain, reference_point=affine_ref) + T @ q

            should_write = analysis.field_output.enabled and (
                step == analysis.load_steps or step % analysis.field_output.output_every == 0
            )
            output_path: Path | None = None
            average_stress: np.ndarray | None = None
            if should_write:
                logger.info("Recovering local fields for case=%s step=%d", case.name, step)
                fields = recover_local_fields(mesh, phase_mapping, displacement)
                average_stress = fields.average_stress
                output_path = output_dir / f"{analysis.field_output.prefix}_{case.name}_step{step:04d}.vtu"
                logger.info("Writing VTU fields: %s", output_path)
                write_result_vtu(
                    mesh,
                    output_path,
                    fields,
                    extra_cell_data={
                        "LoadStep": np.full(mesh.n_cells, step, dtype=np.int32),
                        "LoadFraction": np.full(mesh.n_cells, fraction, dtype=float),
                    },
                )

            step_elapsed = time.perf_counter() - step_start
            logger.info(
                "Completed case=%s step=%d/%d: residual=%.6e, wrote_vtu=%s, elapsed=%.2fs",
                case.name,
                step,
                analysis.load_steps,
                solve.residual_norm,
                output_path is not None,
                step_elapsed,
            )
            summary_outputs.append(
                {
                    "case": case.name,
                    "step": step,
                    "load_fraction": fraction,
                    "input_macro_strain": (fraction * input_strain).tolist(),
                    "input_strain_convention": input_analysis.strain_convention,
                    "internal_engineering_macro_strain": (fraction * final_strain).tolist(),
                    "vtu_path": str(output_path) if output_path is not None else None,
                    "solver_residual": solve.residual_norm,
                    "average_stress": average_stress.tolist() if average_stress is not None else None,
                    "elapsed_seconds": step_elapsed,
                }
            )

    diagnostics = {
        "analysis_type": "macro_strain_local_fields",
        "strain_voigt_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
        "strain_convention": input_analysis.strain_convention,
        "internal_solver_strain_convention": "engineering_shear",
        "source": analysis.source,
        "load_steps": analysis.load_steps,
        "n_cases": len(analysis.cases),
        "n_points": mesh.n_points,
        "n_cells": mesh.n_cells,
        "cell_type": mesh.cell_type,
        "material_counts": mesh.material_counts,
        "assembly_mode": "reduced",
        "assembly_volume": reduced.total_element_volume,
        "geometric_volume": mesh.volume,
        "n_full_dofs": int(3 * mesh.n_points),
        "n_reduced_dofs": int(pbc_map.transformation.shape[1]),
        "n_free_reduced_dofs": int(reduced.K.shape[0]),
        "reduced_nnz": int(reduced.K.nnz),
        "affine_origin": options.affine_origin,
        "affine_reference_point": affine_ref.tolist(),
        "stiffness_cache_hits": int(reduced.cache_hits),
        "stiffness_cache_misses": int(reduced.cache_misses),
        "elapsed_seconds": time.perf_counter() - total_start,
    }
    summary = {
        "analysis_type": "macro_strain_local_fields",
        "field_outputs": summary_outputs,
        "volume": reduced.total_element_volume,
        "phase_volume_fractions": reduced.phase_volume_fractions,
        "material_mapping": phase_mapping.as_metadata(),
        "diagnostics": diagnostics,
    }
    summary_path = write_macro_strain_summary(summary, options.output_dir / "macro_strain_analysis.json")
    logger.info("Macro-strain analysis completed: summary=%s, elapsed=%.2fs", summary_path, diagnostics["elapsed_seconds"])
    return MacroStrainAnalysisResult(
        field_outputs=summary_outputs,
        summary_path=summary_path,
        volume=float(reduced.total_element_volume),
        phase_volume_fractions=reduced.phase_volume_fractions,
        material_mapping=phase_mapping.as_metadata(),
        diagnostics=diagnostics,
    )
