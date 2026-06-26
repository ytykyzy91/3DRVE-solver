from __future__ import annotations

import sys

# Print IMMEDIATELY before any heavy imports
print("RVE-VAM loading (imports may take 30-60 seconds on Python 3.14)...", flush=True)
print(f"Python version: {sys.version}", flush=True)

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from .config import FieldOutputOptions, SolverOptions
from .homogenization import run_homogenization, run_macro_strain_analysis
from .macro_strain import (
    macro_strain_from_legacy_material_config,
    parse_macro_strain_cli,
    parse_macro_strain_config,
    with_cli_field_overrides,
)
from .materials import build_phase_mapping, load_json, parse_material_id_map
from .mesh import read_vtu_mesh
from .postprocess import write_outputs
from .utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RVE finite-element homogenization solver")
    parser.add_argument("--mesh", required=True, type=Path, help="Input VTU mesh path")
    parser.add_argument("--materials", required=True, type=Path, help="Input material JSON path")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--material-mapping", default="auto", choices=["auto"], help="Material ID mapping mode")
    parser.add_argument("--material-id-map", help="Explicit map, e.g. 1:matrix,2:reinforcement")
    parser.add_argument("--pbc-tol", default=1e-8, type=float, help="Periodic node pairing tolerance")
    parser.add_argument("--affine-origin", default="zero", choices=["zero", "min", "center"], help="Reference point for affine macro displacement E*(X-Xref)")
    parser.add_argument("--solver", default="cg", choices=["spsolve", "splu", "cg"], help="Sparse linear solver (cg=Conjugate Gradient, recommended for large meshes)")
    parser.add_argument("--solver-rtol", default=1e-5, type=float, help="Relative tolerance for CG solver (default: 1e-5, use 1e-4 for even faster runs)")
    parser.add_argument("--assembly-mode", default="reduced", choices=["reduced", "full"], help="Assemble directly in periodic reduced space or assemble full K first")
    parser.add_argument("--assembly-chunk-size", default=20000, type=int, help="Elements per sparse assembly chunk")
    parser.add_argument("--stiffness-cache-size", default=4096, type=int, help="Max cached element stiffness patterns; 0 disables cache")
    parser.add_argument("--stiffness-cache-decimals", default=12, type=int, help="Coordinate rounding decimals for stiffness cache keys")
    parser.add_argument("--no-stiffness-cache", action="store_true", help="Disable element stiffness cache")
    parser.add_argument("--no-symmetrize", action="store_true", help="Do not symmetrize final stiffness")
    parser.add_argument("--summary-only", action="store_true", help="Only read mesh/materials and print mapping summary")
    parser.add_argument("--macro-strain-config", type=Path, help="Macro-strain local-field analysis JSON config")
    parser.add_argument("--macro-strain", help="Macro strain in xx,yy,zz,xy,yz,xz order, e.g. 0.01,0,0,0,0,0")
    parser.add_argument("--macro-strain-convention", default="engineering_shear", choices=["engineering_shear", "tensor_shear"], help="How to interpret macro-strain shear entries: engineering gamma_ij or tensor epsilon_ij")
    parser.add_argument("--load-steps", default=1, type=int, help="Number of proportional load steps for macro-strain analysis")
    parser.add_argument("--write-fields", action="store_true", help="Run macro-strain local-field analysis and write VTU fields")
    parser.add_argument("--fields-with-homogenization", action="store_true", help="After --write-fields, also run 6-case homogenization and write stiffness.json/csv")
    parser.add_argument("--field-output-every", default=1, type=int, help="Write every N load steps and always write final step")
    parser.add_argument("--field-output-dir", type=Path, help="Directory for field VTU files; default is OUTPUT/fields")
    parser.add_argument("--field-prefix", default="macro_strain", help="Prefix for VTU field files")
    parser.add_argument("--log-file", type=Path, help="Log file path; default is OUTPUT/rve_vam.log")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel solving of 6 homogenization cases")
    parser.add_argument("--parallel-workers", default=6, type=int, help="Number of parallel workers for homogenization (default: 6, one per case)")
    return parser


def _macro_analysis_requested(args: argparse.Namespace) -> bool:
    return bool(args.macro_strain_config or args.macro_strain or args.write_fields)


def _build_macro_analysis(args: argparse.Namespace, config: dict):
    field_output = FieldOutputOptions(
        enabled=args.write_fields,
        output_every=args.field_output_every,
        output_dir=args.field_output_dir,
        prefix=args.field_prefix,
    )
    if args.macro_strain_config:
        analysis = parse_macro_strain_config(args.macro_strain_config)
        return with_cli_field_overrides(
            analysis,
            write_fields=args.write_fields,
            output_every=args.field_output_every,
            output_dir=args.field_output_dir,
            prefix=args.field_prefix,
            load_steps=args.load_steps if args.load_steps != 1 else analysis.load_steps,
        )
    if args.macro_strain:
        return parse_macro_strain_cli(
            args.macro_strain,
            load_steps=args.load_steps,
            field_output=field_output,
            strain_convention=args.macro_strain_convention,
        )
    return macro_strain_from_legacy_material_config(
        config,
        load_steps=args.load_steps,
        field_output=field_output,
        strain_convention=args.macro_strain_convention,
    )


def main(argv: list[str] | None = None) -> int:
    print("RVE-VAM: Starting up...", flush=True)
    args = build_parser().parse_args(argv)
    print(f"RVE-VAM: Setting up logging to {args.output}", flush=True)
    log_path = setup_logging(args.log_file or (args.output / "rve_vam.log"), args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("RVE-VAM command started")
    logger.info("Arguments: %s", vars(args))
    explicit_map = parse_material_id_map(args.material_id_map)

    logger.info("Reading mesh: %s", args.mesh)
    mesh = read_vtu_mesh(args.mesh)
    logger.info("Reading material config: %s", args.materials)
    config = load_json(args.materials)
    mapping = build_phase_mapping(config, mesh.material_ids, explicit_map)

    print(f"Log file: {log_path}", flush=True)
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} {mesh.cell_type} cells", flush=True)
    print(f"Bounds: min={mesh.bounds[0].tolist()}, max={mesh.bounds[1].tolist()}", flush=True)
    print(f"Material counts: {mesh.material_counts}", flush=True)
    print("Material mapping:", flush=True)
    for mid, item in mapping.as_metadata().items():
        print(f"  {mid} -> {item['phase']} -> {item['material']}", flush=True)

    if args.summary_only:
        logger.info("Summary-only mode completed")
        logging.shutdown()
        return 0

    macro_analysis = _build_macro_analysis(args, config) if _macro_analysis_requested(args) else None
    if macro_analysis is not None and args.assembly_mode != "reduced":
        raise ValueError("Macro-strain local-field analysis currently requires --assembly-mode reduced.")

    options = SolverOptions(
        mesh_path=args.mesh,
        material_json_path=args.materials,
        output_dir=args.output,
        material_mapping_mode=args.material_mapping,
        material_id_map=explicit_map,
        pbc_tolerance=args.pbc_tol,
        solver=args.solver,
        symmetrize=not args.no_symmetrize,
        assembly_chunk_size=args.assembly_chunk_size,
        assembly_mode=args.assembly_mode,
        use_stiffness_cache=not args.no_stiffness_cache and args.stiffness_cache_size > 0,
        affine_origin=args.affine_origin,
        stiffness_cache_size=args.stiffness_cache_size,
        stiffness_cache_decimals=args.stiffness_cache_decimals,
        macro_strain_analysis=macro_analysis,
        parallel=not args.no_parallel,
        parallel_workers=args.parallel_workers,
        log_file=log_path,
        log_level=args.log_level,
    )
    if macro_analysis is not None:
        result = run_macro_strain_analysis(options)
        json_path = None
        csv_path = None
        if args.fields_with_homogenization:
            logger.info("--fields-with-homogenization enabled: running 6-case homogenization and writing stiffness JSON/CSV")
            stiffness_result = run_homogenization(replace(options, macro_strain_analysis=None))
            json_path, csv_path = write_outputs(stiffness_result, args.output)
        else:
            logger.info("Skipping homogenization after fields analysis. Use --fields-with-homogenization to write stiffness.json/csv.")
        written = [item for item in result.field_outputs if item.get("vtu_path")]
        print(f"Macro-strain summary: {result.summary_path}", flush=True)
        print(f"VTU field files written: {len(written)}", flush=True)
        if json_path and csv_path:
            print(f"Wrote {json_path}", flush=True)
            print(f"Wrote {csv_path}", flush=True)
        else:
            print("Skipped stiffness.json/csv. Add --fields-with-homogenization to generate them in fields mode.", flush=True)
        print(f"Diagnostics: {result.diagnostics}", flush=True)
        print(f"Log file: {log_path}", flush=True)
        logger.info("RVE-VAM command completed successfully")
        logging.shutdown()
        return 0

    result = run_homogenization(options)
    json_path, csv_path = write_outputs(result, args.output)
    print(f"Solver residuals: {result.solver_residuals}", flush=True)
    print(f"Symmetry relative error: {result.diagnostics['symmetry_relative_error']:.6e}", flush=True)
    print(f"Diagnostics: {result.diagnostics}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Log file: {log_path}", flush=True)
    logger.info("RVE-VAM command completed successfully")
    logging.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
