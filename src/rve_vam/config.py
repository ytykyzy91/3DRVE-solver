from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MacroStrainCase:
    name: str
    strain: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class FieldOutputOptions:
    enabled: bool = False
    output_every: int = 1
    output_dir: Path | None = None
    prefix: str = "macro_strain"


@dataclass(frozen=True)
class MacroStrainAnalysisOptions:
    cases: tuple[MacroStrainCase, ...]
    load_steps: int = 1
    field_output: FieldOutputOptions = field(default_factory=FieldOutputOptions)
    source: str = "user"
    strain_convention: str = "engineering_shear"


@dataclass(frozen=True)
class SolverOptions:
    mesh_path: Path
    material_json_path: Path
    output_dir: Path
    material_mapping_mode: str = "auto"
    material_id_map: dict[int, str] | None = None
    pbc_tolerance: float = 1e-8
    solver: str = "cg"
    symmetrize: bool = True
    assembly_chunk_size: int = 20000
    assembly_mode: str = "reduced"
    use_stiffness_cache: bool = True
    stiffness_cache_size: int = 4096
    stiffness_cache_decimals: int = 12
    macro_strain_analysis: MacroStrainAnalysisOptions | None = None
    affine_origin: str = "zero"
    parallel: bool = True
    parallel_workers: int = 6
    solver_rtol: float = 1e-6
    cg_preconditioner: str = "ilu"
    log_file: Path | None = None
    log_level: str = "INFO"
