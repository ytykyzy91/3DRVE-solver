from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import FieldOutputOptions, MacroStrainAnalysisOptions, MacroStrainCase
from .materials import VOIGT_ORDER
from .utils import parse_bool

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_case_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", str(name).strip()).strip("_")
    return cleaned or "case"


def _as_strain_tuple(values: object) -> tuple[float, float, float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError("Macro strain must contain exactly 6 values in [xx, yy, zz, xy, yz, xz] order.")
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Macro strain contains non-finite values: {values!r}.")
    return tuple(float(v) for v in arr.tolist())  # type: ignore[return-value]


def _validate_steps(load_steps: int) -> int:
    load_steps = int(load_steps)
    if load_steps < 1:
        raise ValueError("load_steps must be >= 1.")
    return load_steps


def _validate_output_every(output_every: int) -> int:
    output_every = int(output_every)
    if output_every < 1:
        raise ValueError("field output_every must be >= 1.")
    return output_every


def _validate_convention(convention: str) -> str:
    convention = str(convention).strip().lower()
    if convention not in {"engineering_shear", "tensor_shear"}:
        raise ValueError("strain_convention must be 'engineering_shear' or 'tensor_shear'.")
    return convention


def to_internal_engineering_strain(
    strain: tuple[float, float, float, float, float, float],
    convention: str,
) -> tuple[float, float, float, float, float, float]:
    convention = _validate_convention(convention)
    if convention == "engineering_shear":
        return strain
    e = list(strain)
    e[3] *= 0.5
    e[4] *= 0.5
    e[5] *= 0.5
    return tuple(float(v) for v in e)  # type: ignore[return-value]


def analysis_with_internal_engineering_strain(
    analysis: MacroStrainAnalysisOptions,
) -> MacroStrainAnalysisOptions:
    cases = tuple(
        MacroStrainCase(case.name, to_internal_engineering_strain(case.strain, analysis.strain_convention))
        for case in analysis.cases
    )
    return MacroStrainAnalysisOptions(
        cases=cases,
        load_steps=analysis.load_steps,
        field_output=analysis.field_output,
        source=analysis.source,
        strain_convention=analysis.strain_convention,
    )


def parse_macro_strain_config(path: Path | str) -> MacroStrainAnalysisOptions:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    order = data.get("strain_voigt_order", VOIGT_ORDER)
    if list(order) != VOIGT_ORDER:
        raise ValueError(f"Unsupported strain_voigt_order {order!r}; expected {VOIGT_ORDER}.")
    convention = _validate_convention(data.get("strain_convention", "engineering_shear"))

    output = data.get("output", {}) or {}
    field_output = FieldOutputOptions(
        enabled=parse_bool(output.get("write_vtu", True)),
        output_every=_validate_output_every(output.get("output_every", 1)),
        output_dir=Path(output["directory"]) if output.get("directory") else None,
        prefix=str(output.get("prefix", "macro_strain")),
    )

    cases = []
    for idx, item in enumerate(data.get("cases", [])):
        cases.append(
            MacroStrainCase(
                name=sanitize_case_name(item.get("name", f"case_{idx + 1}")),
                strain=_as_strain_tuple(item.get("macro_strain")),
            )
        )
    if not cases:
        raise ValueError(f"Macro strain config {path} contains no cases.")

    return MacroStrainAnalysisOptions(
        cases=tuple(cases),
        load_steps=_validate_steps(data.get("load_steps", 1)),
        field_output=field_output,
        source=str(path),
        strain_convention=convention,
    )


def parse_macro_strain_cli(
    text: str,
    *,
    load_steps: int,
    field_output: FieldOutputOptions,
    name: str = "cli_macro_strain",
    strain_convention: str = "engineering_shear",
) -> MacroStrainAnalysisOptions:
    values = [part.strip() for part in text.split(",")]
    return MacroStrainAnalysisOptions(
        cases=(MacroStrainCase(name=sanitize_case_name(name), strain=_as_strain_tuple(values)),),
        load_steps=_validate_steps(load_steps),
        field_output=field_output,
        source="cli",
        strain_convention=_validate_convention(strain_convention),
    )


def macro_strain_from_legacy_material_config(
    config: dict,
    *,
    load_steps: int,
    field_output: FieldOutputOptions,
    strain_convention: str = "engineering_shear",
) -> MacroStrainAnalysisOptions:
    macro = (
        config.get("Defs", {})
        .get("Composite", {})
        .get("LocalFieldsRecovery", {})
        .get("MacroStrain")
    )
    if not isinstance(macro, dict):
        raise ValueError("No Defs.Composite.LocalFieldsRecovery.MacroStrain entry found in material JSON.")

    strain = _as_strain_tuple(
        [
            macro.get("epsilon11", 0.0),
            macro.get("epsilon22", 0.0),
            macro.get("epsilon33", 0.0),
            macro.get("epsilon12", 0.0),
            macro.get("epsilon23", 0.0),
            macro.get("epsilon13", 0.0),
        ]
    )
    return MacroStrainAnalysisOptions(
        cases=(MacroStrainCase(name="fields", strain=strain),),
        load_steps=_validate_steps(load_steps),
        field_output=field_output,
        source="Defs.Composite.LocalFieldsRecovery.MacroStrain",
        strain_convention=_validate_convention(strain_convention),
    )


def with_cli_field_overrides(
    analysis: MacroStrainAnalysisOptions,
    *,
    write_fields: bool,
    output_every: int,
    output_dir: Path | None,
    prefix: str,
    load_steps: int | None = None,
) -> MacroStrainAnalysisOptions:
    base = analysis.field_output
    field_output = FieldOutputOptions(
        enabled=write_fields or base.enabled,
        output_every=_validate_output_every(output_every if output_every is not None else base.output_every),
        output_dir=output_dir if output_dir is not None else base.output_dir,
        prefix=prefix or base.prefix,
    )
    return MacroStrainAnalysisOptions(
        cases=analysis.cases,
        load_steps=_validate_steps(load_steps if load_steps is not None else analysis.load_steps),
        field_output=field_output,
        source=analysis.source,
        strain_convention=analysis.strain_convention,
    )
