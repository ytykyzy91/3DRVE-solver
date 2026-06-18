from __future__ import annotations

import csv
import json
from pathlib import Path

import meshio

from .fields import LocalFieldResult
from .homogenization import HomogenizationResult
from .materials import VOIGT_ORDER
from .mesh import Mesh


def engineering_constants_from_stiffness(stiffness) -> dict[str, float | list[list[float]]]:
    import numpy as np

    c = np.asarray(stiffness, dtype=float)
    if c.shape != (6, 6):
        raise ValueError(f"Expected 6x6 stiffness matrix, got {c.shape}.")
    s = np.linalg.inv(c)
    return {
        "compliance_matrix": s.tolist(),
        "E1": float(1.0 / s[0, 0]),
        "E2": float(1.0 / s[1, 1]),
        "E3": float(1.0 / s[2, 2]),
        "nu12": float(-s[1, 0] / s[0, 0]),
        "nu13": float(-s[2, 0] / s[0, 0]),
        "nu23": float(-s[2, 1] / s[1, 1]),
        "nu21": float(-s[0, 1] / s[1, 1]),
        "nu31": float(-s[0, 2] / s[2, 2]),
        "nu32": float(-s[1, 2] / s[2, 2]),
        "G12": float(1.0 / s[3, 3]),
        "G23": float(1.0 / s[4, 4]),
        "G13": float(1.0 / s[5, 5]),
    }


def write_stiffness_json(result: HomogenizationResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stiffness_voigt_order": VOIGT_ORDER,
        "strain_convention": "engineering_shear",
        "units": "same_as_input_material_E",
        "C": result.stiffness.tolist(),
        "C_unsymmetrized": result.stiffness_unsymmetrized.tolist(),
        "engineering_constants": engineering_constants_from_stiffness(result.stiffness),
        "volume": result.volume,
        "phase_volume_fractions": result.phase_volume_fractions,
        "material_mapping": result.material_mapping,
        "solver": {
            "relative_residuals": result.solver_residuals,
        },
        "diagnostics": result.diagnostics,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def write_stiffness_csv(result: HomogenizationResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    constants = engineering_constants_from_stiffness(result.stiffness)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Homogenized stiffness matrix C"])
        writer.writerow([""] + VOIGT_ORDER)
        for label, row in zip(VOIGT_ORDER, result.stiffness):
            writer.writerow([label] + [f"{float(value):.16g}" for value in row])
        writer.writerow([])
        writer.writerow(["Engineering constants from compliance S = inv(C)"])
        writer.writerow(["constant", "value"])
        for key in ["E1", "E2", "E3", "nu12", "nu13", "nu23", "nu21", "nu31", "nu32", "G12", "G23", "G13"]:
            writer.writerow([key, f"{float(constants[key]):.16g}"])


def write_outputs(result: HomogenizationResult, output_dir: Path | str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    json_path = output_dir / "stiffness.json"
    csv_path = output_dir / "stiffness.csv"
    write_stiffness_json(result, json_path)
    write_stiffness_csv(result, csv_path)
    return json_path, csv_path


def write_result_vtu(
    mesh: Mesh,
    path: Path | str,
    fields: LocalFieldResult,
    *,
    extra_cell_data: dict[str, object] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    point_data = {name: value for name, value in (mesh.point_data or {}).items()}
    point_data["Displacement"] = fields.displacement

    cell_data = {name: [value] for name, value in (mesh.cell_data or {}).items()}
    cell_data.update(
        {
            "Material": [mesh.material_ids],
            "Strain": [fields.strain],
            "Stress": [fields.stress],
            "MisesStress": [fields.mises],
        }
    )
    if extra_cell_data:
        for name, value in extra_cell_data.items():
            cell_data[name] = [value]
    result_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[(mesh.cell_type, mesh.cells)],
        point_data=point_data,
        cell_data=cell_data,
    )
    meshio.write(path, result_mesh)
    return path


def write_macro_strain_summary(summary: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return path
