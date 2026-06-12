from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np


@dataclass(frozen=True)
class Mesh:
    points: np.ndarray
    cells: np.ndarray
    cell_type: str
    material_ids: np.ndarray
    bounds: np.ndarray
    volume: float
    point_data: dict[str, np.ndarray] | None = None
    cell_data: dict[str, np.ndarray] | None = None

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def material_counts(self) -> dict[int, int]:
        counts = Counter(int(v) for v in self.material_ids.tolist())
        return dict(sorted(counts.items()))


def read_vtu_mesh(path: Path | str) -> Mesh:
    path = Path(path)
    raw = meshio.read(path)
    points = np.asarray(raw.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected 3D points in {path}, got shape {points.shape}.")

    cell_blocks = [(i, block) for i, block in enumerate(raw.cells) if block.type == "hexahedron"]
    unsupported = sorted({block.type for block in raw.cells if block.type != "hexahedron"})
    if unsupported and not cell_blocks:
        raise NotImplementedError(
            f"Unsupported VTU cell types {unsupported}. Current solver implements hexahedron cells."
        )
    if unsupported:
        raise NotImplementedError(
            f"Mixed cell types are not supported yet. Found hexahedron plus {unsupported}."
        )
    if not cell_blocks:
        raise ValueError(f"No hexahedron cells found in {path}.")

    cells = np.vstack([np.asarray(block.data, dtype=np.int64) for _, block in cell_blocks])
    material_blocks = raw.cell_data_dict.get("Material")
    if material_blocks is None or "hexahedron" not in material_blocks:
        raise ValueError(f"VTU file {path} does not contain CellData array 'Material' for hexahedron cells.")
    material_ids = np.asarray(material_blocks["hexahedron"], dtype=np.int64).reshape(-1)
    if material_ids.shape[0] != cells.shape[0]:
        raise ValueError(
            f"Material array length {material_ids.shape[0]} does not match cell count {cells.shape[0]}."
        )

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    bounds = np.vstack([mins, maxs])
    lengths = maxs - mins
    if np.any(lengths <= 0.0):
        raise ValueError(f"Invalid mesh bounds: min={mins}, max={maxs}.")
    volume = float(np.prod(lengths))

    point_data = {name: np.asarray(value) for name, value in raw.point_data.items()}
    cell_data = {
        name: np.asarray(by_type["hexahedron"])
        for name, by_type in raw.cell_data_dict.items()
        if "hexahedron" in by_type
    }

    return Mesh(
        points=points,
        cells=cells,
        cell_type="hexahedron",
        material_ids=material_ids,
        bounds=bounds,
        volume=volume,
        point_data=point_data,
        cell_data=cell_data,
    )
