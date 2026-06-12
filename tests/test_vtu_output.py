import numpy as np
import meshio

from rve_vam.fields import recover_local_fields
from rve_vam.materials import IsotropicMaterial, PhaseMapping, isotropic_stiffness
from rve_vam.mesh import Mesh
from rve_vam.postprocess import write_result_vtu


def test_write_result_vtu_round_trip(tmp_path):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    mesh = Mesh(
        points=points,
        cells=np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64),
        cell_type="hexahedron",
        material_ids=np.array([1], dtype=np.int64),
        bounds=np.vstack([points.min(axis=0), points.max(axis=0)]),
        volume=1.0,
        point_data={"InputPointScalar": np.arange(8, dtype=float)},
        cell_data={"Material": np.array([1], dtype=np.int64), "InputCellScalar": np.array([3.14])},
    )
    C = isotropic_stiffness(10.0, 0.25)
    material = IsotropicMaterial("mat", 10.0, 0.25, C)
    mapping = PhaseMapping({1: "matrix"}, {"matrix": "mat"}, {1: material})
    displacement = (points @ np.diag([0.01, 0.0, 0.0]).T).reshape(-1)
    fields = recover_local_fields(mesh, mapping, displacement)
    path = write_result_vtu(mesh, tmp_path / "fields.vtu", fields)
    read = meshio.read(path)
    assert read.point_data["Displacement"].shape == (8, 3)
    assert read.point_data["InputPointScalar"].shape == (8,)
    assert read.cell_data_dict["InputCellScalar"]["hexahedron"].shape == (1,)
    assert read.cell_data_dict["Strain"]["hexahedron"].shape == (1, 6)
    assert read.cell_data_dict["Stress"]["hexahedron"].shape == (1, 6)
    assert read.cell_data_dict["MisesStress"]["hexahedron"].shape == (1,)
