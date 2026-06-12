import numpy as np

from rve_vam.homogenization import homogenize_mesh
from rve_vam.materials import IsotropicMaterial, PhaseMapping, isotropic_stiffness
from rve_vam.mesh import Mesh


def one_hex_mesh():
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
        ],
        dtype=float,
    )
    return Mesh(
        points=points,
        cells=np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64),
        cell_type="hexahedron",
        material_ids=np.array([1], dtype=np.int64),
        bounds=np.vstack([points.min(axis=0), points.max(axis=0)]),
        volume=1.0,
    )


def test_uniform_isotropic_homogenizes_to_input_stiffness():
    C = isotropic_stiffness(10.0, 0.25)
    material = IsotropicMaterial("mat", 10.0, 0.25, C)
    mapping = PhaseMapping(
        material_id_to_phase={1: "matrix"},
        phase_to_material_name={"matrix": "mat"},
        material_id_to_material={1: material},
    )
    result = homogenize_mesh(one_hex_mesh(), mapping, assembly_chunk_size=1)
    assert np.allclose(result.stiffness, C, rtol=1e-10, atol=1e-10)
    assert result.phase_volume_fractions == {"matrix": 1.0}
