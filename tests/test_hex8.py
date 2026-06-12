import numpy as np

from rve_vam.elements.hex8 import element_average_strain_stress_mises_hex8, stiffness_hex8
from rve_vam.materials import isotropic_stiffness


def unit_cube_coords():
    return np.array(
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


def test_hex8_stiffness_symmetry_volume_and_rigid_translation():
    C = isotropic_stiffness(1.0, 0.3)
    Ke, volume = stiffness_hex8(unit_cube_coords(), C)
    assert Ke.shape == (24, 24)
    assert np.isclose(volume, 1.0)
    assert np.allclose(Ke, Ke.T)

    for comp in range(3):
        mode = np.zeros(24)
        mode[comp::3] = 1.0
        assert np.linalg.norm(Ke @ mode) < 1e-10


def test_hex8_average_strain_stress_mises_for_affine_displacement():
    coords = unit_cube_coords()
    C = isotropic_stiffness(10.0, 0.25)
    strain = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    tensor = np.diag([0.01, 0.0, 0.0])
    displacement = (coords @ tensor.T).reshape(-1)
    avg_strain, avg_stress, mises, volume = element_average_strain_stress_mises_hex8(
        coords, displacement, C
    )
    assert np.isclose(volume, 1.0)
    assert np.allclose(avg_strain, strain)
    assert np.allclose(avg_stress, C @ strain)
    assert mises > 0.0
