import numpy as np

from rve_vam.pbc import affine_displacement, affine_reference_point, build_periodic_dof_map_from_points, macro_strain_vector_to_tensor


def cube_points():
    return np.array([[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)])


def test_pbc_merges_unit_cube_corners():
    points = cube_points()
    bounds = np.vstack([points.min(axis=0), points.max(axis=0)])
    pbc = build_periodic_dof_map_from_points(points, bounds, 1e-8)
    assert len(set(pbc.node_representatives.tolist())) == 1
    assert pbc.transformation.shape == (24, 3)
    assert pbc.pinned_reduced_dofs.tolist() == [0, 1, 2]
    assert pbc.free_reduced_dofs.size == 0


def test_macro_strain_tensor_uses_engineering_shear_halves():
    tensor = macro_strain_vector_to_tensor(np.array([1, 2, 3, 4, 5, 6]))
    expected = np.array([[1, 2, 3], [2, 2, 2.5], [3, 2.5, 3]], dtype=float)
    assert np.allclose(tensor, expected)


def test_macro_strain_tensor_can_use_tensor_shear_directly():
    tensor = macro_strain_vector_to_tensor(np.array([1, 2, 3, 4, 5, 6]), convention="tensor_shear")
    expected = np.array([[1, 4, 6], [4, 2, 5], [6, 5, 3]], dtype=float)
    assert np.allclose(tensor, expected)


def test_affine_origin_min_uses_relative_coordinates():
    points = np.array([[10.0, 20.0, 30.0], [12.0, 25.0, 37.0]])
    bounds = np.vstack([points.min(axis=0), points.max(axis=0)])
    ref = affine_reference_point(points, bounds, "min")
    disp = affine_displacement(points, np.array([0.01, 0, 0, 0, 0, 0]), reference_point=ref).reshape(-1, 3)
    assert np.allclose(disp[0], [0.0, 0.0, 0.0])
    assert np.allclose(disp[1], [0.02, 0.0, 0.0])
