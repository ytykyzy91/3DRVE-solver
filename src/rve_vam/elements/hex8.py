from __future__ import annotations

import numpy as np

# VTK/standard hex node signs for reference coordinates.
_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)
_G = 1.0 / np.sqrt(3.0)
GAUSS_POINTS = [(xi, eta, zeta, 1.0) for xi in (-_G, _G) for eta in (-_G, _G) for zeta in (-_G, _G)]


def shape_function_derivatives_hex8(xi: float, eta: float, zeta: float) -> np.ndarray:
    dN = np.empty((8, 3), dtype=float)
    sx = _SIGNS[:, 0]
    sy = _SIGNS[:, 1]
    sz = _SIGNS[:, 2]
    dN[:, 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
    dN[:, 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
    dN[:, 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
    return dN


def b_matrix_hex8(coords: np.ndarray, xi: float, eta: float, zeta: float) -> tuple[np.ndarray, float]:
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (8, 3):
        raise ValueError(f"Expected hex8 coords shape (8, 3), got {coords.shape}.")

    dN_dnatural = shape_function_derivatives_hex8(xi, eta, zeta)
    jac = dN_dnatural.T @ coords
    det_j = float(np.linalg.det(jac))
    if det_j <= 1e-14:
        raise ValueError(f"Invalid hex8 Jacobian determinant {det_j:.6e}.")
    dN_dx = dN_dnatural @ np.linalg.inv(jac)

    b = np.zeros((6, 24), dtype=float)
    for a in range(8):
        col = 3 * a
        dx, dy, dz = dN_dx[a]
        b[0, col] = dx
        b[1, col + 1] = dy
        b[2, col + 2] = dz
        b[3, col] = dy
        b[3, col + 1] = dx
        b[4, col + 1] = dz
        b[4, col + 2] = dy
        b[5, col] = dz
        b[5, col + 2] = dx
    return b, det_j


def stiffness_hex8(coords: np.ndarray, stiffness: np.ndarray) -> tuple[np.ndarray, float]:
    c = np.asarray(stiffness, dtype=float)
    if c.shape != (6, 6):
        raise ValueError(f"Expected stiffness shape (6, 6), got {c.shape}.")
    ke = np.zeros((24, 24), dtype=float)
    volume = 0.0
    for xi, eta, zeta, weight in GAUSS_POINTS:
        b, det_j = b_matrix_hex8(coords, xi, eta, zeta)
        dv = det_j * weight
        ke += b.T @ c @ b * dv
        volume += dv
    return ke, float(volume)


def mises_stress(stress: np.ndarray) -> float:
    sxx, syy, szz, sxy, syz, sxz = np.asarray(stress, dtype=float).reshape(6)
    return float(
        np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )


def element_average_strain_stress_mises_hex8(
    coords: np.ndarray,
    displacement: np.ndarray,
    stiffness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    u = np.asarray(displacement, dtype=float).reshape(24)
    c = np.asarray(stiffness, dtype=float)
    strain_integral = np.zeros(6, dtype=float)
    stress_integral = np.zeros(6, dtype=float)
    volume = 0.0
    for xi, eta, zeta, weight in GAUSS_POINTS:
        b, det_j = b_matrix_hex8(coords, xi, eta, zeta)
        dv = det_j * weight
        strain = b @ u
        stress = c @ strain
        strain_integral += strain * dv
        stress_integral += stress * dv
        volume += dv
    strain_avg = strain_integral / volume
    stress_avg = stress_integral / volume
    return strain_avg, stress_avg, mises_stress(stress_avg), float(volume)


def element_strain_stress_hex8(
    coords: np.ndarray,
    displacement: np.ndarray,
    stiffness: np.ndarray,
) -> tuple[np.ndarray, float]:
    _, stress_avg, _, volume = element_average_strain_stress_mises_hex8(coords, displacement, stiffness)
    return stress_avg * volume, volume
