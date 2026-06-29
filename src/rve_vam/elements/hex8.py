from __future__ import annotations

import numpy as np

# 尝试导入numba用于JIT加速
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

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
GAUSS_POINTS_ARRAY = np.array(GAUSS_POINTS, dtype=np.float64)


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


# ============================================================================
# Numba JIT 加速版本 (加速约10-20倍)
# ============================================================================
if HAS_NUMBA:
    @jit(nopython=True, fastmath=True)
    def _shape_derivatives_numba(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Numba编译的形函数导数计算"""
        dN = np.empty((8, 3), dtype=np.float64)
        # 节点符号 (硬编码以避免全局变量引用)
        sx = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float64)
        sy = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64)
        sz = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        for i in range(8):
            dN[i, 0] = 0.125 * sx[i] * (1.0 + sy[i] * eta) * (1.0 + sz[i] * zeta)
            dN[i, 1] = 0.125 * sy[i] * (1.0 + sx[i] * xi) * (1.0 + sz[i] * zeta)
            dN[i, 2] = 0.125 * sz[i] * (1.0 + sx[i] * xi) * (1.0 + sy[i] * eta)
        return dN


    @jit(nopython=True, fastmath=True)
    def _element_stress_integral_numba(
        coords: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        gauss_points: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Numba JIT编译的单元应力积分计算
        加速约10-20倍
        """
        stress_integral = np.zeros(6, dtype=np.float64)
        volume = 0.0

        for gp_idx in range(gauss_points.shape[0]):
            xi = gauss_points[gp_idx, 0]
            eta = gauss_points[gp_idx, 1]
            zeta = gauss_points[gp_idx, 2]
            weight = gauss_points[gp_idx, 3]

            # 形函数导数
            dN = _shape_derivatives_numba(xi, eta, zeta)

            # 雅可比矩阵
            jac = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    for k in range(8):
                        jac[i, j] += dN[k, i] * coords[k, j]

            det_j = np.linalg.det(jac)
            inv_jac = np.linalg.inv(jac)
            dN_dx = dN @ inv_jac

            dv = det_j * weight
            volume += dv

            # 计算应变
            strain = np.zeros(6, dtype=np.float64)
            for a in range(8):
                dx, dy, dz = dN_dx[a]
                ux = u[3 * a]
                uy = u[3 * a + 1]
                uz = u[3 * a + 2]
                strain[0] += dx * ux
                strain[1] += dy * uy
                strain[2] += dz * uz
                strain[3] += dy * ux + dx * uy
                strain[4] += dz * uy + dy * uz
                strain[5] += dz * ux + dx * uz

            # 应力积分
            for i in range(6):
                for j in range(6):
                    stress_integral[i] += c[i, j] * strain[j] * dv

        return stress_integral, volume


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
    """
    计算单元应力积分 (自动选择最快可用方法)

    Returns:
        (stress_integral, volume)
        stress_integral = average_stress * volume
    """
    if HAS_NUMBA:
        # 使用Numba JIT加速版本 (快10-20倍)
        u = np.asarray(displacement, dtype=np.float64).reshape(24)
        c = np.asarray(stiffness, dtype=np.float64)
        return _element_stress_integral_numba(coords, u, c, GAUSS_POINTS_ARRAY)
    else:
        # 回退到纯Python版本（但跳过不需要的Mises应力计算）
        u = np.asarray(displacement, dtype=float).reshape(24)
        c = np.asarray(stiffness, dtype=float)
        stress_integral = np.zeros(6, dtype=float)
        volume = 0.0
        for xi, eta, zeta, weight in GAUSS_POINTS:
            b, det_j = b_matrix_hex8(coords, xi, eta, zeta)
            dv = det_j * weight
            stress_integral += c @ (b @ u) * dv
            volume += dv
        return stress_integral, float(volume)
