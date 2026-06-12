# RVE-VAM：复合材料 RVE 有限元均质化求解器

RVE-VAM 是一个基于 Python 的有限元求解器项目，用于复合材料 RVE（Representative Volume Element，代表性体积单元）的线弹性均质化分析。当前版本面向变分渐近法/周期性单胞边值问题的工程实现：对 6 个独立宏观应变分量分别加载，求解周期性 fluctuation 位移场，并输出等效 `6x6` 刚度矩阵。

## 当前输入文件结构

本项目已针对仓库中的两个示例输入完成解析：

- 网格：`example/case0001_fix.vtu`
  - VTU `UnstructuredGrid`。
  - `226981` 个节点。
  - `216000` 个单元。
  - 单元全部为 VTK type `12`，即 8 节点线性六面体 `hexahedron`。
  - `CellData["Material"]` 实际取值为 `1` 和 `2`：
    - `1`: `150672` cells
    - `2`: `65328` cells
- 材料配置：`example/user_RVE_analysis.json`
  - 材料库路径：`Defs.Analysis.Materials`
  - RVE 路径：`Defs.Composite.Settings.rvePath`
  - 复合材料相定义：`Defs.Composite.Materials`
  - 当前自动映射：
    - `Material ID 1 -> matrix -> Epoxy`
    - `Material ID 2 -> reinforcement -> Glass`
  - `Interphase.enabled = "0"`，当前不参与求解。

> 注意：原始需求中提到 `Material[0] -> reinforcement`、`Material[1] -> matrix`、`Material[2] -> Interphase`，但示例 VTU 的实际 Material ID 是 `1` 和 `2`。代码默认支持 0-based 和 1-based 自动识别，也支持命令行显式指定映射。

## 功能特性

- 读取压缩二进制 VTU 网格（通过 `meshio`）。
- 当前支持 8 节点线性六面体单元。
- 当前支持各向同性线弹性材料。
- 保留四面体、正交各向异性和非线性材料扩展接口。
- 周期性边界条件（PBC）：`u(x) = E_macro x + w(x)`，对 fluctuation `w` 周期约束。
- 6 个宏观应变加载：
  - `εxx`
  - `εyy`
  - `εzz`
  - `γxy`
  - `γyz`
  - `γxz`
- 输出均质化 `6x6` 刚度矩阵：
  - `stiffness.json`
  - `stiffness.csv`
- 支持大规模稀疏装配优化，目标支持 1,000,000 级六面体网格。

## 数值约定

Voigt 顺序固定为：

```text
[xx, yy, zz, xy, yz, xz]
```

应变向量采用工程剪切应变：

```text
[εxx, εyy, εzz, γxy, γyz, γxz]
```

其中：

```text
γij = 2 εij
```

因此各向同性刚度矩阵中的剪切对角项为剪切模量 `G`。

## 目录结构

```text
src/rve_vam/
  cli.py              # 命令行入口
  config.py           # 求解选项
  mesh.py             # VTU 网格读取
  materials.py        # JSON 材料解析和 Material ID 映射
  assembly.py         # 稀疏矩阵装配，含大规模 reduced assembly
  pbc.py              # 周期性边界条件和 DOF 降阶
  solver.py           # scipy.sparse 线性求解器封装
  homogenization.py   # 6 工况均质化流程
  postprocess.py      # JSON/CSV 输出
  nonlinear.py        # 非线性接口占位
  elements/
    hex8.py           # 8 节点六面体单元
    tet4.py           # 四面体接口占位

tests/                # 单元测试和小型验证
run_rve.bat           # Windows 一键运行脚本
```

## 环境要求

推荐 Python 3.10+。

依赖：

```text
numpy
scipy
meshio
```

可选测试依赖：

```text
pytest
```

安装方式：

```bat
python -m pip install numpy scipy meshio pytest
```

或者安装当前项目：

```bat
python -m pip install -e .
```

## Windows 一键运行

项目根目录下提供了：

```bat
run_rve.bat
```

### 1. 查看帮助

```bat
run_rve.bat help
```

### 2. 只解析输入并打印摘要

```bat
run_rve.bat summary
```

该命令会输出网格规模、bounds、Material ID 分布和材料映射，不执行大规模求解。

### 3. 运行小型内置验证

```bat
run_rve.bat test
```

该命令会：

- 编译 `src` 和 `tests`。
- 验证各向同性刚度矩阵。
- 验证 Hex8 单元刚度。
- 验证单个均匀六面体 RVE 的均质化刚度等于输入材料刚度。

### 4. 运行示例 RVE 求解

```bat
run_rve.bat run
```

默认使用：

```text
--solver cg
--assembly-mode reduced
--assembly-chunk-size 20000
--stiffness-cache-size 4096
```

输出目录：

```text
outputs/case0001/
```

### 5. 运行 1,000,000 级网格推荐配置

```bat
run_rve.bat large
```

默认使用：

```text
--solver cg
--assembly-mode reduced
--assembly-chunk-size 50000
--stiffness-cache-size 20000
```

如果内存不足，请降低 `--assembly-chunk-size`；如果内存充足，可提高该值来减少 chunk 数量。

## 命令行直接运行

不使用 BAT 时，可直接运行：

```bat
set PYTHONPATH=%CD%\src
python -m rve_vam.cli ^
  --mesh example\case0001_fix.vtu ^
  --materials example\user_RVE_analysis.json ^
  --output outputs\case0001 ^
  --solver cg ^
  --assembly-mode reduced ^
  --assembly-chunk-size 20000 ^
  --stiffness-cache-size 4096
```

只查看摘要：

```bat
set PYTHONPATH=%CD%\src
python -m rve_vam.cli ^
  --mesh example\case0001_fix.vtu ^
  --materials example\user_RVE_analysis.json ^
  --output outputs\case0001 ^
  --summary-only
```

显式指定材料 ID 映射：

```bat
python -m rve_vam.cli ^
  --mesh example\case0001_fix.vtu ^
  --materials example\user_RVE_analysis.json ^
  --output outputs\case0001 ^
  --material-id-map 1:matrix,2:reinforcement
```

## 大规模计算优化说明

为支持 1,000,000 级六面体网格，当前版本加入了以下优化。

### 1. Reduced assembly：默认推荐

传统方式会先装配完整全局刚度矩阵 `K_full`，再构造周期性约束矩阵 `T` 并计算：

```text
K_reduced = Tᵀ K_full T
rhs = -Tᵀ K_full u_macro
```

对于百万级网格，完整 `K_full` 的内存和 `Tᵀ K T` 乘法开销都很高。

当前默认使用：

```text
--assembly-mode reduced
```

它会在单元级直接把 `Ke` 散射到周期降阶后的自由 DOF 空间，避免显式形成完整 `K_full`，同时直接累计 6 个宏观应变工况的 RHS：

```text
K_reduced += A_eᵀ Ke A_e
rhs_case += -A_eᵀ Ke u_macro_e
```

优势：

- 避免完整 3N 全局刚度矩阵。
- 避免 `T.T @ K @ T` 大型稀疏矩阵乘法。
- RHS 在装配阶段一次性得到 6 个工况，减少重复矩阵向量乘法。

### 2. Chunked COO/CSR 装配

装配按元素块进行：

```text
--assembly-chunk-size 20000
```

每个 chunk 内预分配 NumPy 数组，再转 COO/CSR 累加，避免 Python list 存储上亿条条目。

百万级网格建议：

```text
--assembly-chunk-size 50000
```

如果内存不足，降低到：

```text
--assembly-chunk-size 10000
```

### 3. 单元刚度缓存

对于规则体素/结构化六面体网格，大量单元几何形状相同，仅材料不同。当前支持基于局部相对坐标和 Material ID 的 LRU 缓存：

```text
--stiffness-cache-size 4096
```

百万级规则网格建议：

```text
--stiffness-cache-size 20000
```

如果网格高度非结构化或缓存命中率低，可禁用：

```text
--no-stiffness-cache
```

求解结束后的 JSON diagnostics 会包含：

```text
stiffness_cache_hits
stiffness_cache_misses
```

### 4. 迭代求解器

大规模问题默认推荐：

```text
--solver cg
```

直接法 `spsolve` 对百万级 DOF 可能内存不足。`cg` 适用于周期约束并固定刚体位移后的对称正定系统。

如果问题规模较小，可使用：

```text
--solver spsolve
```

### 5. 内存规模提醒

百万级 Hex8 网格的完整单元条目规模约为：

```text
1,000,000 elements × 24 × 24 = 576,000,000 element entries
```

因此不要使用 full assembly 作为默认大规模路径。仅在调试或小网格对照时使用：

```text
--assembly-mode full
```

## 宏观应变应力/应变分析与 VTU 场输出

除 6 工况均质化外，当前版本还支持给定宏观应变的局部应力/应变场分析。该模式会按载荷步比例加载宏观应变，求解周期性 fluctuation 位移，并把结果写回 VTU。

### 独立宏观应变 JSON

示例：

```json
{
  "analysis_type": "macro_strain_local_fields",
  "strain_voigt_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
  "strain_convention": "engineering_shear",
  "load_steps": 3,
  "output": {
    "write_vtu": true,
    "output_every": 1,
    "directory": "outputs/case0001/fields",
    "prefix": "macro_strain"
  },
  "cases": [
    {
      "name": "uniaxial_x",
      "macro_strain": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  ]
}
```

运行：

```bat
set PYTHONPATH=%CD%\src
python -m rve_vam.cli ^
  --mesh example\case0001_fix.vtu ^
  --materials example\user_RVE_analysis.json ^
  --output outputs\case0001 ^
  --macro-strain-config macro_strain.json ^
  --write-fields
```

### 命令行直接输入宏观应变

```bat
python -m rve_vam.cli ^
  --mesh example\case0001_fix.vtu ^
  --materials example\user_RVE_analysis.json ^
  --output outputs\case0001 ^
  --macro-strain 0.01,0,0,0,0,0 ^
  --load-steps 5 ^
  --write-fields
```

### 剪切应变输入约定

宏观应变顺序固定为：

```text
[xx, yy, zz, xy, yz, xz]
```

默认 `strain_convention = engineering_shear`，后三项按工程剪切应变解释：

```text
[εxx, εyy, εzz, γxy, γyz, γxz]
```

此时构造应变张量时会使用：

```text
εxy = γxy / 2
εyz = γyz / 2
εxz = γxz / 2
```

如果你的输入文件或对比软件中的后三项本来就是张量剪应变：

```text
[εxx, εyy, εzz, εxy, εyz, εxz]
```

请使用：

```bat
--macro-strain-convention tensor_shear
```

例如：

```bat
python -m rve_vam.cli ^
  --mesh example\case0001_fix.vtu ^
  --materials example\user_RVE_analysis.json ^
  --output outputs\case0001 ^
  --macro-strain 0,0,0,0.01,0,0 ^
  --macro-strain-convention tensor_shear ^
  --write-fields
```

在独立 `macro_strain.json` 中也可以写：

```json
"strain_convention": "tensor_shear"
```

程序内部仍使用工程剪切 Voigt 约定计算；当选择 `tensor_shear` 时，会把输入后三项乘以 2 后进入内部求解。

### 使用材料 JSON 中的默认宏观应变

如果只指定：

```bat
run_rve.bat fields
```

程序会从以下路径读取默认宏观应变：

```text
Defs.Composite.LocalFieldsRecovery.MacroStrain
```

映射关系为：

```text
epsilon11 -> xx
epsilon22 -> yy
epsilon33 -> zz
epsilon12 -> xy
epsilon23 -> yz
epsilon13 -> xz
```

### 仿射位移参考点

宏观仿射位移按以下形式计算：

```text
u_macro = E_macro · (X - X_ref)
```

可用参数控制 `X_ref`：

```bat
--affine-origin zero
--affine-origin min
--affine-origin center
```

含义：

- `zero`：默认值，`X_ref = [0,0,0]`，即使用全局坐标。
- `min`：`X_ref = [xmin,ymin,zmin]`，最小角点位移从 0 开始，端点位移差就是 `strain * length`。
- `center`：`X_ref = (Xmin + Xmax)/2`，以 RVE 中心为参考，位移正负相对对称。

如果你要和其他软件的 displacement 绝对值对齐，建议重点尝试：

```bat
--affine-origin min
```

应变和应力主要由位移梯度决定，改变参考点通常只改变刚体平移部分；但 VTU 里输出的 displacement 绝对值会明显不同。

### 载荷步

`--load-steps N` 定义载荷步数量。当前线弹性版本中，第 `i` 步加载：

```text
macro_strain_step = i / N * macro_strain_final
```

这为后续塑性/损伤等非线性材料模型预留了增量求解流程。

### VTU 输出字段

每个输出步会生成一个 VTU，例如：

```text
outputs/case0001/fields/macro_strain_uniaxial_x_step0001.vtu
```

字段包括：

- `Displacement`：PointData，节点位移向量，shape `(n_points, 3)`。
- `Strain`：CellData，单元平均应变，shape `(n_cells, 6)`，Voigt 顺序 `[xx, yy, zz, xy, yz, xz]`。
- `Stress`：CellData，单元平均应力，shape `(n_cells, 6)`，Voigt 顺序 `[xx, yy, zz, xy, yz, xz]`。
- `MisesStress`：CellData，von Mises 等效应力标量。
- `Material`：CellData，原始 Material ID。
- `LoadStep`、`LoadFraction`：CellData，当前载荷步和载荷比例。

同时会写出轻量 summary：

```text
outputs/case0001/macro_strain_analysis.json
```

该 JSON 只记录输出文件路径、残差、平均应力和诊断信息，不保存大规模场数组。

### fields 模式是否额外输出均质化 stiffness.json/csv

默认情况下，`--write-fields` / `run_rve.bat fields` **只做宏观应变场分析并写 VTU**，不会再额外做 6 个方向的均质化求解，因此速度更快。

如果你需要在 fields 模式后同时输出均质化结果：

```bat
python -m rve_vam.cli ^
  --mesh example\case0001_fix0.vtu ^
  --materials example\user_RVE_analysis.json ^
  --output outputs\case0001 ^
  --write-fields ^
  --fields-with-homogenization
```

或使用：

```bat
run_rve.bat fields-homogenization
```

该选项会额外进行 6 个单位宏观应变方向的求解，并输出：

```text
outputs/case0001/stiffness.json
outputs/case0001/stiffness.csv
```

## 输出文件

完整求解后输出：

```text
outputs/case0001/stiffness.json
outputs/case0001/stiffness.csv
```

JSON 内容包括：

- `stiffness_voigt_order`
- `strain_convention`
- `C`
- `C_unsymmetrized`
- `volume`
- `phase_volume_fractions`
- `material_mapping`
- `solver.relative_residuals`
- `diagnostics`

CSV 内容为带行列标签的 `6x6` 刚度矩阵。

## 测试

如果安装了 `pytest`：

```bat
set PYTHONPATH=%CD%\src
python -m pytest -q
```

如果未安装 `pytest`，可运行：

```bat
run_rve.bat test
```

## 当前限制与后续扩展

当前已实现：

- Hex8 线性六面体。
- 各向同性线弹性。
- 周期性边界条件。
- 6 工况均质化刚度矩阵。
- 给定宏观应变的局部应力/应变场 VTU 输出。
- 载荷步比例加载流程，为非线性分析预留。
- 大规模 reduced sparse assembly。

当前保留接口但未实现：

- Tet4 四面体单元。
- 正交各向异性材料参与求解。
- 塑性/损伤等材料非线性。
- 预条件器/AMG 等高级迭代求解优化。
