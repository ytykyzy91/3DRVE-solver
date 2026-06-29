@echo off
setlocal EnableExtensions

REM ======================================================================
REM RVE-VAM 单次计算脚本 (使用 conda py39 环境)
REM ======================================================================
REM 使用方法:
REM   run_rve_py39.bat help        : 查看完整帮助
REM   run_rve_py39.bat summary     : 只解析输入，打印网格/材料摘要
REM   run_rve_py39.bat test        : 运行小型内置验证
REM   run_rve_py39.bat run         : 运行示例 RVE 均质化（默认配置）
REM   run_rve_py39.bat large       : 运行 100万 级网格推荐配置
REM   run_rve_py39.bat fields      : 运行宏观应变场输出（VTU）
REM   run_rve_py39.bat fields-large: 大网格场输出配置
REM   run_rve_py39.bat fields-homogenization : 场输出 + 均质化刚度矩阵
REM ======================================================================

cd /d "%~dp0"

REM 激活 conda py39 环境
echo [INFO] Activating conda environment: py39
call "%USERPROFILE%\AppData\Local\miniconda3\condabin\conda.bat" activate py39
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment: py39
    echo Please ensure miniconda3 is installed and py39 environment exists.
    pause
    exit /b 1
)

REM 验证 Python 版本
python --version
if errorlevel 1 (
    echo [ERROR] Python not found in py39 environment
    pause
    exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set PYTHONUNBUFFERED=1

REM 配置：网格、材料、输出路径
set "MESH=%CD%\example\XiewenTtisu_fixed.vtu"
set "MATERIALS=%CD%\example\user_RVE_analysis.json"
set "OUTPUT=%CD%\outputs\XiewenTtisu_fixed"

REM 如果没有参数，显示帮助
if "%~1"=="" goto :help
if /I "%~1"=="help" goto :help
if /I "%~1"=="summary" goto :summary
if /I "%~1"=="test" goto :test
if /I "%~1"=="run" goto :run
if /I "%~1"=="large" goto :large
if /I "%~1"=="fields" goto :fields
if /I "%~1"=="fields-large" goto :fields_large
if /I "%~1"=="fields-homogenization" goto :fields_homogenization

echo Unknown command: %~1
goto :help

REM ======================================================================
REM summary: 只解析输入并打印摘要，不执行求解
REM ======================================================================
:summary
python -u -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --summary-only
goto :end

REM ======================================================================
REM test: 运行小型内置验证
REM ======================================================================
:test
python -u -m compileall -q src tests
if errorlevel 1 goto :end
python -u -c "import numpy as np; from pathlib import Path; from rve_vam.materials import isotropic_stiffness,load_json,build_phase_mapping,IsotropicMaterial,PhaseMapping; from rve_vam.elements.hex8 import stiffness_hex8,element_average_strain_stress_mises_hex8; from rve_vam.mesh import Mesh; from rve_vam.homogenization import homogenize_mesh; C=isotropic_stiffness(10.0,0.25); cfg=load_json(Path('example/user_RVE_analysis.json')); mp=build_phase_mapping(cfg,np.array([1,2])); pts=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float); Ke,v=stiffness_hex8(pts,isotropic_stiffness(1.0,0.3)); disp=(pts@np.diag([0.01,0,0]).T).reshape(-1); eps,sig,vm,vol=element_average_strain_stress_mises_hex8(pts,disp,C); assert np.allclose(eps,[0.01,0,0,0,0,0]); mesh=Mesh(pts,np.array([[0,1,2,3,4,5,6,7]],dtype=np.int64),'hexahedron',np.array([1]),np.vstack([pts.min(axis=0),pts.max(axis=0)]),1.0); mat=IsotropicMaterial('mat',10.0,0.25,C); res=homogenize_mesh(mesh,PhaseMapping({1:'matrix'},{1:'mat'},{1:mat}),assembly_chunk_size=1,solver='spsolve'); assert np.allclose(res.stiffness,C,rtol=1e-10,atol=1e-10); print('manual verification passed')"
goto :end

REM ======================================================================
REM run: 运行示例 RVE 均质化（默认配置，适合中小网格）
REM ======================================================================
:run
echo.
echo [INFO] Running RVE homogenization (default configuration for small/medium meshes)
echo   --solver:           cg (Conjugate Gradient)
echo   --assembly-mode:    reduced
echo   --assembly-chunk-size: 20000
echo   --stiffness-cache-size: 4096
echo.
python -u -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 20000 --stiffness-cache-size 4096
goto :end

REM ======================================================================
REM large: 100万级网格推荐配置
REM ======================================================================
:large
echo.
echo [INFO] Running RVE homogenization (optimized for 1,000,000-level meshes)
echo   --solver:           cg (Conjugate Gradient)
echo   --assembly-mode:    reduced
echo   --assembly-chunk-size: 50000
echo   --stiffness-cache-size: 20000
echo.
echo Note: If out-of-memory, reduce --assembly-chunk-size.
echo       If memory is abundant, increase --assembly-chunk-size for fewer chunks.
echo.
python -u -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 50000 --stiffness-cache-size 20000
goto :end

REM ======================================================================
REM fields: 运行宏观应变场输出（VTU）
REM ======================================================================
:fields
echo.
echo [INFO] Running macro-strain local fields recovery (VTU output)
echo   --solver:           cg
echo   --assembly-mode:    reduced
echo   --assembly-chunk-size: 20000
echo   --write-fields:     enabled
echo   --load-steps:       1
echo.
python -u -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 20000 --stiffness-cache-size 4096 --write-fields --load-steps 1 --field-output-dir "%OUTPUT%\fields"
goto :end

REM ======================================================================
REM fields-large: 大网格场输出配置
REM ======================================================================
:fields_large
echo.
echo [INFO] Running macro-strain local fields recovery (optimized for large meshes)
echo   --solver:           cg
echo   --assembly-mode:    reduced
echo   --assembly-chunk-size: 50000
echo   --write-fields:     enabled
echo   --load-steps:       1
echo.
python -u -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 50000 --stiffness-cache-size 20000 --write-fields --load-steps 1 --field-output-dir "%OUTPUT%\fields" --field-prefix macro_strain
goto :end

REM ======================================================================
REM fields-homogenization: 场输出 + 均质化刚度矩阵
REM ======================================================================
:fields_homogenization
echo.
echo [INFO] Running macro-strain local fields + homogenization stiffness matrix
echo   --solver:           cg
echo   --assembly-mode:    reduced
echo   --write-fields:     enabled
echo   --fields-with-homogenization: enabled (will output stiffness.json/csv)
echo.
python -u -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 20000 --stiffness-cache-size 4096 --write-fields --fields-with-homogenization --load-steps 1 --field-output-dir "%OUTPUT%\fields" --field-prefix macro_strain
goto :end

REM ======================================================================
REM help: 完整帮助文档
REM ======================================================================
:help
echo.
echo ======================================================================
echo RVE-VAM: 复合材料 RVE 有限元均质化求解器
echo ======================================================================
echo.
echo 快捷命令 (run_rve_py39.bat ^<command^>):
echo.
echo   help          显示此帮助
echo   summary       只解析输入，打印网格/材料摘要（不求解）
echo   test          运行小型内置验证（刚度矩阵、均质化正确性）
echo   run           运行示例 RVE 均质化（中小网格推荐配置）
echo   large         运行 100万 级网格推荐配置
echo   fields        运行宏观应变场输出（VTU）
echo   fields-large  大网格场输出配置
echo   fields-homogenization  场输出 + 均质化刚度矩阵
echo.
echo ----------------------------------------------------------------------
echo 完整命令行参数说明
echo ----------------------------------------------------------------------
echo.
echo 必需参数:
echo   --mesh PATH                  VTU 网格文件路径
echo   --materials PATH             材料配置 JSON 文件路径
echo   --output PATH                输出目录
echo.
echo 材料映射:
echo   --material-id-map MAP        显式指定材料ID映射，例如: 1:matrix,2:reinforcement
echo   --material-mapping MODE      映射模式，当前仅支持 auto (默认: auto)
echo.
echo PBC 配置:
echo   --pbc-tol VALUE              周期性节点配对容差 (默认: 1e-8)
echo   --affine-origin MODE         仿射位移参考点: zero, min, center (默认: zero)
echo.
echo 求解器配置:
echo   --solver TYPE                求解器类型: spsolve, splu, cg (默认: cg)
echo   --solver-rtol VALUE          CG 求解器相对残差阈值 (默认: 1e-5)
echo   --no-symmetrize              不对最终刚度矩阵对称化
echo.
echo 装配配置:
echo   --assembly-mode MODE         装配模式: reduced, full (默认: reduced)
echo   --assembly-chunk-size N      每次装配的单元数量 (默认: 20000)
echo                                内存不足时降低，内存充足时提高
echo   --stiffness-cache-size N     单元刚度缓存大小 (默认: 4096)
echo   --stiffness-cache-decimals N 缓存坐标的四舍五入小数位数 (默认: 12)
echo   --no-stiffness-cache         禁用单元刚度缓存
echo.
echo 并行配置:
echo   --no-parallel                禁用 6 个均质化方向的并行求解
echo   --parallel-workers N         并行 worker 数量，大网格推荐 2-3 (默认: 6)
echo.
echo 场输出配置（宏观应变分析）:
echo   --macro-strain-config PATH   宏观应变配置 JSON 文件
echo   --macro-strain VALUE         命令行直接输入宏观应变，例如: 0.01,0,0,0,0,0
echo   --macro-strain-convention MODE 剪切应变约定: engineering_shear, tensor_shear
echo   --load-steps N               载荷步数量 (默认: 1)
echo   --write-fields               启用 VTU 场输出
echo   --fields-with-homogenization 场输出后额外运行 6 方向均质化（输出 stiffness.json）
echo   --field-output-every N       每 N 步输出一次场 (默认: 1)
echo   --field-output-dir PATH      VTU 输出目录 (默认: OUTPUT/fields)
echo   --field-prefix STRING        VTU 文件名前缀 (默认: macro_strain)
echo.
echo 其他:
echo   --summary-only               只打印摘要，不执行求解
echo   --log-file PATH              日志文件路径 (默认: OUTPUT/rve_vam.log)
echo   --log-level LEVEL            日志级别: DEBUG, INFO, WARNING, ERROR (默认: INFO)
echo.
echo ----------------------------------------------------------------------
echo 常用配置示例
echo ----------------------------------------------------------------------
echo.
echo 1. 基本均质化（中小网格）:
echo    python -m rve_vam.cli --mesh example\case.vtu --materials example\config.json --output outputs\case
echo.
echo 2. 大网格优化配置:
echo    python -m rve_vam.cli --mesh example\case.vtu --materials example\config.json --output outputs\case --solver cg --assembly-chunk-size 50000 --stiffness-cache-size 20000
echo.
echo 3. 快速计算（5%%残差 + 2个并行worker）:
echo    python -m rve_vam.cli --mesh example\case.vtu --materials example\config.json --output outputs\case --solver cg --solver-rtol 5e-2 --parallel-workers 2
echo.
echo 4. 宏观应变场输出:
echo    python -m rve_vam.cli --mesh example\case.vtu --materials example\config.json --output outputs\case --write-fields --load-steps 5
echo.
echo 5. 场输出 + 刚度矩阵:
echo    python -m rve_vam.cli --mesh example\case.vtu --materials example\config.json --output outputs\case --write-fields --fields-with-homogenization
echo.
echo ----------------------------------------------------------------------
echo 输出文件
echo ----------------------------------------------------------------------
echo.
echo 均质化模式输出:
echo   outputs/case/stiffness.json      完整结果（刚度矩阵、诊断信息等）
echo   outputs/case/stiffness.csv       6x6 刚度矩阵（CSV 格式）
echo   outputs/case/rve_vam.log         详细日志
echo.
echo 场输出模式输出:
echo   outputs/case/fields/macro_strain_*_step*.vtu  VTU 场文件
echo   outputs/case/macro_strain_analysis.json       场分析摘要
echo.
echo ----------------------------------------------------------------------
echo 重要提示
echo ----------------------------------------------------------------------
echo.
echo - 必须使用 Python 3.9（本脚本自动激活 py39 环境）
echo - Numba 加速需要 py39 环境，可将应力平均计算加速约 10-20 倍
echo - 大规模网格推荐使用 --parallel-workers 2 而不是 6，避免内存带宽瓶颈
echo - splu 求解器（直接法）最快，但对百万级网格可能内存不足
echo - cg 求解器（迭代法）内存占用低，配合 --solver-rtol 5e-2 可快速出结果
echo.
pause

:end
if errorlevel 1 (
    echo.
    echo Command failed with exit code %ERRORLEVEL%.
) else (
    echo.
    echo Done.
)
endlocal
pause
