@echo off
REM ======================================================================
REM RVE-VAM 批量计算脚本
REM ======================================================================
REM 使用方法:
REM   batch_solve.bat              : 使用默认配置运行（串行，1个算例）
REM   batch_solve.bat help         : 查看帮助
REM   batch_solve.bat test         : 测试运行1个算例（默认模式）
REM   batch_solve.bat all          : 运行所有算例
REM   batch_solve.bat resume       : 续算模式（跳过已完成的算例）
REM   batch_solve.bat from51       : 从第51个算例开始
REM ======================================================================

cd /d D:\ClaudeCode\RVE_VAM
set PYTHONPATH=%CD%\src

REM 先激活 conda py39 环境（必须，否则 Numba 不生效）
echo [INFO] Activating conda py39 environment...
call conda activate py39
if errorlevel 1 (
    echo [ERROR] Failed to activate conda py39 environment!
    echo Please ensure miniconda/anaconda is installed and py39 environment exists.
    pause
    exit /b 1
)

REM 检查是否有参数
if "%~1"=="" goto :run_test
if /I "%~1"=="help" goto :help
if /I "%~1"=="test" goto :test
if /I "%~1"=="all" goto :run_all
if /I "%~1"=="resume" goto :resume
if /I "%~1"=="from51" goto :from51

echo 未知命令: %~1
goto :help

REM ======================================================================
REM 测试模式：只运行1个算例（串行）
REM ======================================================================
:run_test
:test
echo [INFO] 测试模式：只运行第1个算例（串行）
echo.
echo 配置:
echo   - 求解器: cg (共轭梯度)
echo   - 残差阈值: 5e-2 (5%)
echo   - 并行: 禁用（串行）
echo   - 只输出刚度矩阵 (--no-fields)
echo.
python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --solver-rtol 5e-2 --no-parallel --no-fields --limit 1
pause
goto :end

REM ======================================================================
REM 运行所有算例（2个并行worker）
REM ======================================================================
:run_all
echo [INFO] 运行所有算例（2个并行worker）
echo.
echo 配置:
echo   - 求解器: cg (共轭梯度)
echo   - 残差阈值: 5e-2 (5%)
echo   - 并行worker: 2
echo   - 只输出刚度矩阵 (--no-fields)
echo.
python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --solver-rtol 5e-2 --parallel-workers 2 --no-fields
pause
goto :end

REM ======================================================================
REM 续算模式
REM ======================================================================
:resume
echo [INFO] 续算模式：自动跳过已完成的算例
echo.
python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --solver-rtol 5e-2 --parallel-workers 2 --no-fields --resume
pause
goto :end

REM ======================================================================
REM 从第51个开始
REM ======================================================================
:from51
echo [INFO] 从第51个算例开始计算
echo.
python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --solver-rtol 5e-2 --parallel-workers 2 --no-fields --start-from 51
pause
goto :end

REM ======================================================================
REM 帮助文档
REM ======================================================================
:help
echo.
echo ======================================================================
echo RVE-VAM 批量计算脚本帮助
echo ======================================================================
echo.
echo 快捷命令:
echo   batch_solve.bat              测试模式，只运行第1个算例（串行）
echo   batch_solve.bat test         同上
echo   batch_solve.bat all          运行所有算例（2个并行worker）
echo   batch_solve.bat resume       续算模式：自动跳过已完成的算例
echo   batch_solve.bat from51       从第51个算例开始计算
echo   batch_solve.bat help         显示此帮助
echo.
echo ----------------------------------------------------------------------
echo 完整参数说明 (直接使用 python batch_run.py):
echo ----------------------------------------------------------------------
echo.
echo 必需参数:
echo   --root PATH                  包含所有算例的根目录
echo.
echo 运行控制:
echo   --limit N                    仅处理前N个算例，用于测试
echo   --start-from N               从第N个算例开始计算 (默认: 1)
echo   --resume                     续算模式：自动跳过已有 stiffness.json 的算例
echo.
echo 计算配置:
echo   --no-fields                  只计算刚度矩阵，不输出 VTU 场文件（更快）
echo   --solver TYPE                求解器类型: cg, splu, spsolve (默认: cg)
echo   --solver-rtol VALUE          CG 求解器相对残差阈值 (默认: 1e-5)
echo.
echo 并行配置:
echo   --no-parallel                禁用并行求解
echo   --parallel-workers N         并行 worker 数量，大网格推荐 2-3 (默认: 2)
echo.
echo 输出配置:
echo   --output-dir PATH            统一输出目录，默认每个算例在自己的 outputs/ 目录
echo.
echo ----------------------------------------------------------------------
echo 常用组合示例:
echo ----------------------------------------------------------------------
echo.
echo 1. 快速测试1个算例:
echo    python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --no-fields --limit 1
echo.
echo 2. 批量计算所有算例（快速模式，5%%残差）:
echo    python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --no-fields --solver-rtol 5e-2 --parallel-workers 2
echo.
echo 3. 智能续算（自动跳过已完成的）:
echo    python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --no-fields --resume
echo.
echo 4. 从第51个开始，算10个:
echo    python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --no-fields --start-from 51 --limit 10
echo.
echo 5. 从第51个开始 + 自动续算:
echo    python batch_run.py --root C:\3DRVE_dataset\Weave2D\woven_weave_2d_batch0001 --no-fields --start-from 51 --resume
echo.
echo ----------------------------------------------------------------------
echo 重要提示:
echo ----------------------------------------------------------------------
echo - 本脚本会自动激活 conda py39 环境，确保 Numba 加速生效
echo - 确认日志开头显示 "Python version: 3.9.x" 而不是 3.14.x
echo - 大规模计算建议: --solver-rtol 5e-2 --parallel-workers 2 --no-fields
echo - 已完成的算例检测标准: 输出目录存在 stiffness.json
echo - 6个并行worker可能导致内存带宽瓶颈，应力平均阶段反而变慢
echo.
pause

:end
