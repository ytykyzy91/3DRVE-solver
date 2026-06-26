"""
批量RVE均质化计算脚本

功能:
    遍历指定根目录下的所有子文件夹，对每个子文件夹执行RVE计算
    每个子文件夹需包含:
        - *.vtu: 网格文件
        - user_RVE_analysis.json: 材料配置文件

用法:
    # 计算指定目录下的所有子文件夹
    python batch_run.py --root D:\Data\RVE_cases

    # 仅计算前N个文件夹（用于测试）
    python batch_run.py --root D:\Data\RVE_cases --limit 3

    # 只输出stiffness.json（不计算场输出，更快）
    python batch_run.py --root D:\Data\RVE_cases --no-fields

    # 指定输出目录（默认在每个子文件夹下的outputs）
    python batch_run.py --root D:\Data\RVE_cases --output-dir D:\Data\RVE_results

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from rve_vam.homogenization import run_homogenization, run_macro_strain_analysis
from rve_vam.config import SolverOptions
from rve_vam.utils import setup_logging


def find_case_folders(root_dir: Path) -> list[Path]:
    """查找所有包含 vtu 和 user_RVE_analysis.json 的子文件夹"""
    cases = []
    for item in root_dir.iterdir():
        if not item.is_dir():
            continue
        # 检查是否包含必需文件
        vtu_files = list(item.glob("*.vtu"))
        json_file = item / "user_RVE_analysis.json"
        if vtu_files and json_file.exists():
            cases.append(item)
    return sorted(cases)


def run_single_case(
    case_dir: Path,
    output_dir: Path | None = None,
    write_fields: bool = True,
    solver: str = "cg",
    solver_rtol: float = 1e-6,
    cg_preconditioner: str = "ilu",
    parallel: bool = True,
    parallel_workers: int = 6,
) -> dict:
    """执行单个RVE算例计算

    Args:
        case_dir: 算例文件夹路径
        output_dir: 输出目录，默认在 case_dir / "outputs"
        write_fields: 是否输出场文件VTU，默认True
        solver: 求解器类型: cg, splu, spsolve
        solver_rtol: CG求解器相对残差
        cg_preconditioner: 预条件器: ilu, jacobi, none
        parallel: 是否启用并行
        parallel_workers: 并行worker数量

    Returns:
        计算结果统计字典
    """
    start_time = time.time()

    # 查找输入文件
    vtu_files = list(case_dir.glob("*.vtu"))
    if not vtu_files:
        raise FileNotFoundError(f"在 {case_dir} 中找不到 VTU 文件")
    mesh_path = vtu_files[0]

    material_path = case_dir / "user_RVE_analysis.json"
    if not material_path.exists():
        raise FileNotFoundError(f"找不到 {material_path}")

    # 输出目录
    if output_dir is None:
        out_dir = case_dir / "outputs"
    else:
        out_dir = output_dir / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    log_path = setup_logging(out_dir / "rve.log", "INFO")
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"开始计算: {case_dir.name}")
    logger.info(f"网格文件: {mesh_path.name}")
    logger.info(f"输出目录: {out_dir}")
    logger.info(f"求解器: {solver}")
    logger.info(f"写场输出: {write_fields}")
    logger.info("=" * 60)

    # 构建求解选项
    options = SolverOptions(
        mesh_path=mesh_path,
        material_json_path=material_path,
        output_dir=out_dir,
        material_mapping_mode="auto",
        material_id_map=None,
        pbc_tolerance=1e-8,
        solver=solver,
        symmetrize=True,
        assembly_chunk_size=20000,
        assembly_mode="reduced",
        use_stiffness_cache=True,
        stiffness_cache_size=4096,
        stiffness_cache_decimals=12,
        affine_origin="zero",
        macro_strain_analysis=None,  # None表示只做均质化
        parallel=parallel,
        parallel_workers=parallel_workers,
        solver_rtol=solver_rtol,
        cg_preconditioner=cg_preconditioner,
        log_file=log_path,
        log_level="INFO",
    )

    # 执行均质化
    logger.info("开始执行均质化计算...")
    result = run_homogenization(options)
    logger.info("均质化完成")
    logger.info(f"刚度矩阵已写入: {out_dir}")

    # 场输出
    if write_fields:
        logger.info("开始计算场输出...")
        from rve_vam.macro_strain import macro_strain_from_legacy_material_config, FieldOutputOptions

        field_options = SolverOptions(
            mesh_path=mesh_path,
            material_json_path=material_path,
            output_dir=out_dir,
            material_mapping_mode="auto",
            material_id_map=None,
            pbc_tolerance=1e-8,
            solver=solver,
            symmetrize=True,
            assembly_chunk_size=20000,
            assembly_mode="reduced",
            use_stiffness_cache=True,
            stiffness_cache_size=4096,
            stiffness_cache_decimals=12,
            affine_origin="zero",
            macro_strain_analysis=macro_strain_from_legacy_material_config(
                config=json.load(open(material_path)),
                load_steps=1,
                field_output=FieldOutputOptions(
                    enabled=True,
                    output_every=1,
                    output_dir=out_dir / "fields",
                    prefix="macro_strain",
                ),
                strain_convention="engineering_shear",
            ),
            parallel=parallel,
            parallel_workers=parallel_workers,
            solver_rtol=solver_rtol,
            cg_preconditioner=cg_preconditioner,
            log_file=log_path,
            log_level="INFO",
        )
        field_result = run_macro_strain_analysis(field_options)
        logger.info(f"场输出完成，文件数: {len(field_result.field_outputs)}")
    else:
        field_result = None

    elapsed = time.time() - start_time
    logger.info(f"{case_dir.name} 计算完成! 总耗时: {elapsed:.1f} 秒")

    return {
        "case_name": case_dir.name,
        "success": True,
        "elapsed_seconds": elapsed,
        "output_dir": str(out_dir),
        "stiffness_json": str(out_dir / "stiffness.json"),
        "stiffness_csv": str(out_dir / "stiffness.csv"),
        "diagnostics": result.diagnostics,
        "solver_residuals": result.solver_residuals,
    }


def main():
    parser = argparse.ArgumentParser(
        description="批量RVE均质化计算脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", required=True, type=Path, help="包含所有RVE算例的根目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="统一输出目录，默认每个算例在自己目录下的outputs")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前N个算例，用于测试")
    parser.add_argument("--no-fields", action="store_true", help="跳过场输出计算，只计算刚度矩阵")
    parser.add_argument("--solver", default="cg", choices=["cg", "splu", "spsolve"], help="求解器类型")
    parser.add_argument("--solver-rtol", default=1e-6, type=float, help="CG求解器相对残差")
    parser.add_argument("--cg-preconditioner", default="ilu", choices=["ilu", "jacobi", "none"], help="CG预条件器")
    parser.add_argument("--no-parallel", action="store_true", help="禁用并行求解")
    parser.add_argument("--parallel-workers", default=6, type=int, help="并行worker数量")

    args = parser.parse_args()

    root_dir: Path = args.root
    if not root_dir.exists():
        print(f"错误: 根目录不存在: {root_dir}")
        sys.exit(1)

    # 查找所有算例
    cases = find_case_folders(root_dir)
    if not cases:
        print(f"在 {root_dir} 中未找到任何有效算例文件夹")
        print("每个算例文件夹需要包含:")
        print("  - *.vtu (网格文件)")
        print("  - user_RVE_analysis.json (材料配置)")
        sys.exit(1)

    print(f"找到 {len(cases)} 个算例:")
    for i, case in enumerate(cases, 1):
        print(f"  {i:2d}. {case.name}")

    if args.limit and args.limit < len(cases):
        print(f"\n--limit={args.limit}, 仅处理前 {args.limit} 个算例")
        cases = cases[:args.limit]

    print(f"\n输出场文件: {'是' if not args.no_fields else '否'}")
    print(f"求解器: {args.solver}")
    print(f"并行: {'启用' if not args.no_parallel else '禁用'}")
    print(f"\n开始计算，共 {len(cases)} 个算例...\n")

    # 执行批量计算
    results = []
    failed = []
    total_start = time.time()

    for i, case_dir in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] 开始: {case_dir.name}")
        try:
            result = run_single_case(
                case_dir=case_dir,
                output_dir=args.output_dir,
                write_fields=not args.no_fields,
                solver=args.solver,
                solver_rtol=args.solver_rtol,
                cg_preconditioner=args.cg_preconditioner,
                parallel=not args.no_parallel,
                parallel_workers=args.parallel_workers,
            )
            results.append(result)
            print(f"  ✓ 完成，耗时: {result['elapsed_seconds']:.1f} 秒\n")
        except Exception as e:
            print(f"  ✗ 失败: {e}\n")
            failed.append({"case_name": case_dir.name, "error": str(e)})

    # 输出统计
    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"批量计算完成! 总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"成功: {len(results)} / {len(cases)}")
    if failed:
        print(f"失败: {len(failed)}")
        for f in failed:
            print(f"  - {f['case_name']}: {f['error']}")

    # 保存统计结果
    summary_path = root_dir / "batch_summary.json"
    summary = {
        "total_cases": len(cases),
        "success_count": len(results),
        "failed_count": len(failed),
        "total_elapsed_seconds": total_elapsed,
        "avg_elapsed_seconds": total_elapsed / len(results) if results else 0,
        "results": results,
        "failed": failed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n统计结果已写入: {summary_path}")


if __name__ == "__main__":
    main()
