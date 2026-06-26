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
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


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
    solver_rtol: float = 1e-4,
    parallel: bool = True,
    parallel_workers: int = 6,
) -> dict:
    """执行单个RVE算例计算

    Args:
        case_dir: 算例文件夹路径
        output_dir: 输出目录，默认在 case_dir / "outputs"
        write_fields: 是否输出场文件VTU，默认True
        solver: 求解器类型: cg, splu, spsolve
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

    print(f"  网格文件: {mesh_path.name}")
    print(f"  输出目录: {out_dir}")

    # 构建命令行参数
    cmd = [
        sys.executable, "-m", "rve_vam.cli",
        "--mesh", str(mesh_path),
        "--materials", str(material_path),
        "--output", str(out_dir),
        "--solver", solver,
        "--solver-rtol", str(solver_rtol),
    ]

    if not parallel:
        cmd.append("--no-parallel")
    else:
        cmd.extend(["--parallel-workers", str(parallel_workers)])

    if not write_fields:
        # 不写场文件，只做均质化
        pass
    else:
        cmd.extend([
            "--write-fields",
            "--load-steps", "1",
            "--field-output-dir", str(out_dir / "fields"),
            "--field-prefix", "macro_strain",
        ])

    # 重置 logging handlers，避免跨算例问题
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)

    # 执行命令
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path(__file__).parent),
        capture_output=False,
        text=True,
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        raise RuntimeError(f"计算失败，返回码: {result.returncode}")

    return {
        "case_name": case_dir.name,
        "success": True,
        "elapsed_seconds": elapsed,
        "output_dir": str(out_dir),
        "stiffness_json": str(out_dir / "stiffness.json"),
        "stiffness_csv": str(out_dir / "stiffness.csv"),
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
    parser.add_argument("--solver-rtol", default=1e-5, type=float, help="CG求解器相对残差阈值 (default: 1e-5, 1e-4更快但精度稍低)")
    parser.add_argument("--no-parallel", action="store_true", help="禁用并行求解")
    parser.add_argument("--parallel-workers", default=2, type=int, help="并行worker数量 (大网格推荐2-3个，避免内存带宽瓶颈)")

    args = parser.parse_args()

    import os

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
