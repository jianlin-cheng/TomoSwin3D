#!/usr/bin/env python3
"""
Run full tomogram preprocessing pipeline for prediction in one command.

Author: Ashwin Dhakal
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_stage(stage_name, cmd, repo_root, stop_on_error=True):
    print("\n" + "=" * 90)
    print(f"[START] {stage_name}")
    print("Command:", " ".join(cmd))
    print("=" * 90)
    start = time.time()

    result = subprocess.run(cmd, cwd=str(repo_root))

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"[DONE]  {stage_name} ({elapsed:.1f}s)")
        return True

    print(f"[FAIL]  {stage_name} (exit code: {result.returncode}, {elapsed:.1f}s)")
    if stop_on_error:
        raise RuntimeError(f"Pipeline stopped at stage: {stage_name}")
    return False


def resolve_scripts(repo_root):
    scripts_dir = repo_root / "utils" / "preparing_tomograms_for_prediction"
    return {
        "step1": scripts_dir / "step_1_normalize_tomogram.py",
        "step2a": scripts_dir / "step_2a_generate_DoG_blob_features.py",
        "step2b": scripts_dir / "step_2b_generate_sobel_features.py",
        "step2c": scripts_dir / "step_2c_generate_tophat_features.py",
        "step3": scripts_dir / "step_3_split_tomograms_into_grids.py",
        "step4a": scripts_dir / "step_4a_split_tomograms_sobel_gradmag_features_into_grids.py",
        "step4b": scripts_dir / "step_4b_split_tomograms_tophat_combined__features_into_grids.py",
        "step4c": scripts_dir / "step_4c_split_tomograms_DoG_blob_features_into_grids.py",
    }


def validate_scripts_exist(scripts):
    missing = [str(path) for path in scripts.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Missing required pipeline scripts:\n{missing_text}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all tomogram prediction preprocessing steps sequentially."
    )
    parser.add_argument(
        "--input-path",
        default="sample_input_data/tomogram_collection",
        help="Tomogram collection root, relative to repo root by default.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=48,
        help="Grid size used in step 3/4 splitting scripts.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Padding used in step 3/4 splitting scripts.",
    )
    parser.add_argument(
        "--default-voxel-size",
        type=float,
        default=1,
        help="Fallback voxel size (angstrom) for unknown dataset folder names in step 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker count passed to top-hat step (step 2c).",
    )
    parser.add_argument(
        "--no-multiprocessing-tophat",
        action="store_true",
        help="Disable multiprocessing in top-hat stage (step 2c).",
    )
    parser.add_argument(
        "--parallel-scales-tophat",
        action="store_true",
        help="Enable experimental parallel scales mode in top-hat stage (step 2c).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining stages even if one stage fails.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    scripts = resolve_scripts(repo_root)
    validate_scripts_exist(scripts)

    stop_on_error = not args.continue_on_error
    py = sys.executable

    step1_cmd = [py, str(scripts["step1"]), "--input-path", args.input_path]
    if args.default_voxel_size is not None:
        step1_cmd += ["--default-voxel-size", str(args.default_voxel_size)]

    run_stage("Step 1 - Normalize tomograms", step1_cmd, repo_root, stop_on_error)

    run_stage(
        "Step 2a - Generate DoG blob features",
        [py, str(scripts["step2a"]), "--root-dir", args.input_path],
        repo_root,
        stop_on_error,
    )
    run_stage(
        "Step 2b - Generate Sobel gradient features",
        [py, str(scripts["step2b"]), "--root-dir", args.input_path],
        repo_root,
        stop_on_error,
    )

    step2c_cmd = [py, str(scripts["step2c"]), "--input-dir", args.input_path]
    if args.workers is not None:
        step2c_cmd += ["--workers", str(args.workers)]
    if args.no_multiprocessing_tophat:
        step2c_cmd += ["--no-multiprocessing"]
    if args.parallel_scales_tophat:
        step2c_cmd += ["--parallel-scales"]

    run_stage("Step 2c - Generate Top-hat features", step2c_cmd, repo_root, stop_on_error)

    run_stage(
        "Step 3 - Split normalized tomograms into grids",
        [
            py,
            str(scripts["step3"]),
            "--base-dir",
            str((repo_root / args.input_path).resolve()),
            "--grid-size",
            str(args.grid_size),
            "--padding",
            str(args.padding),
        ],
        repo_root,
        stop_on_error,
    )
    run_stage(
        "Step 4a - Split Sobel feature maps into grids",
        [
            py,
            str(scripts["step4a"]),
            "--base-dir",
            str((repo_root / args.input_path).resolve()),
            "--grid-size",
            str(args.grid_size),
            "--padding",
            str(args.padding),
        ],
        repo_root,
        stop_on_error,
    )
    run_stage(
        "Step 4b - Split Top-hat feature maps into grids",
        [
            py,
            str(scripts["step4b"]),
            "--base-dir",
            str((repo_root / args.input_path).resolve()),
            "--grid-size",
            str(args.grid_size),
            "--padding",
            str(args.padding),
        ],
        repo_root,
        stop_on_error,
    )
    run_stage(
        "Step 4c - Split DoG feature maps into grids",
        [
            py,
            str(scripts["step4c"]),
            "--base-dir",
            str((repo_root / args.input_path).resolve()),
            "--grid-size",
            str(args.grid_size),
            "--padding",
            str(args.padding),
        ],
        repo_root,
        stop_on_error,
    )

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
