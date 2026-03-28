#!/usr/bin/env python3
# Author: Ashwin Dhakal
import os
import time
import argparse
import numpy as np
import mrcfile
from scipy.ndimage import sobel
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable
from functools import wraps
import multiprocessing as mp


# Captures edges/boundaries of particles.

ROOT_DIR = 'sample_input_data/tomogram_collection'
INPUT_NAME = 'reconstruction_normalized_map.mrc'

OUT_MAG = 'tomogram_feature_maps/tomogram_filter_sobel_gradmag.mrc'

# ---- Performance optimization params ----
USE_MULTIPROCESSING = True        # Use multiprocessing for file processing
MAX_WORKERS = min(8, mp.cpu_count())  # Number of parallel workers


def time_function(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"    {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def ensure_finite_array(vol: np.ndarray) -> np.ndarray:
    """Ensure array has finite values, replacing NaN/Inf with 0."""
    vol = np.asarray(vol, dtype=np.float32)
    if not np.isfinite(vol).all():
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return vol


def write_mrc_optimized(path_out: str, data: np.ndarray, voxel_size_xyz, origin=None):
    """Optimized MRC file writing with proper error handling."""
    vol = np.ascontiguousarray(data.astype(np.float32))

    with mrcfile.new(path_out, overwrite=True) as mrc:
        mrc.set_data(vol)

        if voxel_size_xyz is not None:
            try:
                mrc.voxel_size = voxel_size_xyz
            except Exception:
                try:
                    mrc.voxel_size = (
                        float(voxel_size_xyz.x),
                        float(voxel_size_xyz.y),
                        float(voxel_size_xyz.z),
                    )
                except Exception:
                    pass

        if origin is not None:
            try:
                mrc.header.origin = origin
            except Exception:
                pass

        mrc.flush()


def find_input_files(root_dir: str, input_name: str) -> List[str]:
    """Find all input files matching the pattern in the directory tree."""
    input_files = []
    for root, dirs, files in os.walk(root_dir):
        if input_name in files:
            in_path = os.path.join(root, input_name)
            input_files.append(in_path)
    return input_files


def process_files_parallel(
    file_paths: List[str],
    process_func: Callable,
    max_workers: int,
    use_multiprocessing: bool,
) -> int:
    """Process files in parallel using multiprocessing."""
    if not file_paths:
        return 0

    print(f'Processing {len(file_paths)} files with {max_workers} workers...')

    if use_multiprocessing and len(file_paths) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_func, file_paths))
        successful = sum(results)
    else:
        successful = 0
        for file_path in file_paths:
            if process_func(file_path):
                successful += 1

    return successful

@time_function
def compute_sobel_gradients(vol: np.ndarray) -> tuple:
    """Compute 3D Sobel gradients with optimized operations."""
    vol = ensure_finite_array(vol)
    
    # 3D Sobel per axis (in intensity per voxel)
    # mode='nearest' matches the earlier defaults to avoid padding artifacts
    gx_vox = sobel(vol, axis=2, mode='nearest')  # x-direction
    gy_vox = sobel(vol, axis=1, mode='nearest')  # y-direction
    gz_vox = sobel(vol, axis=0, mode='nearest')  # z-direction
    
    return gx_vox, gy_vox, gz_vox


def process_one(mrc_path: str):
    """Process a single MRC file with Sobel filtering."""
    print(f'--> Processing: {mrc_path}')
    try:
        with mrcfile.open(mrc_path, permissive=True) as orig:
            vol = orig.data.copy().astype(np.float32)
            voxel_size = orig.voxel_size   # namedtuple (x, y, z) in Å/pixel
            origin = getattr(orig.header, 'origin', None)

        # Report voxel size
        try:
            print(f'    voxel size (Å/px): x={voxel_size.x:.3f}, y={voxel_size.y:.3f}, z={voxel_size.z:.3f}')
            sx, sy, sz = float(voxel_size.x), float(voxel_size.y), float(voxel_size.z)
        except Exception:
            # Fallback: assume isotropic 1 Å/px if missing (rare)
            print('    (voxel size missing or malformed; assuming 1.0 Å/px isotropic)')
            sx = sy = sz = 1.0

        # Compute Sobel gradients
        gx_vox, gy_vox, gz_vox = compute_sobel_gradients(vol)

        # Convert to physical gradients (intensity per Å)
        # Vectorized division for better performance
        gx = gx_vox / sx
        gy = gy_vox / sy
        gz = gz_vox / sz

        # Gradient magnitude - vectorized computation
        gradmag = np.sqrt(gx*gx + gy*gy + gz*gz)

        # Output paths
        d = os.path.dirname(mrc_path)
        out_mag = os.path.join(d, OUT_MAG)

        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(out_mag), exist_ok=True)

        # Write gradient magnitude only (preserve voxel size & origin)
        write_mrc_optimized(out_mag, gradmag, voxel_size, origin=origin)

        print(f'    wrote: {out_mag}')
        return True
        
    except Exception as e:
        print(f'    ERROR processing {mrc_path}: {e}')
        return False


def main():
    """Main function with optimized parallel processing."""
    parser = argparse.ArgumentParser(description='Generate Sobel gradient magnitude feature maps.')
    parser.add_argument(
        '--root-dir',
        default=ROOT_DIR,
        help='Dataset root directory (relative to repository root by default).',
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    root_dir = os.path.abspath(os.path.join(repo_root, args.root_dir))

    print(f'Searching for "{INPUT_NAME}" files in {root_dir}...')
    input_files = find_input_files(root_dir, INPUT_NAME)
    
    if not input_files:
        print(f'No "{INPUT_NAME}" files found under {root_dir}')
        return
    
    print(f'Found {len(input_files)} files to process')
    
    # Process files
    successful = process_files_parallel(input_files, process_one, MAX_WORKERS, USE_MULTIPROCESSING)
    
    print(f'Done. Successfully processed {successful}/{len(input_files)} volume(s).')

if __name__ == '__main__':
    main()
