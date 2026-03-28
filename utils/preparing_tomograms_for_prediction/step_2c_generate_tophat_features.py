#!/usr/bin/env python3
# Author: Ashwin Dhakal
import os
import time
import numpy as np
import mrcfile
from scipy.ndimage import white_tophat, black_tophat
from scipy.ndimage import generate_binary_structure
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import argparse
from typing import List, Callable
from functools import wraps

# Why: Specifically designed to find bright particles on dark background (typical in cryo-ET)
# Best for: Removing gradual intensity variations while enhancing small, bright protein particles
# Advantage: Very effective for the typical contrast in cryo-ET images


ROOT_DIR = 'sample_input_data/tomogram_collection'
INPUT_NAME = 'reconstruction_normalized_map.mrc'

OUT_COMBINED = 'tomogram_feature_maps/tomogram_filter_tophat_combined.mrc'

# ---- Top-hat params ----
SIGMAS = [1.0, 2.0, 4.0, 8.0]     # in voxels (covers a wide range if particle size is unknown)
STRUCTURE_ELEMENT = 'ball'         # 'ball', 'cube', or 'cross' for morphological operations
COMBINE_METHOD = 'max'             # 'max', 'sum', or 'weighted' for combining scales

# ---- Performance optimization params ----
USE_MULTIPROCESSING = True        # Use multiprocessing for file processing
MAX_WORKERS = min(8, mp.cpu_count())  # Number of parallel workers
PARALLEL_SCALES = False           # Process different scales in parallel (experimental)


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

def get_structure_element(size, structure_type='ball'):
    """Generate structuring element for morphological operations."""
    if structure_type == 'ball':
        # Create a ball-shaped structuring element
        radius = max(1, int(size))
        return generate_binary_structure(3, radius)
    elif structure_type == 'cube':
        # Create a cube-shaped structuring element
        size_int = max(1, int(size * 2) + 1)
        return np.ones((size_int, size_int, size_int), dtype=bool)
    else:  # cross
        # Create a cross-shaped structuring element
        return generate_binary_structure(3, 1)

@time_function
def compute_tophat_multiscale(vol: np.ndarray,
                             sigmas=SIGMAS,
                             structure_type=STRUCTURE_ELEMENT,
                             combine_method=COMBINE_METHOD) -> np.ndarray:
    """
    Returns combined top-hat response across scales.
    Combined response uses both white and black top-hat features.
    Optimized version with vectorized operations and memory efficiency.
    """
    vol = ensure_finite_array(vol)

    # Initialize accumulators
    combined_max = np.zeros_like(vol, dtype=np.float32)

    for s in sigmas:
        s_val = float(s)
        
        # Generate structuring element based on sigma
        se = get_structure_element(s_val, structure_type)
        
        # White top-hat: bright particles on dark background
        white_th = white_tophat(vol, structure=se, mode='nearest')
        
        # Black top-hat: dark particles on bright background
        black_th = black_tophat(vol, structure=se, mode='nearest')
        
        # Scale normalization (optional)
        if s_val > 0:
            white_th = white_th * (s_val ** 0.5)  # Gentle scale normalization
            black_th = black_th * (s_val ** 0.5)
        
        # Combine across scales
        if combine_method == 'max':
            combined_max = np.maximum(combined_max, np.maximum(white_th, black_th))
        elif combine_method == 'sum':
            combined_max += np.maximum(white_th, black_th)
        else:  # weighted
            weight = 1.0 / (s_val + 1e-6)
            combined_max += np.maximum(white_th, black_th) * weight

    # Final numeric cleanup - use more efficient percentile calculation
    p01_combined, p99_combined = np.percentile(combined_max, [0.01, 99.99])
    
    combined_max = np.clip(combined_max, p01_combined, p99_combined)
    
    return combined_max.astype(np.float32)


def compute_tophat_single_scale(vol: np.ndarray, sigma: float, structure_type: str) -> np.ndarray:
    """Compute combined TopHat for a single scale - used for parallel scale processing."""
    vol = ensure_finite_array(vol)
    s_val = float(sigma)
    
    # Generate structuring element based on sigma
    se = get_structure_element(s_val, structure_type)
    
    # White top-hat: bright particles on dark background
    white_th = white_tophat(vol, structure=se, mode='nearest')
    
    # Black top-hat: dark particles on bright background
    black_th = black_tophat(vol, structure=se, mode='nearest')
    
    # Scale normalization (optional)
    if s_val > 0:
        white_th = white_th * (s_val ** 0.5)  # Gentle scale normalization
        black_th = black_th * (s_val ** 0.5)
    
    return np.maximum(white_th, black_th)


def compute_tophat_multiscale_parallel(vol: np.ndarray,
                                      sigmas=SIGMAS,
                                      structure_type=STRUCTURE_ELEMENT,
                                      combine_method=COMBINE_METHOD) -> np.ndarray:
    """
    Parallel version of TopHat multiscale computation.
    Processes different scales in parallel for better performance on multi-core systems.
    """
    vol = ensure_finite_array(vol)
    
    if not PARALLEL_SCALES or len(sigmas) < 2:
        # Fall back to sequential processing
        return compute_tophat_multiscale(vol, sigmas, structure_type, combine_method)
    
    # Process scales in parallel
    with ProcessPoolExecutor(max_workers=min(len(sigmas), MAX_WORKERS)) as executor:
        # Submit all scale computations
        future_to_sigma = {
            executor.submit(compute_tophat_single_scale, vol, sigma, structure_type): sigma 
            for sigma in sigmas
        }
        
        # Collect results
        scale_results = {}
        for future in future_to_sigma:
            sigma = future_to_sigma[future]
            try:
                combined_scale = future.result()
                scale_results[sigma] = combined_scale
            except Exception as e:
                print(f"    Warning: Scale {sigma} failed: {e}")
                continue
    
    # Initialize accumulators
    combined_max = np.zeros_like(vol, dtype=np.float32)
    
    # Combine results from all scales
    for sigma in sigmas:
        if sigma not in scale_results:
            continue
            
        combined_scale = scale_results[sigma]
        
        # Combine across scales
        if combine_method == 'max':
            combined_max = np.maximum(combined_max, combined_scale)
        elif combine_method == 'sum':
            combined_max += combined_scale
        else:  # weighted
            weight = 1.0 / (sigma + 1e-6)
            combined_max += combined_scale * weight

    # Final numeric cleanup
    p01_combined, p99_combined = np.percentile(combined_max, [0.01, 99.99])
    
    combined_max = np.clip(combined_max, p01_combined, p99_combined)
    
    return combined_max.astype(np.float32)

def process_one(mrc_path: str, parallel_scales: bool = PARALLEL_SCALES):
    """Process a single MRC file with TopHat filtering."""
    print(f'--> Processing: {mrc_path}')
    try:
        with mrcfile.open(mrc_path, permissive=True) as orig:
            orig_data   = orig.data.copy()
            orig_voxel  = orig.voxel_size         # namedtuple (x, y, z) in Å/pixel
            orig_origin = getattr(orig.header, 'origin', None)

        # Report voxel size for transparency
        try:
            print(f'    voxel size (Å/px): x={orig_voxel.x:.3f}, y={orig_voxel.y:.3f}, z={orig_voxel.z:.3f}')
        except Exception:
            print('    (voxel size not found or malformed)')

        # Apply TopHat filtering (with optional parallel scale processing)
        if parallel_scales:
            combined = compute_tophat_multiscale_parallel(
                orig_data, 
                sigmas=SIGMAS, 
                structure_type=STRUCTURE_ELEMENT,
                combine_method=COMBINE_METHOD
            )
        else:
            combined = compute_tophat_multiscale(
                orig_data, 
                sigmas=SIGMAS, 
                structure_type=STRUCTURE_ELEMENT,
                combine_method=COMBINE_METHOD
            )

        # Output paths
        d = os.path.dirname(mrc_path)
        out_combined = os.path.join(d, OUT_COMBINED)

        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(out_combined), exist_ok=True)

        # Write combined result using optimized function
        write_mrc_optimized(out_combined, combined, orig_voxel, origin=orig_origin)
        
        print(f'    wrote: {out_combined}')
        return True
        
    except Exception as e:
        print(f'    ERROR processing {mrc_path}: {e}')
        return False


# Global variable to store parallel_scales setting for multiprocessing
_parallel_scales_setting = PARALLEL_SCALES

def process_one_wrapper(mrc_path: str):
    """Wrapper function for multiprocessing that uses the global parallel_scales setting."""
    return process_one(mrc_path, _parallel_scales_setting)

def main():
    """Main function with optimized parallel processing and command-line options."""
    parser = argparse.ArgumentParser(description='Optimized TopHat filter for CryoET data')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                       help=f'Number of parallel workers (default: {MAX_WORKERS})')
    parser.add_argument('--no-multiprocessing', action='store_true',
                       help='Disable multiprocessing (sequential processing)')
    parser.add_argument('--parallel-scales', action='store_true',
                       help='Enable parallel scale processing (experimental)')
    parser.add_argument('--input-dir', type=str, default=ROOT_DIR,
                       help=f'Input directory to search (default: {ROOT_DIR})')
    parser.add_argument('--input-name', type=str, default=INPUT_NAME,
                       help=f'Input filename pattern (default: {INPUT_NAME})')
    
    args = parser.parse_args()
    
    # Use local variables instead of modifying globals
    use_multiprocessing = not args.no_multiprocessing
    max_workers = args.workers
    parallel_scales = args.parallel_scales
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_dir = os.path.abspath(os.path.join(repo_root, args.input_dir))
    input_name = args.input_name
    
    print(f'TopHat Filter Configuration:')
    print(f'  Input directory: {input_dir}')
    print(f'  Input pattern: {input_name}')
    print(f'  Multiprocessing: {use_multiprocessing}')
    print(f'  Max workers: {max_workers}')
    print(f'  Parallel scales: {parallel_scales}')
    print()
    
    print(f'Searching for "{input_name}" files in {input_dir}...')
    input_files = find_input_files(input_dir, input_name)
    
    if not input_files:
        print(f'No "{input_name}" files found under {input_dir}')
        return
    
    print(f'Found {len(input_files)} files to process')
    
    # Set the global parallel_scales setting for multiprocessing
    global _parallel_scales_setting
    _parallel_scales_setting = parallel_scales
    
    # Process files using optimized parallel processing
    successful = process_files_parallel(input_files, process_one_wrapper, max_workers, use_multiprocessing)
    
    print(f'Done. Successfully processed {successful}/{len(input_files)} volume(s).')

if __name__ == '__main__':
    main()
