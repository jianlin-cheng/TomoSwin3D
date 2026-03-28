#!/usr/bin/env python3
# Author: Ashwin Dhakal
import os
import math
import argparse
import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

ROOT_DIR = 'sample_input_data/tomogram_collection'
INPUT_NAME = 'reconstruction_normalized_map.mrc'
OUTPUT_NAME = 'tomogram_feature_maps/tomogram_filter_DoG_blob.mrc'

# ---- DoG params (good general-purpose starters) ----
SIGMAS = [1.0, 2.0, 4.0, 8.0]     # in voxels (covers a wide range if particle size is unknown)
K = math.sqrt(2.0)                # DoG large/small sigma ratio
SCALE_NORMALIZE = True            # multiply response by sigma^2 (LoG approx. scale norm)

# ---- Performance optimization params ----
USE_MULTIPROCESSING = True        # Use multiprocessing for file processing
MAX_WORKERS = min(8, mp.cpu_count())  # Number of parallel workers
CHUNK_SIZE = 1                    # Files per chunk for multiprocessing

def compute_dog_multiscale(vol: np.ndarray,
                           sigmas=SIGMAS,
                           k=K,
                           scale_normalize=SCALE_NORMALIZE) -> np.ndarray:
    """
    Returns the per-voxel maximum DoG response across scales.
    DoG(s) = G(s*k) - G(s); optional scale normalization by s^2 (LoG approximation).
    Optimized version with vectorized operations and memory efficiency.
    """
    vol = np.asarray(vol, dtype=np.float32)
    # Ensure finite values - vectorized operation
    if not np.isfinite(vol).all():
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    # Pre-allocate arrays for better memory efficiency
    Z, Y, X = vol.shape
    max_resp = np.zeros_like(vol, dtype=np.float32)
    max_magnitude = np.zeros_like(vol, dtype=np.float32)
    
    # Pre-compute sigma values to avoid repeated calculations
    sigma_pairs = [(float(s), float(s) * float(k)) for s in sigmas]
    
    for s_small, s_large in sigma_pairs:
        # Compute both Gaussian filters
        g_small = gaussian_filter(vol, sigma=s_small, mode='nearest')
        g_large = gaussian_filter(vol, sigma=s_large, mode='nearest')
        
        # Compute DoG response
        dog = g_large - g_small  # classic DoG (approximate LoG up to scale)

        if scale_normalize:
            dog = dog * (s_small ** 2)  # LoG scale normalization proxy

        # Vectorized maximum magnitude selection
        dog_magnitude = np.abs(dog)
        take = dog_magnitude > max_magnitude
        
        # Update max response where magnitude is larger
        max_resp[take] = dog[take]
        max_magnitude[take] = dog_magnitude[take]

    # Final numeric cleanup - use more efficient percentile calculation
    p01, p99 = np.percentile(max_resp, [0.01, 99.99])
    max_resp = np.clip(max_resp, p01, p99)
    return max_resp.astype(np.float32)

def write_mrc(path_out: str, data: np.ndarray, voxel_size_xyz, origin=None):
    # Ensure C-order float32
    vol = np.ascontiguousarray(data.astype(np.float32))
    with mrcfile.new(path_out, overwrite=True) as mrc:
        mrc.set_data(vol)
        # Preserve voxel size (Å/pix) from source header if available
        if voxel_size_xyz is not None:
            try:
                mrc.voxel_size = voxel_size_xyz
            except Exception:
                # Fallback: set each axis explicitly
                mrc.voxel_size = (float(voxel_size_xyz.x),
                                  float(voxel_size_xyz.y),
                                  float(voxel_size_xyz.z))
        # Preserve origin if present
        if origin is not None:
            try:
                mrc.header.origin = origin
            except Exception:
                pass
        mrc.flush()

def process_one(mrc_path: str):
    """Process a single MRC file with DoG filtering."""
    print(f'--> Processing: {mrc_path}')
    try:
        with mrcfile.open(mrc_path, permissive=True) as orig:
            orig_data   = orig.data.copy()
            orig_hdr    = orig.header.copy()
            orig_voxel  = orig.voxel_size         # namedtuple (x, y, z) in Å/pixel
            orig_origin = getattr(orig.header, 'origin', None)

        # Report voxel size for transparency
        try:
            print(f'    voxel size (Å/px): x={orig_voxel.x:.3f}, y={orig_voxel.y:.3f}, z={orig_voxel.z:.3f}')
        except Exception:
            print('    (voxel size not found or malformed)')

        # Apply DoG filtering
        dog = compute_dog_multiscale(orig_data, sigmas=SIGMAS, k=K, scale_normalize=SCALE_NORMALIZE)

        # Write output
        out_path = os.path.join(os.path.dirname(mrc_path), OUTPUT_NAME)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        write_mrc(out_path, dog, orig_voxel, origin=orig_origin)
        print(f'    wrote: {out_path}')
        return True
        
    except Exception as e:
        print(f'    ERROR processing {mrc_path}: {e}')
        return False

def find_input_files(root_dir):
    """Find all input files to process."""
    input_files = []
    for root, dirs, files in os.walk(root_dir):
        if INPUT_NAME in files:
            in_path = os.path.join(root, INPUT_NAME)
            input_files.append(in_path)
    return input_files

def process_files_parallel(file_paths):
    """Process files in parallel using multiprocessing."""
    if not file_paths:
        return 0
    
    print(f'Processing {len(file_paths)} files with {MAX_WORKERS} workers...')
    
    if USE_MULTIPROCESSING and len(file_paths) > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_one, file_paths))
        successful = sum(results)
    else:
        # Sequential processing for single file or when multiprocessing is disabled
        successful = 0
        for file_path in file_paths:
            if process_one(file_path):
                successful += 1
    
    return successful

def main():
    """Main function with optimized parallel processing."""
    parser = argparse.ArgumentParser(description='Generate DoG blob feature maps from normalized tomograms.')
    parser.add_argument(
        '--root-dir',
        default=ROOT_DIR,
        help='Dataset root directory (relative to repository root by default).',
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    root_dir = os.path.abspath(os.path.join(repo_root, args.root_dir))

    print(f'Searching for "{INPUT_NAME}" files in {root_dir}...')
    input_files = find_input_files(root_dir)
    
    if not input_files:
        print(f'No "{INPUT_NAME}" files found under {root_dir}')
        return
    
    print(f'Found {len(input_files)} files to process')
    
    # Process files
    successful = process_files_parallel(input_files)
    
    print(f'Done. Successfully processed {successful}/{len(input_files)} volume(s).')

if __name__ == '__main__':
    main()
