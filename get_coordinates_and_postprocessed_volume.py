#!/usr/bin/env python3
"""
Integrated Post-processing Script for CryoET Protein Particle Predictions.

This script processes predicted MRC volumes to:
1. Eliminate noise using minimum blob size threshold
2. Find connected components of protein particles 
3. Calculate centroids of protein blobs
4. Export coordinates to CSV format

Author: Ashwin Dhakal
Date: 2025-09-17
Updated: 2026-03-24 
"""

import numpy as np
import mrcfile
import pandas as pd
import os
import argparse
from scipy import ndimage
import time
from tqdm import tqdm
import logging
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProteinParticleProcessor:
    """
    A class to process predicted protein particle volumes and extract coordinates.
    """
    
    def __init__(self, min_blob_size=5, connectivity=1, background_value=0.0):
        """
        Initialize the processor.
        
        Args:
            min_blob_size (int): Minimum number of voxels for a valid protein blob
            connectivity (int): Connectivity for connected components (1=6-connectivity, 2=18-connectivity, 3=26-connectivity)
            background_value (float): Value representing background/noise
        """
        self.min_blob_size = min_blob_size
        self.connectivity = connectivity
        self.background_value = background_value
        
        # Connectivity mapping for user convenience
        self.connectivity_map = {
            1: "6-connectivity (faces only)",
            2: "18-connectivity (faces and edges)", 
            3: "26-connectivity (faces, edges, and corners)"
        }
        
    def load_mrc_volume(self, mrc_path):
        """
        Load MRC volume and extract metadata.
        
        Args:
            mrc_path (str): Path to MRC file
            
        Returns:
            tuple: (data, voxel_size, origin)
        """
        logger.info(f"Loading MRC file: {mrc_path}")
        
        with mrcfile.open(mrc_path, mode='r') as mrc:
            data = mrc.data.copy()
            
            # Extract voxel size (physical spacing)
            # MRC header contains cell dimensions and grid size
            voxel_size = np.array([
                mrc.voxel_size.x,
                mrc.voxel_size.y, 
                mrc.voxel_size.z
            ])
            
            # Origin coordinates (physical origin)
            origin = np.array([
                mrc.header.origin.x,
                mrc.header.origin.y,
                mrc.header.origin.z
            ])
            
            logger.info(f"Volume shape: {data.shape}")
            logger.info(f"Data type: {data.dtype}")
            logger.info(f"Voxel size (Å): {voxel_size}")
            logger.info(f"Origin (Å): {origin}")
            
        return data, voxel_size, origin
    
    def analyze_volume_statistics(self, data):
        """
        Analyze and log volume statistics.
        
        Args:
            data (np.ndarray): Volume data
        """
        unique_values, counts = np.unique(data, return_counts=True)
        total_voxels = data.size
        
        logger.info("Volume Statistics:")
        logger.info("-" * 50)
        logger.info(f"Total voxels: {total_voxels:,}")
        logger.info(f"Unique values: {len(unique_values)}")
        
        # Show distribution of values
        for val, count in zip(unique_values, counts):
            percentage = (count / total_voxels) * 100
            logger.info(f"  Value {val}: {count:,} voxels ({percentage:.2f}%)")
            
        # Identify non-background values (potential protein classes)
        non_background_mask = unique_values != self.background_value
        protein_classes = unique_values[non_background_mask]
        
        if len(protein_classes) > 0:
            logger.info(f"Detected protein classes: {sorted(protein_classes)}")
        else:
            logger.warning("No protein classes detected! All values are background.")
            
        return protein_classes
    
    def find_connected_components(self, data, protein_classes):
        """
        Find connected components for each protein class.
        
        Args:
            data (np.ndarray): Volume data
            protein_classes (np.ndarray): Array of protein class values
            
        Returns:
            dict: Dictionary mapping protein_class -> (labeled_array, num_components)
        """
        logger.info(f"Finding connected components with {self.connectivity_map[self.connectivity]}")
        
        components_dict = {}
        total_components = 0
        
        for protein_class in tqdm(protein_classes, desc="Processing protein classes"):
            # Create binary mask for this protein class
            binary_mask = (data == protein_class)
            
            if not np.any(binary_mask):
                logger.warning(f"No voxels found for protein class {protein_class}")
                continue
                
            # Find connected components
            labeled_array, num_components = ndimage.label(binary_mask, 
                                                        structure=ndimage.generate_binary_structure(3, self.connectivity))
            
            if num_components > 0:
                components_dict[protein_class] = (labeled_array, num_components)
                total_components += num_components
                logger.info(f"Protein class {protein_class}: {num_components} components found")
            
        logger.info(f"Total components found across all classes: {total_components}")
        return components_dict
    
    def filter_components_by_size(self, components_dict):
        """
        Filter connected components by minimum size threshold.
        
        Args:
            components_dict (dict): Dictionary of labeled arrays per protein class
            
        Returns:
            dict: Filtered components dictionary
        """
        logger.info(f"Filtering components with minimum blob size: {self.min_blob_size}")
        
        filtered_dict = {}
        total_before = 0
        total_after = 0
        
        for protein_class, (labeled_array, num_components) in components_dict.items():
            # Get component sizes
            unique_labels, counts = np.unique(labeled_array, return_counts=True)
            
            # Remove background (label 0)
            if unique_labels[0] == 0:
                unique_labels = unique_labels[1:]
                counts = counts[1:]
            
            total_before += len(unique_labels)
            
            # Filter by minimum size
            valid_mask = counts >= self.min_blob_size
            valid_labels = unique_labels[valid_mask]
            valid_counts = counts[valid_mask]
            
            if len(valid_labels) > 0:
                # Create new labeled array with only valid components
                filtered_labeled = np.zeros_like(labeled_array)
                for i, original_label in enumerate(valid_labels):
                    filtered_labeled[labeled_array == original_label] = i + 1
                
                filtered_dict[protein_class] = (filtered_labeled, len(valid_labels))
                total_after += len(valid_labels)
                
                logger.info(f"Protein class {protein_class}: {len(valid_labels)}/{num_components} components kept "
                           f"(sizes: {valid_counts.min()}-{valid_counts.max()} voxels)")
            else:
                logger.warning(f"Protein class {protein_class}: No components meet minimum size threshold")
        
        logger.info(f"Component filtering: {total_after}/{total_before} components kept")
        return filtered_dict
    
    def save_filtered_components_to_mrc(self, filtered_components, original_shape, 
                                         output_path, voxel_size, origin):
        """
        Save filtered connected components to MRC file with original header information.
        
        Args:
            filtered_components (dict): Filtered components dictionary
            original_shape (tuple): Shape of the original volume
            output_path (str): Output MRC file path
            voxel_size (np.ndarray): Voxel size in physical units
            origin (np.ndarray): Origin in physical coordinates
        """
        logger.info(f"Saving filtered components to MRC file: {output_path}")
        
        # Create empty volume with same shape as original
        output_volume = np.zeros(original_shape, dtype=np.float32)
        
        # Combine all protein classes into single volume
        for protein_class, (labeled_array, num_components) in filtered_components.items():
            # Replace non-zero labels with protein class value
            mask = labeled_array > 0
            output_volume[mask] = protein_class
            logger.info(f"Added {num_components} components for protein class {protein_class}")
        
        # Write to MRC file with original header information
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(output_volume)
            
            # Set voxel size
            mrc.voxel_size = tuple(voxel_size)
            
            # Set origin
            mrc.header.origin.x = origin[0]
            mrc.header.origin.y = origin[1]
            mrc.header.origin.z = origin[2]
            
            logger.info(f"MRC file saved with shape {output_volume.shape}")
            logger.info(f"Voxel size: {voxel_size}")
            logger.info(f"Origin: {origin}")
    
    def calculate_centroids(self, filtered_components, voxel_size, origin):
        """
        Calculate centroids of filtered components in physical coordinates.
        Optimized with batch operations for significant speedup (10-100x faster).
        
        Args:
            filtered_components (dict): Filtered components dictionary
            voxel_size (np.ndarray): Voxel size in physical units
            origin (np.ndarray): Origin in physical coordinates
            
        Returns:
            list: List of dictionaries containing particle information
        """
        logger.info("Calculating centroids in physical coordinates (optimized batch mode)...")
        
        particles = []
        protein_id = 1
        
        for protein_class, (labeled_array, num_components) in filtered_components.items():
            logger.info(f"Processing {num_components} components for protein class {protein_class}")
            
            # Get unique labels (excluding background)
            unique_labels = np.unique(labeled_array)
            unique_labels = unique_labels[unique_labels > 0]  # Remove background
            
            if len(unique_labels) == 0:
                logger.warning(f"No valid labels found for protein class {protein_class}")
                continue
            
            # OPTIMIZATION: Batch calculate centroids for all components at once
            # This replaces the loop with a single vectorized operation
            centroids_voxel = ndimage.center_of_mass(
                labeled_array > 0,  # Binary input for center_of_mass
                labels=labeled_array,
                index=unique_labels
            )
            
            # OPTIMIZATION: Batch calculate volumes for all components at once
            volumes = ndimage.sum_labels(
                np.ones_like(labeled_array, dtype=np.int32),
                labeled_array,
                index=unique_labels
            )
            
            # Convert to numpy array for vectorized operations
            centroids_voxel = np.array(centroids_voxel)
            
            # OPTIMIZATION: Vectorized physical coordinate conversion
            # All centroids converted in one operation instead of one-by-one
            centroids_physical = origin + centroids_voxel * 1 #voxel_size thiyo inplace of 1 here
            
            # Build particle list efficiently
            for i, label in enumerate(unique_labels):
                particle_info = {
                    'protein_id': protein_id,
                    'protein_class': int(protein_class),
                    'x': centroids_physical[i, 2],  # Note: MRC convention is often Z,Y,X
                    'y': centroids_physical[i, 1],
                    'z': centroids_physical[i, 0],
                    'volume': int(volumes[i])
                }
                
                particles.append(particle_info)
                protein_id += 1
        
        logger.info(f"Calculated centroids for {len(particles)} particles")
        return particles
    
    def save_to_csv(self, particles, output_path):
        """
        Save particle information to CSV file.
        
        Args:
            particles (list): List of particle dictionaries
            output_path (str): Output CSV file path
        """
        logger.info(f"Saving {len(particles)} particles to CSV: {output_path}")
        
        if not particles:
            logger.warning("No particles to save!")
            # Create empty CSV with headers
            df = pd.DataFrame(columns=['protein_id', 'protein_class', 'x', 'y', 'z', 'volume'])
        else:
            df = pd.DataFrame(particles)
            
            # Sort by protein class and then by protein_id
            df = df.sort_values(['protein_class', 'protein_id']).reset_index(drop=True)
            
            # Log statistics
            logger.info("Particle Statistics:")
            logger.info("-" * 30)
            for protein_class in sorted(df['protein_class'].unique()):
                count = len(df[df['protein_class'] == protein_class])
                logger.info(f"Protein class {protein_class}: {count} particles")
        
        # Save to CSV
        df.to_csv(output_path, index=False, float_format='%.3f')
        logger.info(f"CSV file saved successfully: {output_path}")
        
        return df

def process_single_mrc(mrc_path, output_dir=None, min_blob_size=5, connectivity=1):
    """
    Process a single MRC file and extract particle coordinates.
    Args:
        mrc_path (str): Path to input MRC file
        output_dir (str): Output directory (default: same as input MRC)
        min_blob_size (int): Minimum blob size threshold
        connectivity (int): Connectivity for connected components
        
    Returns:
        str: Path to generated CSV file, or None on failure
    """
    start_time = time.time()
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(mrc_path)
    
    # Create processor
    processor = ProteinParticleProcessor(
        min_blob_size=min_blob_size,
        connectivity=connectivity
    )
    
    # Load volume
    data, voxel_size, origin = processor.load_mrc_volume(mrc_path)
    
    # Analyze statistics
    protein_classes = processor.analyze_volume_statistics(data)
    
    if len(protein_classes) == 0:
        logger.error("No protein classes found in volume!")
        return None
    
    # Find connected components
    components_dict = processor.find_connected_components(data, protein_classes)
    
    if not components_dict:
        logger.error("No connected components found!")
        return None
    
    # Filter by size
    filtered_components = processor.filter_components_by_size(components_dict)
    
    if not filtered_components:
        logger.error("No components passed size filtering!")
        return None
    
    # Save filtered components to MRC file with "postproc_" prefix
    mrc_basename = os.path.splitext(os.path.basename(mrc_path))[0]
    postproc_mrc_filename = f"postproc_{mrc_basename}.mrc"
    postproc_mrc_path = os.path.join(output_dir, postproc_mrc_filename)
    processor.save_filtered_components_to_mrc(
        filtered_components, 
        data.shape, 
        postproc_mrc_path, 
        voxel_size, 
        origin
    )
    
    # Calculate centroids
    particles = processor.calculate_centroids(filtered_components, voxel_size, origin)
    
    # Generate output filename for CSV
    csv_filename = f"{mrc_basename}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Save to CSV
    df = processor.save_to_csv(particles, csv_path)
    
    # Log processing time
    end_time = time.time()
    logger.info(f"CSV generation completed in {end_time - start_time:.2f} seconds")
    
    return csv_path

def process_prediction_directory(prediction_dir, min_blob_size=5, connectivity=1, tomogram_name=None):
    """
    Process all MRC files in a prediction directory structure.
    Expected structure: prediction_dir/tomogram_name/predicted_tomogram_name_reconstructed/*.mrc
    
    Args:
        prediction_dir (str): Path to prediction results directory
        min_blob_size (int): Minimum blob size threshold  
        connectivity (int): Connectivity for connected components
        tomogram_name (str): Specific tomogram to process (None = process all)
    """
    logger.info(f"Processing prediction directory: {prediction_dir}")
    
    # Find tomogram directories
    if tomogram_name:
        # Process specific tomogram
        tomogram_dirs = [tomogram_name]
        logger.info(f"Processing specific tomogram: {tomogram_name}")
    else:
        # Process all tomograms in the directory
        tomogram_dirs = [d for d in os.listdir(prediction_dir) 
                        if os.path.isdir(os.path.join(prediction_dir, d))]
        logger.info(f"Found {len(tomogram_dirs)} tomogram directories to process")
    
    if not tomogram_dirs:
        logger.error(f"No tomogram directories found in {prediction_dir}")
        return
    
    processed_count = 0
    failed_count = 0
    
    for tomo_name in tomogram_dirs:
        logger.info(f"\n{'='*70}")
        logger.info(f"PROCESSING TOMOGRAM: {tomo_name}")
        logger.info(f"{'='*70}")
        
        try:
            # Find the predicted subdirectory
            tomogram_dir = os.path.join(prediction_dir, tomo_name)
            
            if not os.path.exists(tomogram_dir):
                logger.error(f"Tomogram directory not found: {tomogram_dir}")
                failed_count += 1
                continue
            
            # Look for predicted_*_reconstructed subdirectory
            predicted_subdir = f"predicted_{tomo_name}_reconstructed"
            predicted_dir = os.path.join(tomogram_dir, predicted_subdir)
            
            if not os.path.exists(predicted_dir):
                # Try to find any predicted_* subdirectory
                subdirs = [d for d in os.listdir(tomogram_dir) 
                          if os.path.isdir(os.path.join(tomogram_dir, d)) and d.startswith('predicted_')]
                
                if subdirs:
                    predicted_dir = os.path.join(tomogram_dir, subdirs[0])
                    logger.info(f"Using predicted directory: {subdirs[0]}")
                else:
                    logger.error(f"No predicted subdirectory found in {tomogram_dir}")
                    failed_count += 1
                    continue
            
            # Find MRC files in the predicted directory
            mrc_files = [f for f in os.listdir(predicted_dir) if f.endswith('.mrc') and not f.startswith('postproc_')]
            
            if not mrc_files:
                logger.error(f"No MRC files found in {predicted_dir}")
                failed_count += 1
                continue
            
            logger.info(f"Found {len(mrc_files)} MRC file(s) to process")
            
            # Process each MRC file (typically just one per tomogram)
            for mrc_file in mrc_files:
                mrc_path = os.path.join(predicted_dir, mrc_file)
                logger.info(f"\nProcessing MRC file: {mrc_file}")
                
                csv_path = process_single_mrc(
                    mrc_path, 
                    output_dir=predicted_dir,
                    min_blob_size=min_blob_size, 
                    connectivity=connectivity
                )
                
                if csv_path:
                    processed_count += 1
                    logger.info(f"✓ Successfully processed: {tomo_name}")
                else:
                    logger.error(f"✗ Failed to process: {tomo_name}")
                    failed_count += 1
                    
        except Exception as e:
            logger.error(f"Error processing {tomo_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    # Print overall summary
    logger.info(f"\n{'='*70}")
    logger.info(f"PROCESSING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total tomograms processed: {processed_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {processed_count}/{processed_count + failed_count}")
    
def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Post-process CryoET protein particle predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default prediction directory
  python step_1_post_processing_clean_junk_get_coordinates.py
  
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('-i', '--input', type=str, 
                           help='Input MRC file path')
    input_group.add_argument('-d', '--directory', type=str,
                           default='output/results/TomoSwin3D_results',
                           help='Input directory containing predicted MRC files (default: %(default)s)')
    
    # Processing parameters
    parser.add_argument('--min-blob-size', type=int, default=40,
                       help='Minimum blob size (number of voxels) to keep (default: 10)')
    parser.add_argument('--connectivity', type=int, choices=[1, 2, 3], default=2,
                       help='Connectivity for connected components: 1=6-conn, 2=18-conn, 3=26-conn (default: 2)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output directory (default: same as input)')
    parser.add_argument('--tomogram_name', type=str, default=None,
                       help='Specific tomogram to process (if None, process all tomograms in directory)')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("="*70)
    logger.info("CryoET Post-Processing Pipeline Configuration")
    logger.info("="*70)
    logger.info("Processing Parameters:")
    logger.info(f"  Minimum blob size: {args.min_blob_size} voxels")
    logger.info(f"  Connectivity: {args.connectivity} ({'6-connectivity' if args.connectivity==1 else '18-connectivity' if args.connectivity==2 else '26-connectivity'})")
    logger.info("="*70)
    
    if args.input:
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return
        
        csv_path = process_single_mrc(
            args.input, 
            output_dir=args.output,
            min_blob_size=args.min_blob_size,
            connectivity=args.connectivity
        )
        
        if csv_path:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing completed successfully!")
            logger.info(f"Output CSV: {csv_path}")
            logger.info(f"{'='*70}")
        else:
            logger.error("Processing failed!")
            
    else:
        # Process directory (either explicitly provided or default)
        directory = args.directory
        if not os.path.exists(directory):
            logger.error(f"Input directory not found: {directory}")
            return
        
        process_prediction_directory(
            directory,
            min_blob_size=args.min_blob_size,
            connectivity=args.connectivity,
            tomogram_name=args.tomogram_name
        )

if __name__ == "__main__":
    main()