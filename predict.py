"""
Author: Ashwin Dhakal
"""

import math
import numpy as np

import os

from dataset.dataset import CryoEMTestDataset
from torch.utils.data import DataLoader
import torch
import config
from glob import glob
from models.model import SwinUnet3D
from torch.nn.parallel import DataParallel
import os
from scipy.ndimage import maximum_filter
import shutil
import mrcfile
from datetime import datetime
from math import ceil


def extract_value(val):
    """Extract scalar value from various formats."""
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return int(val.item())
        else:
            return int(val.flatten()[0])
    elif isinstance(val, (list, tuple)):
        return int(val[0])
    else:
        return int(val)


def reconstruct_volume_from_grids(grid_files, padding=8):
    """
    Reconstruct full volume from grid files using the provided logic.
    
    Args:
        grid_files: List of grid file paths (.npz files)
        padding: Padding size used during grid creation
    
    Returns:
        tuple: (reconstructed_volume, metadata_dict) or (None, None) if failed
    """
    try:
        # Load first grid to get metadata
        first_grid = np.load(grid_files[0], allow_pickle=True)
        
        # Extract metadata - handle both direct keys and nested metadata
        if 'metadata' in first_grid:
            metadata = first_grid['metadata'].item()
        else:
            # If metadata is not nested, use the data directly
            metadata = {}
            for key in ['orig_shape', 'grid_size', 'padding', 'voxel_size', 'origin', 'mapc', 'mapr', 'maps']:
                if key in first_grid:
                    metadata[key] = first_grid[key]
        
        # Handle different orig_shape formats
        if isinstance(metadata['orig_shape'], np.ndarray):
            if metadata['orig_shape'].ndim > 0:
                original_shape = metadata['orig_shape'].flatten()
            else:
                # Handle 0-dimensional array
                original_shape = [int(metadata['orig_shape'].item())]
        else:
            original_shape = list(metadata['orig_shape'])
        
        print(f"📊 Original shape: {original_shape}")
        print(f"📊 Padding: {padding}")
        
        # Initialize reconstruction volume
        volume = np.zeros(original_shape, dtype=np.float32)
        
        # Reconstruct from all grids
        print(f"🔧 Reconstructing volume from {len(grid_files)} grids...")
        
        successful_grids = 0
        failed_grids = 0
        
        for grid_file in grid_files:
            try:
                data = np.load(grid_file, allow_pickle=True)
                
                # Handle both data structures
                if 'data' in data and 'metadata' in data:
                    # Nested structure
                    grid = data['data']
                    grid_metadata = data['metadata'].item()
                else:
                    # Direct structure
                    grid = data['grid']
                    grid_metadata = data
                
                # Extract position information with robust handling
                try:
                    i = extract_value(grid_metadata['i'])
                    j = extract_value(grid_metadata['j'])
                    k = extract_value(grid_metadata['k'])
                    di = extract_value(grid_metadata['di'])
                    dj = extract_value(grid_metadata['dj'])
                    dk = extract_value(grid_metadata['dk'])
                except Exception as meta_error:
                    print(f"Metadata extraction failed for {os.path.basename(grid_file)}: {str(meta_error)}")
                    failed_grids += 1
                    continue
                
                # Extract central region (remove padding)
                try:
                    central_grid = grid[padding:padding+di, padding:padding+dj, padding:padding+dk]
                    volume[i:i+di, j:j+dj, k:k+dk] = central_grid
                    successful_grids += 1
                except Exception as grid_error:
                    print(f"Grid processing failed for {os.path.basename(grid_file)}: {str(grid_error)}")
                    failed_grids += 1
                    continue
                    
            except Exception as e:
                print(f"Error processing {os.path.basename(grid_file)}: {str(e)}")
                failed_grids += 1
                continue
        
        print(f"✅ Successfully processed: {successful_grids} grids")
        print(f"❌ Failed to process: {failed_grids} grids")
        
        if failed_grids > 0:
            raise Exception(f"Failed to process {failed_grids} out of {len(grid_files)} grids")
        
        # Prepare metadata for MRC file
        mrc_metadata = {
            'voxel_size': metadata.get('voxel_size'),
            'origin': metadata.get('origin'),
            'mapc': metadata.get('mapc', 1),
            'mapr': metadata.get('mapr', 2),
            'maps': metadata.get('maps', 3)
        }
        
        return volume, mrc_metadata
        
    except Exception as e:
        print(f"Volume reconstruction failed: {str(e)}")
        raise e


def save_volume_as_mrc(volume, metadata, output_path):
    """
    Save reconstructed volume as MRC file with proper metadata.
    
    Args:
        volume: 3D numpy array
        metadata: Dictionary containing MRC metadata
        output_path: Path to save the MRC file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(volume.astype(np.float32))
            
            # Set voxel size
            if 'voxel_size' in metadata and metadata['voxel_size'] is not None:
                mrc.voxel_size = metadata['voxel_size']
            
            # Set origin
            if 'origin' in metadata and metadata['origin'] is not None:
                mrc.header.origin = metadata['origin']
            
            # Set axis mapping
            if 'mapc' in metadata and metadata['mapc'] is not None:
                mrc.header.mapc = metadata['mapc']
            if 'mapr' in metadata and metadata['mapr'] is not None:
                mrc.header.mapr = metadata['mapr']
            if 'maps' in metadata and metadata['maps'] is not None:
                mrc.header.maps = metadata['maps']
        
        print(f"✅ Successfully saved MRC file: {output_path}")
        print(f"📊 Volume shape: {volume.shape}")
        print(f"📊 Volume data type: {volume.dtype}")
        print(f"📊 Volume range: [{volume.min():.6f}, {volume.max():.6f}]")
        print(f"📊 Non-zero voxels: {np.count_nonzero(volume):,} / {volume.size:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save MRC file: {str(e)}")
        raise e


#there are new changes to be reflected
# data_ids = ['shrec_2021_model_9', 'shrec_2021_model_8', 'shrec_2020_model_9', 'MaxPlanck_model_r9_08', 'MaxPlanck_model_r1_08', 'MaxPlanck_model_r11_08', 'MaxPlanck_model_r10_08', 'CryoETPortal_model_26', \
    # 'CryoETPortal_model_25', 'CryoETPortal_model_24']
data_ids = ['tomogram_ID_1']  

# Model Configuration Parameters
comment = ''
threshold = 0.7  # Global threshold for multiclass prediction if binary use 0.9

def create_detailed_filename(data_id, comment, model_checkpoint):
    """
    Create filename with full model checkpoint name and comment.
    """
    # Extract the full model checkpoint name (without path and extension)
    model_name = os.path.basename(model_checkpoint).replace('.pth', '')
    
    # Create filename: data_id_modelname_comment_pred.mrc
    filename = "{}{}{}.mrc".format(data_id, model_name, comment)
    return filename
    
# Load model checkpoint and extract architecture
model_checkpoint ="pretrained_models/TomoSwin3D_model_1.pth"
checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu'))

# Extract configuration from checkpoint
if 'config' in checkpoint:
    print("✅ Loading model configuration from checkpoint...")
    config_data = checkpoint['config']
    
    # Extract all architecture parameters from checkpoint
    prediction_type = config_data['prediction_type']
    hidden_dimension = config_data['hidden_dimension']
    layers = config_data['layers']
    heads = config_data['heads']
    downscaling_factors = config_data['downscaling_factors']
    window_size = config_data['window_size']
    num_classes = config_data['num_classes']
    dropout = config_data['dropout']
    input_channel = config_data['input_channel']
    head_dimension = config_data['head_dimension']
    relative_pos_embedding = config_data['relative_pos_embedding']
    skip_style = config_data['skip_style']
    second_to_last_channels = config_data['second_to_last_channels']
    
    print(f"📊 Loaded configuration from checkpoint:")
    print(f"   - Prediction type: {prediction_type}")
    print(f"   - Hidden dimension: {hidden_dimension}")
    print(f"   - Layers: {layers}")
    print(f"   - Heads: {heads}")
    print(f"   - Num classes: {num_classes}")
    print(f"   - Input channels: {input_channel}")
    
else:
    print("⚠️  No configuration found in checkpoint. Using fallback hardcoded values...")
    # Fallback to hardcoded values for backward compatibility with old checkpoints
    prediction_type = "multiclass_standardized_across_shrec_2021"
    hidden_dimension = 32
    layers = (2, 6, 6, 2)
    heads = (3, 6, 12, 24)
    downscaling_factors = (2, 2, 2, 2)
    window_size = 2
    num_classes = 130
    dropout = 0.5
    input_channel = 4
    head_dimension = 32
    relative_pos_embedding = True
    skip_style = 'add'
    second_to_last_channels = 32

# Initialize model with loaded architecture
model = SwinUnet3D(hidden_dimension = hidden_dimension, layers = layers, heads = heads,
                    downscaling_factors = downscaling_factors, window_size = window_size, num_classes = num_classes, dropout = dropout, input_channel = input_channel,
                    head_dimension = head_dimension, relative_pos_embedding = relative_pos_embedding, 
                    skip_style = skip_style, second_to_last_channels = second_to_last_channels).to(config.device)

model = DataParallel(model) #if the model was trained on parallel fashion, it should be parallel model while testing

# Extract model state dict from checkpoint
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"✅ Loaded model state dict from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
else:
    # Fallback: assume the checkpoint is the state dict itself
    state_dict = checkpoint
    print("✅ Loaded checkpoint as direct state dict")

model.load_state_dict(state_dict)
model.eval()

# Create main timestamped output directory
current_datetime = datetime.now()
timestamp = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
main_output_path = f"output/results/DATETIME_{timestamp}"
os.makedirs(main_output_path, exist_ok=True)

# Process each data ID
total_data_ids = len(data_ids)
successful_processing = []
failed_processing = []

for idx, data_id in enumerate(data_ids, 1):
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING DATA ID {idx}/{total_data_ids}: {data_id}")
    print(f"{'='*80}")
    
    try:
        # Get tomogram path for current data ID
        tomo_path = glob(f"sample_input_data/test_data/Grids_64_normalized/tomograms/{data_id}/*.npz")
        
        if not tomo_path:
            raise Exception(f"No tomogram files found for {data_id}")
        
        test_ds = CryoEMTestDataset(tomo_dir=tomo_path, transform=None)
        print(f"[INFO] Found {len(test_ds)} examples in the testing set for {data_id}...")
        
        test_loader = DataLoader(test_ds, shuffle=True, batch_size=1, pin_memory=config.pin_memory, num_workers=config.num_workers)

        # Create output directories for current data ID
        output_file_path = f"{main_output_path}/{data_id}"
        original_tomogram = f"sample_input_data/tomogram_collection/{data_id}/reconstruction.mrc"   #this is used just to get the original shape of the mrc file # for shrec

        predicted_grid_path = f"{output_file_path}/predicted_{data_id}_grids/"
        output_directory = f"{output_file_path}/predicted_{data_id}_reconstructed/"
        os.makedirs(output_directory, exist_ok = True)
        os.makedirs(predicted_grid_path, exist_ok = True)

        # Load original tomogram metadata
        original_map = mrcfile.open(original_tomogram, permissive=True)
        original_shape = original_map.data.shape
        original_voxel_size = original_map.voxel_size
        print(f"📊 Original Volume Shape for {data_id}: {original_shape}")
        print(f"📊 Voxel size for {data_id}: {original_voxel_size}")

        all_probabilities = []
        
        # Run inference on current data ID
        print(f"🔍 Starting inference for {data_id}...")
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, f_name = data
                x = x.to(config.device)
               
                # Load metadata from original input file
                original_data = np.load(f_name[0], allow_pickle=True)
                
                # Extract metadata
                i_idx = original_data['i']
                j_idx = original_data['j'] 
                k_idx = original_data['k']
                di = original_data['di']
                dj = original_data['dj']
                dk = original_data['dk']
                orig_shape = original_data['orig_shape']
                grid_size_meta = original_data['grid_size']
                padding = original_data['padding']
                voxel_size = original_data['voxel_size']
                origin = original_data['origin']
                mapc = original_data['mapc']
                mapr = original_data['mapr']
                maps = original_data['maps']
                
                logits = model(x)
                if prediction_type.startswith("multiclass"): 
                    probs = torch.softmax(logits, dim = 1)
                    max_probs, preds = torch.max(probs, dim=1)
                    preds[max_probs < threshold] = 0
                    output_mask = (preds.squeeze(0).detach().cpu().numpy()) 
                    new_result = output_mask.astype(int)

                #new
                elif prediction_type == "binary": 
                    probs = torch.softmax(logits, dim=1)
                    particle_probs = probs[:, 1]  # Get particle class probabilities            
                    output_mask = (particle_probs > threshold).float()
                    output_mask = output_mask.squeeze(0).detach().cpu().numpy()
                    new_result = output_mask.astype(int)
                
                else:
                    raise ValueError(f"Unknown prediction type: {prediction_type}")
                
                # Create filename and filepath
                filename = f"grid_i{i_idx}_j{j_idx}_k{k_idx}.npz"
                filepath = os.path.join(predicted_grid_path, filename)
                
                # Save grid with its metadata
                np.savez(filepath,
                        grid=new_result,
                        i=i_idx, j=j_idx, k=k_idx,
                        di=di, dj=dj, dk=dk,
                        orig_shape=orig_shape,
                        grid_size=grid_size_meta,
                        padding=padding,
                        voxel_size=voxel_size,
                        origin=origin,
                        mapc=mapc,
                        mapr=mapr,
                        maps=maps)
                
                # Close the original data file
                original_data.close()
        
        print(f"✅ Inference completed for {data_id}")
        
        # Reconstruct the 3D volume from predicted grids
        print(f"\n🔧 Starting 3D volume reconstruction for {data_id}...")
        print(f"📁 Predicted grids directory: {predicted_grid_path}")

        # Find all grid files
        grid_files = glob(os.path.join(predicted_grid_path, "grid_i*_j*_k*.npz"))
        if not grid_files:
            # Try alternative pattern
            grid_files = glob(os.path.join(predicted_grid_path, "*.npz"))

        if not grid_files:
            raise Exception(f"No grid files found in {predicted_grid_path}")

        print(f"✅ Found {len(grid_files)} grid files for {data_id}")

        # Reconstruct volume
        reconstructed_volume, mrc_metadata = reconstruct_volume_from_grids(grid_files, padding=8)

        if reconstructed_volume is None:
            raise Exception(f"Volume reconstruction failed for {data_id}")

        # Save as MRC file with simple naming convention
        output_filename = create_detailed_filename(data_id, comment, model_checkpoint)
        output_path = os.path.join(output_directory, output_filename)

        print(f"💾 Saving reconstructed volume as MRC file for {data_id}...")
        save_volume_as_mrc(reconstructed_volume, mrc_metadata, output_path)

        print(f"🎉 Successfully reconstructed and saved: {output_path}")
        successful_processing.append(data_id)
        
    except Exception as e:
        error_msg = f"❌ ERROR processing {data_id}: {str(e)}"
        print(error_msg)
        failed_processing.append((data_id, str(e)))
        continue

# Print final summary
print(f"\n{'='*80}")
print(f"📊 PROCESSING SUMMARY")
print(f"{'='*80}")
print(f"✅ Successfully processed: {len(successful_processing)}/{total_data_ids}")
for data_id in successful_processing:
    print(f"   - {data_id}")

if failed_processing:
    print(f"❌ Failed to process: {len(failed_processing)}/{total_data_ids}")
    for data_id, error in failed_processing:
        print(f"   - {data_id}: {error}")
else:
    print(f"🎉 All data IDs processed successfully!")

print(f"\n📁 Main output directory: {main_output_path}")
