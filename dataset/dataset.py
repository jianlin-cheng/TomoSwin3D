# Code for creating dataset
# Author: Ashwin Dhakal

from torch.utils.data import Dataset
import numpy as np
import mrcfile
import torch
import config
from scipy.ndimage import zoom
import time
from config import args

class CryoEMDataset_NonZeroGrid(Dataset):
    def __init__(self, mask_dir, transform, prediction_type):
        super().__init__()
        self.mask_dir = mask_dir
        self.transform = transform
        self.prediction_type = prediction_type

    def __len__(self):
        return len(self.mask_dir)

    def __getitem__(self, idx):
        mask_path = self.mask_dir[idx]
        mask_data = np.load(mask_path)
        mask = mask_data['grid']  # Extract grid data from npz file
        tomo_path = mask_path.replace(f"{args.mask_type}_{args.prediction_type}_nonzero_grid_masks", 'tomograms')
        
        # DEBUG: Print paths to verify replacement
        # print("="*80)
        # print("DEBUG: Path Replacement Check")
        # print("="*80)
        # print(f"MASK_PATH:\n  {mask_path}")
        # print(f"\nREPLACEMENT STRING:\n  '{args.mask_type}_{args.prediction_type}_nonzero_grid_masks'")
        # print(f"\nTOMO_PATH:\n  {tomo_path}")
        # print(f"\nReplacement {'SUCCESS ✅' if mask_path != tomo_path else 'FAILED ❌'}")
        # print("="*80)
        # exit()

        # Load tomogram
        tomo_data = np.load(tomo_path)
        tomo = tomo_data['grid']  # Extract grid data from npz file
        
        # Load feature maps
        # sobel_gx_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_sobel_gx')
        # sobel_gy_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_sobel_gy')
        # sobel_gz_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_sobel_gz')
        sobel_gradmag_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_sobel_gradmag')
        tophat_combined_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_tophat_combined')
        # white_tophat_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_white_tophat')
        # black_tophat_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_black_tophat')
        dog_blob_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_DoG_blob')
        # log_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_LoG')
        
        # sobel_gx_data = np.load(sobel_gx_path)
        # sobel_gy_data = np.load(sobel_gy_path)
        # sobel_gz_data = np.load(sobel_gz_path)
        sobel_gradmag_data = np.load(sobel_gradmag_path)
        tophat_combined_data = np.load(tophat_combined_path)
        # white_tophat_data = np.load(white_tophat_path)
        # black_tophat_data = np.load(black_tophat_path)
        dog_blob_data = np.load(dog_blob_path)
        # log_data = np.load(log_path)
        
        # sobel_gx = sobel_gx_data['grid']'
        # sobel_gy = sobel_gy_data['grid']
        # sobel_gz = sobel_gz_data['grid']'
        sobel_gradmag = sobel_gradmag_data['grid']
        tophat_combined = tophat_combined_data['grid']
        # white_tophat = white_tophat_data['grid']
        # black_tophat = black_tophat_data['grid']
        dog_blob = dog_blob_data['grid']
        # log = log_data['grid']
        
        # Concatenate all features: [tomogram, sobel_gx, sobel_gy, sobel_gz, sobel_gradmag, tophat_combined, white_tophat, black_tophat, dog_blob, log]
        # combined_features = np.stack([tomo, sobel_gx, sobel_gy, sobel_gz, sobel_gradmag, tophat_combined, white_tophat, black_tophat, dog_blob, log], axis=0)  # Shape: (10, 64, 64, 64)
        combined_features = np.stack([tomo, sobel_gradmag, tophat_combined, dog_blob], axis=0)  # Shape: (4, 64, 64, 64)
        combined_features = torch.from_numpy(combined_features).float()
        mask = torch.from_numpy(mask).long() #float() if binary

        return combined_features, mask
    
class CryoEMDataset_FullGrid(Dataset):
    def __init__(self, tomo_dir, transform, prediction_type):
        super().__init__()
        self.tomo_dir = tomo_dir
        self.transform = transform
        self.prediction_type = prediction_type

    def __len__(self):
        return len(self.tomo_dir)

    def __getitem__(self, idx):
        tomo_path = self.tomo_dir[idx]
        tomo = np.load(tomo_path)
        mask_path = tomo_path.replace('tomograms', f"{args.mask_type}_{args.prediction_type}_full_grid_masks", )

        mask = np.load(mask_path)
        tomo = torch.from_numpy(tomo).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long() #float() if binary

        return tomo, mask
    
    
        
class CryoEMTestDataset(Dataset):
    def __init__(self, tomo_dir, transform):
        super().__init__()
        self.tomo_dir = tomo_dir
        self.transform = transform

    def __len__(self):
        return len(self.tomo_dir)

    def __getitem__(self, idx):
        tomo_path = self.tomo_dir[idx]
        
        # Load tomogram
        tomo_data = np.load(tomo_path)
        tomo = tomo_data['grid']  # Extract grid data from npz file
        
        # Load feature maps
        # sobel_gx_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_sobel_gx')
        # sobel_gy_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_sobel_gy')
        # sobel_gz_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_sobel_gz')
        sobel_gradmag_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_sobel_gradmag')
        tophat_combined_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_tophat_combined')
        # white_tophat_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_white_tophat')
        # black_tophat_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_black_tophat')
        dog_blob_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_DoG_blob')
        # log_path = tomo_path.replace('tomograms', 'tomograms_feature_maps_LoG')
        
        # sobel_gx_data = np.load(sobel_gx_path)
        # sobel_gy_data = np.load(sobel_gy_path)
        # sobel_gz_data = np.load(sobel_gz_path)
        sobel_gradmag_data = np.load(sobel_gradmag_path)
        tophat_combined_data = np.load(tophat_combined_path)
        # white_tophat_data = np.load(white_tophat_path)
        # black_tophat_data = np.load(black_tophat_path)
        dog_blob_data = np.load(dog_blob_path)
        # log_data = np.load(log_path)
        
        # sobel_gx = sobel_gx_data['grid']
        # sobel_gy = sobel_gy_data['grid']
        # sobel_gz = sobel_gz_data['grid']
        sobel_gradmag = sobel_gradmag_data['grid']
        tophat_combined = tophat_combined_data['grid']
        # white_tophat = white_tophat_data['grid']
        # black_tophat = black_tophat_data['grid']
        dog_blob = dog_blob_data['grid']
        # log = log_data['grid']
        
        # Concatenate all features: [tomogram, sobel_gx, sobel_gy, sobel_gz, sobel_gradmag, tophat_combined, white_tophat, black_tophat, dog_blob, log]
        # combined_features = np.stack([tomo, sobel_gx, sobel_gy, sobel_gz, sobel_gradmag, tophat_combined, white_tophat, black_tophat, dog_blob, log], axis=0)  # Shape: (10, 64, 64, 64)
        combined_features = np.stack([tomo, sobel_gradmag, tophat_combined, dog_blob], axis=0)  # Shape: (4, 64, 64, 64)  #use this one later
        # combined_features = np.stack([tomo, sobel_gx,sobel_gradmag, tophat_combined], axis=0)  # Shape: (4, 64, 64, 64)


        combined_features = torch.from_numpy(combined_features).float()

        return combined_features, self.tomo_dir[idx]
    
