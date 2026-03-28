# Code for training
import os

from dataset.dataset import CryoEMDataset_FullGrid, CryoEMDataset_NonZeroGrid
from models.model import SwinUnet3D
from sklearn.model_selection import train_test_split
import numpy as np
import config
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loss import combined_loss
import glob
from tqdm import tqdm
import time
from datetime import datetime
import os
import wandb
from torch.nn.parallel import DataParallel
from datetime import datetime


if config.grid_type == 'full_grid':
    tomo_path = list(glob.glob(config.train_dataset_path + '*.npz'))
    train_path, val_path = train_test_split(tomo_path, test_size=0.2, random_state=42)
    train_ds = CryoEMDataset_FullGrid(tomo_dir=train_path, transform=None, prediction_type=config.prediction_type)
    val_ds = CryoEMDataset_FullGrid(tomo_dir=val_path, transform=None, prediction_type=config.prediction_type)

elif config.grid_type == 'non_zero_grid':
    mask_path = list(glob.glob(config.train_dataset_path + '*.npz'))
    train_path, val_path = train_test_split(mask_path, test_size=0.2, random_state=42)
    train_ds = CryoEMDataset_NonZeroGrid(mask_dir=train_path, transform=None, prediction_type=config.prediction_type)
    val_ds = CryoEMDataset_NonZeroGrid(mask_dir=val_path, transform=None, prediction_type=config.prediction_type)

print(f"[INFO] Found {len(train_ds)} examples in the training set...")
print(f"[INFO] Found {len(val_ds)} examples in the validation set...")

train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.batch_size, pin_memory=config.pin_memory, num_workers=config.num_workers)
val_loader = DataLoader(val_ds, shuffle=True, batch_size=config.batch_size, pin_memory=config.pin_memory, num_workers=config.num_workers)
print(f"[INFO] Train Loader Length {len(train_loader)}...")


    
model_name = f"{config.architecture_name}.pth"

print(f"[INFO] Model Name: {model_name}")

model = SwinUnet3D(hidden_dimension = config.hidden_dimension, layers = config.layers, heads = config.heads,
                    downscaling_factors = config.downscaling_factors, window_size = config.window_size, num_classes = config.num_classes, dropout = config.dropout, input_channel = config.num_channels,
                    head_dimension = config.head_dimension, relative_pos_embedding = config.relative_pos_embedding, 
                    skip_style = config.skip_style, second_to_last_channels = config.second_to_last_channels).to(config.device)

model = DataParallel(model)

# previous (results in overfitting, validation loss keep increasing, but did not stop overfitting)
# def calculate_class_weights(batch, num_classes = config.num_classes):
#     class_weights = torch.zeros(num_classes)
#     total_samples = batch.numel()
#     for class_idx in range(num_classes):
#         class_samples = (batch == class_idx).sum().item()  
#         if class_samples == 0:
#             class_weights[class_idx] = total_samples / (num_classes * 10)
#         else:
#             class_weights[class_idx] = total_samples / (num_classes * class_samples)
#     class_weights = class_weights / torch.sum(class_weights)
#     class_weights = torch.FloatTensor(class_weights).to(config.device)
    
#     return class_weights

#new
def calculate_class_weights(batch, num_classes = config.num_classes):
    num_samples = [(batch == i).sum().item() for i in range(num_classes)]
    total_samples = sum(num_samples)
    
    normed_weights = [1 - (x / total_samples) + 1e-5 for x in num_samples]
    balance_weights = torch.FloatTensor(normed_weights).to(config.device)
    
    return balance_weights




# initialize loss function and optimizer
optimizer = Adam(model.parameters(), lr=config.learning_rate)

# Checkpoint loading and resuming functionality
start_epoch = 0
if config.model_checkpoint and config.model_checkpoint.strip() != "":
    if os.path.exists(config.model_checkpoint):
        print(f"[INFO] Loading checkpoint from: {config.model_checkpoint}")
        try:
            checkpoint = torch.load(config.model_checkpoint, map_location=config.device)
            
            # Check if this is the new comprehensive checkpoint format or old model-only format
            if 'model_state_dict' in checkpoint:
                # New format - comprehensive checkpoint
                model.load_state_dict(checkpoint['model_state_dict'])
                print("[INFO] Model state dict loaded successfully (new format)")
                
                # Load optimizer state dict if available
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("[INFO] Optimizer state dict loaded successfully")
                
                # Load training history if available
                if 'train_loss' in checkpoint and 'val_loss' in checkpoint and 'epochs' in checkpoint:
                    H = {
                        "train_loss": checkpoint['train_loss'],
                        "val_loss": checkpoint['val_loss'], 
                        "epochs": checkpoint['epochs']
                    }
                    start_epoch = len(H['epochs'])
                    print(f"[INFO] Training history loaded. Resuming from epoch {start_epoch + 1}")
                else:
                    print("[INFO] No training history found in checkpoint. Starting fresh training history.")
                
                # Load best validation loss if available
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                    print(f"[INFO] Best validation loss loaded: {best_val_loss:.4f}")
                
                print(f"[INFO] Successfully loaded comprehensive checkpoint. Resuming training from epoch {start_epoch + 1}")
                
            else:
                # Old format - just model state dict
                model.load_state_dict(checkpoint)
                print("[INFO] Model state dict loaded successfully (old format)")
                print("[WARNING] This is an old-format checkpoint that only contains model weights.")
                print("[WARNING] No training history, optimizer state, or other training information is available.")
                print("[WARNING] Training will start from epoch 1 with fresh optimizer state and training history.")
                print("[INFO] Consider retraining to create a new comprehensive checkpoint for future resuming.")
            
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            print("[INFO] Starting training from scratch...")
            start_epoch = 0
    else:
        print(f"[WARNING] Checkpoint file not found: {config.model_checkpoint}")
        print("[INFO] Starting training from scratch...")
        start_epoch = 0
else:
    print("[INFO] No checkpoint provided. Starting training from scratch...")
    start_epoch = 0


# calculate steps per epoch for training and test set
train_steps = len(train_ds) // config.batch_size
val_steps = len(val_ds) // config.batch_size
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"[INFO] Number of Training Steps : {train_steps}")
print(f"[INFO] Number of Validation Steps : {val_steps}")
#print(f"[INFO] Total Number of Parameters : {total_params}")

# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss": [], "epochs": []}
best_val_loss = float("inf")

if config.logging:
    # start a new wandb run to track this script
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="CryoETPick", name = model_name + " Date: " + str(datetime.today()),
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": config.learning_rate,
        "architecture": model_name,
        "dataset": "shrec_19_20_21_AND_CryoPortal",
        "epochs": config.num_epochs,
        }
    )


# loop over epochs
print(f"[INFO] Training the network from epoch {start_epoch + 1} to {config.num_epochs}...")
start_time = time.time()
for e in tqdm(range(start_epoch, config.num_epochs)):
    model.train()
    
    train_loss = 0
    # loop over the training set

    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(config.device), y.to(config.device)

        optimizer.zero_grad()
        
        pred = model(x)
      
        class_weights = calculate_class_weights(y)
        loss = combined_loss(pred, y, class_weights = class_weights, weighted = config.weighted, loss_function = config.loss_function, loss_activation = config.loss_activation, focal_alpha = config.focal_alpha, focal_gamma = config.focal_gamma)
        loss.backward()
        #this is for the gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip_grad_norm)
        
        optimizer.step()
        
        
        train_loss += loss.item() 
        
    # Calculate train loss
    train_loss /= len(train_loader)
    
    val_loss = 0    
    
    model.eval()
    with torch.no_grad(): 
        for i, data in enumerate(val_loader):
            x, y = data
            x, y = x.to(config.device), y.to(config.device)
            
            pred = model(x)    
            
            class_weights = calculate_class_weights(y)
            # loss = combined_loss(pred, y, class_weights = class_weights, weighted = config.weighted, loss_function = config.loss_function, loss_activation = config.loss_activation)
            loss = combined_loss(pred, y, class_weights = class_weights, weighted = config.weighted, loss_function = config.loss_function, loss_activation = config.loss_activation)

            # Accumulate the validation loss
            val_loss += loss.item() * 1

    # Calculate validation loss
    val_loss /= len(val_loader)
    
    # update our training history
    H["train_loss"].append(train_loss)
    H["val_loss"].append(val_loss)
    H["epochs"].append(e + 1)
    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.num_epochs))
    print("Train Loss: {:.4f}, Validation Loss: {:.4f}".format(train_loss, val_loss, ))
    
    if config.logging:
        wandb.log({"train_loss": np.round(train_loss, 4), "val_loss": np.round(val_loss, 4)})
        
    
    # Save checkpoint with all necessary information for resuming
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': H["train_loss"],
        'val_loss': H["val_loss"],
        'epochs': H["epochs"],
        'best_val_loss': best_val_loss,
        'epoch': e + 1,
        'config': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'architecture_name': config.architecture_name,
            # Architecture parameters
            'hidden_dimension': config.hidden_dimension,
            'layers': config.layers,
            'heads': config.heads,
            'downscaling_factors': config.downscaling_factors,
            'window_size': config.window_size,
            'num_classes': config.num_classes,
            'dropout': config.dropout,
            'input_channel': config.num_channels,
            'head_dimension': config.head_dimension,
            'relative_pos_embedding': config.relative_pos_embedding,
            'skip_style': config.skip_style,
            'second_to_last_channels': config.second_to_last_channels,
            'prediction_type': config.prediction_type,
            'grid_type': config.grid_type,
            'mask_type': config.mask_type,
            'normalized_NY': config.normalized_NY,
            'loss_activation': config.loss_activation,
            'weighted': config.weighted,
            'loss_function': config.loss_function,
            'clip_grad_norm': config.clip_grad_norm
        }
    }
    
    # Save regular checkpoint
    torch.save(checkpoint_data, os.path.join(f"{config.output_path}/models/", f"{model_name}"))
    
    # Save best model if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint_data, os.path.join(f"{config.output_path}/models/", f"early_stop_{model_name}"))
        print(f"[INFO] New best model saved with validation loss: {val_loss:.4f}")

# display the total time needed to perform the training
end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    end_time - start_time))


