# Configuration file
## Author: Ashwin Dhakal

import argparse
import torch
from datetime import datetime
import os

# Create an argument parser
parser = argparse.ArgumentParser(description="CryoETPick Training")
# Add arguments
parser.add_argument("--grid_size", type=int, default=64, help="Grid Size")
parser.add_argument("--normalized_NY", type=str, default='normalized', help="Training Data Type (options: 'normalized' or 'non_normalized')")
parser.add_argument("--window_size", type=int, default=2, help="Window Size")
parser.add_argument("--hidden_dimension", type=int, default=32, help="Hidden Dimension") #working on this; ablation
parser.add_argument("--layers", default=(2, 6, 6, 2), help="Number of Layers")
parser.add_argument("--heads", default=(3, 6, 12, 24), help="Number of Heads")
parser.add_argument("--head_dimension", type=int, default=32, help="Head dimension for attention")
parser.add_argument("--relative_pos_embedding", type=bool, default=True, help="Use relative position embedding")
parser.add_argument("--skip_style", type=str, default='add', help="Skip connection style (add or stack)")
parser.add_argument("--second_to_last_channels", type=int, default=32, help="Second to last layer channels")
parser.add_argument("--weighted", type=bool, default=True, help="Weighted Loss") 
parser.add_argument("--num_channels", type=int, default=4, help="Number of input channels")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--dropout", default=0.5, help="Dropout Rate")
parser.add_argument("--downscaling_factors", default=(2, 2, 2, 2), help="Downscaling Factors")
parser.add_argument("--loss_function", type=str, default='CE', help="Loss Function (options: 'Dice', 'GeneralizedDice', 'CE', 'Focal', 'DiceCE')")  #CE: Uses softmax activation BCE: Uses sigmoid activation
parser.add_argument("--loss_activation", type=str, default='__', help="Loss Activation Function (options: 'softmax' or 'sigmoid')")
parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal Loss alpha parameter (balancing factor)")
parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma parameter (focusing parameter, 0-5). Higher=more focus on hard examples")
parser.add_argument("--mask_type", type=str, default='class_mask', help="Prediction Type (options: 'center' OR spherical6 OR occupancy OR class_mask)") 
parser.add_argument("--prediction_type", type=str, default='multiclass_standardized_across_shrec_2020', \
    help="Prediction Type (options: 'binary', 'multiclass', 'multiclass_standardized_across_shrec2020_21_CryoETPortal_MaxPlanck', \
    'multiclass_standardized_across_shrec_2020', 'multiclass_standardized_across_shrec_2021', \
    'multiclass_standardized_across_CryoETPortal', 'multiclass_standardized_across_MaxPlanck')") 
parser.add_argument("--grid_type", type=str, default='non_zero_grid', help="Training Type (options: 'full_grid' or 'non_zero_grid')")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda:0 or cpu)") # if single : cuda:1 #if parallel :cuda
parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Maximum L2-norm for gradients; set to 0 to disable")

if parser.parse_args().prediction_type == 'binary':
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
elif parser.parse_args().prediction_type == 'multiclass_standardized_across_shrec_2020_21_CryoETPortal_MaxPlanck':
    parser.add_argument("--num_classes", type=int, default=130, help="Number of classes")

elif parser.parse_args().prediction_type == 'multiclass_standardized_across_shrec_2020':
    parser.add_argument("--num_classes", type=int, default=13, help="Number of classes")

elif parser.parse_args().prediction_type == 'multiclass_standardized_across_shrec_2021':
    parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")

elif parser.parse_args().prediction_type == 'multiclass_standardized_across_CryoETPortal':
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")

elif parser.parse_args().prediction_type == 'multiclass_standardized_across_MaxPlanck':
    parser.add_argument("--num_classes", type=int, default=110, help="Number of classes")

else:
    print("Error: number of class invalid")

parser.add_argument("--train_dataset_path", type=str, default=f"/cluster/pixstor/chengji-lab/ashwin/CryoET/train_test_dataset/Training_on_shrec_20_21_CryoPortal_MaxPlanck_wo_augment/Grids_{parser.parse_args().grid_size}_{parser.parse_args().normalized_NY}/{parser.parse_args().mask_type}_{parser.parse_args().prediction_type}_nonzero_grid_masks/*/", help="Path to the training dataset")


# Logging-related arguments
parser.add_argument("--logging", type=bool, default=True, help="Enable logging")


# Data related arguments
parser.add_argument("--my_dataset_path", type=str, default="my_dataset", help="Path to your own dataset")
parser.add_argument("--output_path", type=str, default="output", help="Output directory")
# parser.add_argument("--model_checkpoint", type=str, default="", help="Path to CryoETPick checkpoint (empty string for training from scratch). 
parser.add_argument("--model_checkpoint", type=str, default="", help="Path to CryoETPick checkpoint (empty string for training from scratch") 

# Device-related arguments
parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory for data loading if using CUDA")
parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of data loading workers")

current_datetime = datetime.now()
timestamp = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")


# Additional info in architecture name - Parameters ordered by importance (High to Low Impact)
architecture_name = "CryoETPick_{}_HiD{}_LR{}_L{}_B{}_D{}_WS{}_L{}_He{}_HeD{}_R{}_S{}_SL{}_W{}_C{}_Gr{}_Ch{}_M{}_P{}_GT{}_N{}_A{}_C{}".format(
    timestamp,
    parser.parse_args().hidden_dimension,
    parser.parse_args().learning_rate,
    parser.parse_args().loss_function,
    parser.parse_args().batch_size,
    parser.parse_args().dropout,
    parser.parse_args().window_size,
    "_".join(map(str, parser.parse_args().layers)),  # Convert tuple to string
    "_".join(map(str, parser.parse_args().heads)),   # Convert tuple to string
    parser.parse_args().head_dimension,
    int(parser.parse_args().relative_pos_embedding),  # Convert bool to int
    parser.parse_args().skip_style,
    parser.parse_args().second_to_last_channels,
    int(parser.parse_args().weighted),  # Convert bool to int
    parser.parse_args().clip_grad_norm,
    parser.parse_args().grid_size,
    parser.parse_args().num_channels,
    parser.parse_args().mask_type,
    parser.parse_args().prediction_type,
    parser.parse_args().grid_type,
    parser.parse_args().normalized_NY,
    parser.parse_args().loss_activation,
    parser.parse_args().num_classes
)
parser.add_argument("--architecture_name", type=str, default=architecture_name, help="Model architecture name")

# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
grid_size = args.grid_size
normalized_NY = args.normalized_NY
window_size = args.window_size
hidden_dimension = args.hidden_dimension
layers = args.layers
heads = args.heads
head_dimension = args.head_dimension
relative_pos_embedding = args.relative_pos_embedding
skip_style = args.skip_style
second_to_last_channels = args.second_to_last_channels
downscaling_factors = args.downscaling_factors
loss_function = args.loss_function
loss_activation = args.loss_activation
focal_alpha = args.focal_alpha
focal_gamma = args.focal_gamma
mask_type = args.mask_type
prediction_type = args.prediction_type
grid_type = args.grid_type
weighted = args.weighted
num_channels = args.num_channels
num_classes = args.num_classes
learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size
clip_grad_norm = args.clip_grad_norm
dropout = args.dropout
logging = args.logging

train_dataset_path = args.train_dataset_path
# test_dataset_path = args.test_dataset_path
my_dataset_path = args.my_dataset_path
output_path = args.output_path
model_checkpoint = args.model_checkpoint

device = args.device
pin_memory = args.pin_memory
num_workers = args.num_workers


architecture_name = args.architecture_name