# Author: Ashwin Dhakal
import torch.nn as nn
import torch
import numpy as np
from typing import Union, List
from models.encoder_decoder import Encoder, Decoder
from models.blocks import Converge, FinalExpand3D
from timm.models.layers import trunc_normal_
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)



class SwinUnet3D(nn.Module):
    def __init__(
            self, 
            *, 
            hidden_dimension=96, 
            layers=(2, 2, 6, 2), 
            heads=(3, 6, 9, 12), 
            input_channel=1, 
            num_classes=2, 
            head_dimension=32,
            window_size: Union[int, List[int]] = 7, 
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True, 
            dropout: float = 0.0, 
            skip_style='add', # can be stack
            second_to_last_channels: int = 32
            ):
        super().__init__()

        # Initialize parameters
        self.downscaling_factors = downscaling_factors
        self.window_size = window_size

        # Define encoder and decoder blocks
        self.encoder12 = Encoder(
            input_dimension = input_channel,
            hidden_dimension = hidden_dimension,
            layers = layers[0],
            downscaling_factor = downscaling_factors[0],
            num_heads = heads[0],
            head_dimension = head_dimension,
            window_size = window_size,
            dropout = dropout,
            relative_pos_embedding = relative_pos_embedding            
        )
        
        self.encoder3 = Encoder(
            input_dimension = hidden_dimension,
            hidden_dimension = hidden_dimension * 2,
            layers = layers[1],
            downscaling_factor = downscaling_factors[1],
            num_heads = heads[1],
            head_dimension = head_dimension,
            window_size = window_size,
            dropout = dropout,
            relative_pos_embedding = relative_pos_embedding            
        )
        
        self.encoder4 = Encoder(
            input_dimension = hidden_dimension * 2,
            hidden_dimension = hidden_dimension * 4,
            layers = layers[2],
            downscaling_factor = downscaling_factors[2],
            num_heads = heads[2],
            head_dimension = head_dimension,
            window_size = window_size,
            dropout = dropout,
            relative_pos_embedding = relative_pos_embedding            
        )
        
        self.bottleneck = Encoder(
            input_dimension = hidden_dimension * 4,
            hidden_dimension = hidden_dimension * 8,
            layers = layers[3],
            downscaling_factor = downscaling_factors[3],
            num_heads = heads[3],
            head_dimension = head_dimension,
            window_size = window_size,
            dropout = dropout,
            relative_pos_embedding = relative_pos_embedding            
        )
        
        self.decoder4 = Decoder(
            input_dimension = hidden_dimension * 8,
            output_dimension = hidden_dimension * 4,
            layers = layers[2],
            upscaling_factor = downscaling_factors[3],
            num_heads = heads[2],
            head_dimension = head_dimension,
            window_size = window_size,
            dropout = dropout,
            relative_pos_embedding = relative_pos_embedding
        )
        
        self.decoder3 = Decoder(
            input_dimension = hidden_dimension * 4,
            output_dimension = hidden_dimension * 2,
            layers = layers[1],
            upscaling_factor = downscaling_factors[2],
            num_heads = heads[1],
            head_dimension = head_dimension,
            window_size = window_size,
            dropout = dropout,
            relative_pos_embedding = relative_pos_embedding
        )
        
        self.decoder21 = Decoder(
            input_dimension = hidden_dimension * 2,
            output_dimension = hidden_dimension,
            layers = layers[0],
            upscaling_factor = downscaling_factors[1],
            num_heads = heads[0],
            head_dimension = head_dimension,
            window_size = window_size,
            dropout = dropout,
            relative_pos_embedding = relative_pos_embedding
        )


        # Define convergence and final blocks
        self.converge4 = Converge(hidden_dimension * 4, skip_style)
        self.converge3 = Converge(hidden_dimension * 2, skip_style)
        self.converge12 = Converge(hidden_dimension, skip_style)

        self.final = FinalExpand3D(
            input_dimension = hidden_dimension, 
            output_dimension = second_to_last_channels,
            upscaling_factor = downscaling_factors[0]
            )
        
        self.out = nn.Sequential(
            nn.Conv3d(second_to_last_channels, num_classes, kernel_size=1)
            )
        
        # Initialize parameters
        self.init_weight()

    def forward(self, img):
        window_size = self.window_size
        assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
        
        # Check window size
        if type(window_size) is int:
            window_size = np.array([window_size, window_size, window_size])
        _, _, x_s, y_s, z_s = img.shape
        x_ws, y_ws, z_ws = window_size

        # # Check if dimensions are divisible by the window size
        # assert x_s % (x_ws * 32) == 0, f'x-axis size must be divisible by x_window_size*32'
        # assert y_s % (y_ws * 32) == 0, f'y-axis size must be divisible by y_window_size*32'
        # assert z_s % (z_ws * 32) == 0, f'z-axis size must be divisible by z_window_size*32'

        # Forward pass through the network
        down12 = self.encoder12(img)
        down3 = self.encoder3(down12)
        down4 = self.encoder4(down3)
        
        features = self.bottleneck(down4)
        # print("Features", features.shape)
        
        up4 = self.decoder4(features)
        up4 = self.converge4(up4, down4)
        
        up3 = self.decoder3(up4)
        up3 = self.converge3(up3, down3)
        
        up21 = self.decoder21(up3)
        up21 = self.converge12(up21, down12)
        
        out = self.final(up21)
        out = self.out(out)
        
        return out

    def init_weight(self):
        # Add weight initialization logic if needed
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std = 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

