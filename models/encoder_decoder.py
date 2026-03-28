# Author: Ashwin Dhakal
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from typing import Union, List
from models.blocks import PatchMerging3D, PatchExpand3D, SwinBlock3D

class Encoder(nn.Module):
    def __init__(
            self, 
            input_dimension, 
            hidden_dimension, 
            layers, 
            downscaling_factor, 
            num_heads, 
            head_dimension,
            window_size: Union[int, List[int]], 
            relative_pos_embedding: bool = True, 
            dropout: float = 0.0
            ):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        # Patch partitioning using PatchMerging3D
        self.patch_partition = PatchMerging3D(
                                    input_dimension = input_dimension, 
                                    output_dimension = hidden_dimension,
                                    downscaling_factor = downscaling_factor
                                )

        # Rearrange dimensions for processing in SwinBlocks
        self.rearrange1 = Rearrange('b c h w d -> b h w d c')

        # Swin Transformer layers
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            # Regular Swin Block
            regular_block = SwinBlock3D(
                                dimension = hidden_dimension,
                                heads = num_heads,
                                head_dimension = head_dimension,
                                mlp_dimension = hidden_dimension * 4,
                                window_size = window_size,
                                shifted = False,
                                relative_pos_embedding = relative_pos_embedding,
                                dropout = dropout
                            )
                       
            # Shifted Swin Block
            shifted_block = SwinBlock3D(
                                dimension = hidden_dimension,
                                heads = num_heads,
                                head_dimension = head_dimension,
                                mlp_dimension = hidden_dimension * 4,
                                window_size = window_size,
                                shifted = True,
                                relative_pos_embedding = relative_pos_embedding,
                                dropout = dropout
                            )
            
            self.swin_layers.append(nn.ModuleList([regular_block, shifted_block]))

        # Rearrange dimensions back after processing in SwinBlocks
        self.rearrange2 = Rearrange('b h w d c -> b c h w d')

    def forward(self, x):
        x = self.patch_partition(x)
        x = self.rearrange1(x)
        for regular_block, shifted_block in self.swin_layers:
            x = regular_block(x)
            x = shifted_block(x)
        x = self.rearrange2(x)
        
        return x
    
    
class Decoder(nn.Module):
    def __init__(
            self, 
            input_dimension, 
            output_dimension, 
            layers, 
            upscaling_factor, 
            num_heads, 
            head_dimension,
            window_size: Union[int, List[int]], 
            relative_pos_embedding: bool = True, 
            dropout: float = 0.0
            ):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_expand = PatchExpand3D(
                                input_dimension = input_dimension,
                                output_dimension = output_dimension,
                                upscaling_factor = upscaling_factor
                            )

        self.rearrange1 = Rearrange('b c h w d -> b h w d c')
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            # Regular Swin Block
            regular_block = SwinBlock3D(
                                dimension = output_dimension,
                                heads = num_heads,
                                head_dimension = head_dimension,
                                mlp_dimension = output_dimension * 4,
                                window_size = window_size,
                                shifted = False,
                                relative_pos_embedding = relative_pos_embedding,
                                dropout = dropout
                            )
                       
            # Shifted Swin Block
            shifted_block = SwinBlock3D(
                                dimension = output_dimension,
                                heads = num_heads,
                                head_dimension = head_dimension,
                                mlp_dimension = output_dimension * 4,
                                window_size = window_size,
                                shifted = True,
                                relative_pos_embedding = relative_pos_embedding,
                                dropout = dropout
                            )
            
            self.swin_layers.append(nn.ModuleList([regular_block, shifted_block]))
            
        self.rearrange2 = Rearrange('b h w d c -> b c h w d')

    def forward(self, x):
        x = self.patch_expand(x)
        x = self.rearrange1(x)
        for regular_block, shifted_block in self.swin_layers:
            x = regular_block(x)
            x = shifted_block(x)
        x = self.rearrange2(x)
        
        return x