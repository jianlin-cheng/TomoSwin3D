# Author: Ashwin Dhakal
import torch
import torch.nn as nn
from torch import einsum
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Union, List

class CyclicShift3D(nn.Module):
    def __init__(self, displacement):
        super().__init__()

        assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
        if type(displacement) is int:
            displacement = np.array([displacement, displacement, displacement])
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement[0], self.displacement[1], self.displacement[2]), dims=(1, 2, 3))
    
class Residual3D(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm3D(nn.Module):
    def __init__(self, dimension, fn, norm_type = 'prenorm'):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.fn = fn
        self.norm_type = norm_type

    def forward(self, x, **kwargs):
        if self.norm_type == 'prenorm':
            return self.fn(self.norm(x), **kwargs)
        else:
            return self.norm(self.fn(x), **kwargs)


class FeedForward3D(nn.Module):
    def __init__(self, dimension, hidden_dimension, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, dimension),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.net(x)
        x = self.drop(x)
        return x


class Norm(nn.Module):
    def __init__(self, dimension, channel_first: bool = True):
        super(Norm, self).__init__()
        if channel_first:
            self.net = nn.Sequential(
                Rearrange('b c h w d -> b h w d c'),
                nn.LayerNorm(dimension),
                Rearrange('b h w d c -> b c h w d')
            )
        else:
            self.net = nn.LayerNorm(dimension)

    def forward(self, x):
        x = self.net(x)
        return x

class Converge(nn.Module):
    def __init__(self, dimension: int, converge_style = 'add'):
        '''
        Converge module to combine features using either stack or add operation.

        Args:
        - dimension (int): Dimension of the input features.
        - converge_style (str): The fusion style. Options are 'add' for element-wise addition or 'stack' for stacking.
        '''
        super().__init__()
        self.converge_style = converge_style
        self.norm = Norm(dimension = dimension)

    def forward(self, x, encoded_x):
        '''
        Forward pass of the Converge module.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        - encoded_x (torch.Tensor): Encoded tensor of shape (B, C, H, W, D).

        Returns:
        - torch.Tensor: Output tensor after convergence.
        '''
        assert x.shape == encoded_x.shape, "Input shapes must match."
        
        if self.converge_style == 'add':
            # Element-wise addition
            x = x + encoded_x
        elif self.converge_style == 'stack':
            # Stack and linear transform
            x = torch.cat([x, encoded_x], dimension=1)
            x = self.linear_transform(x)
        
        # Normalization
        x = self.norm(x)
        
        return x
    
def create_mask3D(window_size: Union[int, List[int]], displacement: Union[int, List[int]],
                  x_shift: bool, y_shift: bool, z_shift: bool):
    # Ensure window_size and displacement are valid inputs
    assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
    if type(window_size) is int:
        window_size = np.array([window_size, window_size, window_size])

    assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
    if type(displacement) is int:
        displacement = np.array([displacement, displacement, displacement])

    # Check that each displacement is within the bounds of the corresponding window size dimension
    assert len(window_size) == len(displacement)
    for i in range(len(window_size)):
        assert 0 < displacement[i] < window_size[i], \
            f'Displacement along dimension {i} is incorrect. Dimensions include X (i=0), Y (i=1), and Z (i=2)'

    # Initialize a mask tensor with shape (wx*wy*wz, wx*wy*wz)
    mask = torch.zeros(window_size[0] * window_size[1] * window_size[2],
                       window_size[0] * window_size[1] * window_size[2])
    mask = rearrange(mask, '(x1 y1 z1) (x2 y2 z2) -> x1 y1 z1 x2 y2 z2',
                     x1=window_size[0], y1=window_size[1], x2=window_size[0], y2=window_size[1])

    # Apply padding to the mask based on specified shifts
    x_dist, y_dist, z_dist = displacement[0], displacement[1], displacement[2]

    if x_shift:
        # Apply padding in the x-axis
        mask[-x_dist:, :, :, :-x_dist, :, :] = float('-inf')
        mask[:-x_dist, :, :, -x_dist:, :, :] = float('-inf')

    if y_shift:
        # Apply padding in the y-axis
        mask[:, -y_dist:, :, :, :-y_dist, :] = float('-inf')
        mask[:, :-y_dist, :, :, -y_dist:, :] = float('-inf')

    if z_shift:
        # Apply padding in the z-axis
        mask[:, :, -z_dist:, :, :, :-z_dist] = float('-inf')
        mask[:, :, :-z_dist, :, :, -z_dist:] = float('-inf')

    # Rearrange the mask tensor to the final shape (x1 y1 z1 x2 y2 z2)
    mask = rearrange(mask, 'x1 y1 z1 x2 y2 z2 -> (x1 y1 z1) (x2 y2 z2)')
    return mask


def get_relative_distances(window_size):
    assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
    if type(window_size) is int:
        window_size = np.array([window_size, window_size, window_size])
    indices = torch.tensor(
        np.array(
            [[x, y, z] for x in range(window_size[0]) for y in range(window_size[1]) for z in range(window_size[2])]))

    distances = indices[None, :, :] - indices[:, None, :]
    # distance:(n,n,3) n =window_size[0]*window_size[1]*window_size[2]
    return distances
 
class WindowAttention3D(nn.Module):
    def __init__(self, dimension: int, heads: int, head_dimension: int, shifted: bool, window_size: Union[int, List[int]],
                 relative_pos_embedding: bool = True):
        super().__init__()

        # Validate input dimensions
        assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimensions'
        if type(window_size) is int:
            window_size = np.array([window_size, window_size, window_size])
        else:
            window_size = np.array(window_size)

        # Define internal parameters
        inner_dimension = head_dimension * heads
        self.heads = heads
        self.scale = head_dimension ** -0.5
        self.window_size = window_size
        self.shifted = shifted
        self.relative_pos_embedding = relative_pos_embedding

        # Initialize shifted attention parameters
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift3D(-displacement)
            self.cyclic_back_shift = CyclicShift3D(displacement)
            self.x_mask = nn.Parameter(create_mask3D(window_size = window_size, displacement = displacement,
                                                     x_shift = True, y_shift = False, z_shift = False), requires_grad = False)
            self.y_mask = nn.Parameter(create_mask3D(window_size = window_size, displacement = displacement,
                                                     x_shift = False, y_shift = True, z_shift = False), requires_grad = False)
            self.z_mask = nn.Parameter(create_mask3D(window_size = window_size, displacement = displacement,
                                                     x_shift = False, y_shift = False, z_shift = True), requires_grad = False)

        # Linear transformation for Q, K, and V
        self.to_qkv = nn.Linear(dimension, inner_dimension * 3, bias = False)

        # Initialize relative position embedding
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size)
            for i in range(len(window_size)):
                self.relative_indices[:, :, i] += window_size[i] - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size[0] - 1, 2 * window_size[1] - 1, 2 * window_size[2] - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size[0] * window_size[1] * window_size[2],
                                                          window_size[0] * window_size[1] * window_size[2]))

        # Softmax activation and linear transformation for output
        self.softmax = nn.Softmax(dim = -1)
        self.to_out = nn.Linear(inner_dimension, dimension)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_x, n_y, n_z, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        nw_x = n_x // self.window_size[0]
        nw_y = n_y // self.window_size[1]
        nw_z = n_z // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d) -> b h (nw_x nw_y nw_z) (w_x w_y w_z) d',
                                h = h, w_x = self.window_size[0], w_y = self.window_size[1], w_z = self.window_size[2]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0].long(),
                                       self.relative_indices[:, :, 1].long(),
                                       self.relative_indices[:, :, 2].long()]
        else:
          dots += self.pos_embedding

        if self.shifted:
            dots = rearrange(dots, 'b h (n_x n_y n_z) i j -> b h n_y n_z n_x i j',
                             n_x = nw_x, n_y = nw_y)
            dots[:, :, :, :, -1] += self.x_mask

            dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h n_x n_z n_y i j')
            dots[:, :, :, :, -1] += self.y_mask

            dots = rearrange(dots, 'b h n_x n_z n_y i j -> b h n_x n_y n_z i j')
            dots[:, :, :, :, -1] += self.z_mask

            dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h (n_x n_y n_z) i j')

        attn = self.softmax(dots)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (nw_x nw_y nw_z) (w_x w_y w_z) d -> b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d)',
                        h = h, w_x = self.window_size[0], w_y = self.window_size[1], w_z = self.window_size[2],
                        nw_x = nw_x, nw_y = nw_y, nw_z = nw_z)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


     
class PatchMerging3D(nn.Module):
    def __init__(
            self, 
            input_dimension, 
            output_dimension, 
            downscaling_factor
        ):
        super().__init__()
        self.net = nn.Sequential(
                    Rearrange('b c (n_h dsf_h) (n_w dsf_w) (n_d dsf_d) -> b n_h n_w n_d (dsf_h dsf_w dsf_d c)',
                            dsf_h = downscaling_factor, dsf_w = downscaling_factor, dsf_d = downscaling_factor),
                    nn.Linear(input_dimension * (downscaling_factor ** 3), output_dimension),
                    Norm(output_dimension, channel_first=False),
                    Rearrange('b h w d c -> b c h w d')
                )

    def forward(self, x):
        '''X: B, C, H, W, D'''
        return self.net(x)
        
    
class PatchExpand3D(nn.Module):
    def __init__(
            self, 
            input_dimension, 
            output_dimension, 
            upscaling_factor
        ):
        super().__init__()

        hidden_dimension = (upscaling_factor ** 3) * output_dimension
        self.net = nn.Sequential(
                    Rearrange('b c h w d -> b h w d c'),
                    nn.Linear(input_dimension, hidden_dimension),
                    Rearrange('b h_s w_s d_s (usf1 usf2 usf3 c) -> b c (h_s usf1) (w_s usf2) (d_s usf3)',
                            usf1 = upscaling_factor, usf2 = upscaling_factor, usf3 = upscaling_factor),
                    Norm(output_dimension),
                )

    def forward(self, x):
        '''X: B, C, H, W, D'''
        
        return self.net(x)
    
class FinalExpand3D(nn.Module):
    def __init__(
            self, 
            input_dimension, 
            output_dimension, 
            upscaling_factor
        ):
        super().__init__()

        # Calculate the hidden dimension after the linear transformation
        hidden_dimension = (upscaling_factor ** 3) * output_dimension

        self.net = nn.Sequential(
                    Rearrange('b c h w d -> b h w d c'),
                    nn.Linear(input_dimension, hidden_dimension),
                    Rearrange('b h_s w_s d_s (usf1 usf2 usf3 c) -> b c (h_s usf1) (w_s usf2) (d_s usf3)',
                            usf1 = upscaling_factor, usf2 = upscaling_factor, usf3 = upscaling_factor),
                    Norm(output_dimension),
                    nn.PReLU()
                )

    def forward(self, x):
        '''X: B, C, H, W, D'''
        x = self.net(x)
        return x
    
class SwinBlock3D(nn.Module):
    def __init__(
            self, 
            dimension, 
            heads, 
            head_dimension, 
            mlp_dimension, 
            window_size: Union[int, List[int]],            
            shifted: bool = False, 
            relative_pos_embedding: bool = True, 
            dropout: float = 0.0
        ):
        super().__init__()

        # Attention Block
        self.attention_block = Residual3D(
                                        PreNorm3D(
                                                dimension, 
                                                WindowAttention3D(
                                                    dimension = dimension,
                                                    heads = heads,
                                                    head_dimension = head_dimension,
                                                    shifted = shifted,
                                                    window_size = window_size,
                                                    relative_pos_embedding = relative_pos_embedding
                                                )
                                        )
                                )

        # MLP Block
        self.mlp_block = Residual3D(
                                PreNorm3D(
                                        dimension, 
                                        FeedForward3D(
                                                dimension = dimension, 
                                                hidden_dimension = mlp_dimension, 
                                                dropout = dropout
                                        )
                                )
                        )

    def forward(self, x):
        # Attention Block
        x = self.attention_block(x)

        # MLP Block
        x = self.mlp_block(x)

        return x




    
