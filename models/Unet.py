from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_attention import SpatialTransformer

import pdb

class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups..
    """
    return GroupNorm32(32, channels)

class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )
        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)
            
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Time step embeddings
        # pdb.set_trace()
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        # Add time step embeddings
        h = h + t_emb[:, :, None, None]
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h
            
class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules such as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                # pdb.set_trace()
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x
    
class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)
    
class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
# class Unit(nn.Module):
    
#     def __init__(
#         self, *,
#         channels: int,
#         n_res_blocks: int,
#         attn: bool = False,
#         n_heads: int,
#         tf_layers: int = 1,
#         d_cond: int = 768,        
#     ):
#         '''
#         channels: number of channels in this unit
#         n_res_blocks: number of residual blocks at each level
#         n_heads: number of attention heads in the transformer
#         attn: whether this unet has attention
#         tf_layer: number of transformer layers in the transformers
#         d_cond: size of the conditional embedding in the transformers
#         '''
#         super().__init__()
        
#         d_time_emb = channels * 4
        
#         self.net = nn.ModuleList()
        
#         for _ in range(n_res_blocks):
#             layers = [ResBlock(channels, d_time_emb, out_channels=channels)]
            
#             if attn:
#                 layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
            
#             self.net.append(TimestepEmbedSequential(*layers))
            
class UnetA128(nn.Module):
    def __init__(self, *,
                 in_channels: int,
                 out_channels: int,
                 channels: int,
                #  n_res_blocks: List[int],
                #  attention_levels: List[int],
                 if_down: bool = False,
                #  channels_multipliers: List[int],
                 n_heads:int,
                 tf_layers: int = 1,
                 d_cond: int = 768):
        '''
        in_channels: number of channels in the input feature map
        out_channels: number of channels in the output feature map
        channels: base channel count for the model
        n_res_blocks: number of residual blocks at each level
        attention_levels: levels at which attention be performed
        if_down: if additional input image which is the downsample from the groundtruth, (e.g. 256Parallel-unet)
        channel_multipliers: multiplicative factors for numebr of channels for each level
        n_heads: number of transformer layers in the transformers
        tf_layers: number of transformer layers in the transformers
        d_cond: size of the conditional embedding in the transformers
        '''
        
        super().__init__()
        
        self.channels_multipliers = [1, 2, 4, 4]
        self.attention_levels = [2, 3]
        n_res_blocks = 3
        
        levels = len(self.channels_multipliers)
        
        d_time_emb = channels * 4 * 3
        in_channels = in_channels * (1 + 1 + if_down) # in_channels default as 4
        
        # self.time_embed = nn.Sequential(
        #     nn.Linear(channels, d_time_emb),
        #     nn.SiLU(),
        #     nn.Linear(d_time_emb, d_time_emb),
        # )
        
        self.conv = nn.Conv2d(in_channels, channels, 3, padding=1)
        
        self.input_blocks = nn.ModuleList()     
               
        input_block_channels = [channels]
        
        channels_list = [channels * m for m in self.channels_multipliers]
        
        for i in range(levels):
            sub_net = nn.ModuleList()
            # sub_net = mySequential()

            for j in range(n_res_blocks):
                
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                
                if i in self.attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                    
                sub_net.append(TimestepEmbedSequential(*layers))
                # sub_net.add_module(str(j),TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            if i != levels - 1:
                sub_net.append(TimestepEmbedSequential(DownSample(channels)))
                # sub_net.add_module(str(i),TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)
            self.input_blocks.append(sub_net)
            
            
        # self.middle_block = TimestepEmbedSequential(
        #     ResBlock(channels, d_time_emb),
        #     SpatialTransformer(channels, n_heads, tf_layers, d_cond),
        #     ResBlock(channels, d_time_emb),
        # )
        
        self.output_blocks = nn.ModuleList()
        
        
        for i in reversed(range(levels)):
            sub_net = nn.ModuleList()
            # sub_net = mySequential()
            for j in range(n_res_blocks + 1):
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                
                if i in self.attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                    
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                    
                sub_net.append(TimestepEmbedSequential(*layers))
                # sub_net.add_module(str(j),TimestepEmbedSequential(*layers))
            self.output_blocks.append(sub_net)
                
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        
    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channels // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
    def forward(self, x: torch.Tensor, cat_t_embs: torch.Tensor, cond: List[torch.Tensor]):
        
        self.x_input_block = []
        
        # t_emb = self.time_step_embedding(time_steps)
        # t_emb = self.time_embed(t_emb)
        
        # input half
        x = self.conv(x)
        
        now_cond = None
        for i,module in enumerate(self.input_blocks):
            if i==2: now_cond = cond[0]
                # x = module(x,cat_t_embs,cond[0])
            if i==3: now_cond = cond[1]
                # x = module(x,cat_t_embs,cond[1])
            # else: x = module(x,cat_t_embs)
            else: now_cond = None
            
            if isinstance(module, nn.ModuleList):
                x = self._forward_nested(module, x, cat_t_embs, now_cond)
            else:
                x = module(x)            
            
            # for sub_module in module:
            #     x = sub_module(x, cat_t_embs, now_cond)
            
            self.x_input_block.append(x)
        
        # x = self.input_blocks[0](x,cat_t_embs)
        # x_input_block.append(x)
        
        # x = self.input_blocks[1](x,cat_t_embs)
        # x_input_block.append(x)
        
        # x = self.input_blocks[2](x,cat_t_embs,cond[0])
        # x_input_block.append(x)
        
        # x = self.input_blocks[3](x,cat_t_embs,cond[1])
        # x_input_block.append(x)
        
        # output half 
        now_cond = None
        for i,module in enumerate(self.output_blocks):
            # x = torch.cat([x, x_input_block.pop()], dim=1)
            if i==0: now_cond = cond[2]
                # x = module(x, cat_t_embs, cond[2])
            if i==1: now_cond = cond[3]
                # x = module(x, cat_t_embs, cond[3])
            else: now_cond = None
            
            if isinstance(module, nn.ModuleList):
                x = self._forward_nested_output(module, x)
            else:
                x = module(x)
            
            # for sub_module in module:                
            #     x = sub_module(x, cat_t_embs, now_cond)
            
        # x = torch.cat([x, x_input_block.pop()], dim=1)
        # x = self.output_blocks[0](x, cat_t_embs, cond[2])
        
        # x = torch.cat([x, x_input_block.pop()], dim=1)
        # x = self.output_blocks[1](x, cat_t_embs, cond[3])
        
        # x = torch.cat([x, x_input_block.pop()], dim=1)
        # x = self.output_blocks[2](x, cat_t_embs)
        
        # x = torch.cat([x, x_input_block.pop()], dim=1)
        # x = self.output_blocks[3](x, cat_t_embs)
        
        return  self.out(x)
    
    def _forward_nested(self,module_list,x,cat_t_embs,cond):
        # pdb.set_trace()
        for layer in module_list:
            # x = torch.cat([x, self.x_input_block.pop()], dim=1)
            x = layer(x, cat_t_embs, cond)
        return x
    
    def _forward_nested_output(self,module_list,x,cat_t_embs,cond):
        # pdb.set_trace()
        for layer in module_list:
            x = torch.cat([x, self.x_input_block.pop()], dim=1)
            x = layer(x, cat_t_embs, cond)
        return x

class UnetB128(nn.Module):
    def __init__(self, *,
                 in_channels: int,
                 out_channels: int, #unused
                 channels: int,
                #  n_res_blocks: int,
                #  attention_levels: List[int], 
                 if_down: bool = False,
                #  channels_multipliers: List[int],
                 n_heads:int,
                 tf_layers: int = 1,
                 d_cond: int = 768):
        '''
        in_channels: number of channels in the input feature map
        out_channels: number of channels in the output feature map
        channels: base channel count for the model
        n_res_blocks: number of residual blocks at each level
        attention_levels: levels at which attention be performed
        if_down: if additional input image which is the downsample from the groundtruth, (e.g. 256Parallel-unet)
        channel_multipliers: multiplicative factors for numebr of channels for each level
        n_heads: number of transformer layers in the transformers
        tf_layers: number of transformer layers in the transformers
        d_cond: size of the conditional embedding in the transformers
        '''
        
        super().__init__()
        self.channels_multipliers = [1, 2, 4, 4]
        levels = len(self.channels_multipliers)
        n_res_blocks = 3
        
        d_time_emb = channels * 4 * 3 # * 3 for cat_time_emb 
        in_channels = in_channels # in_channels default as 4, UNetB only input segmented garment
        
        # self.time_embed = nn.Sequential(
        #     nn.Linear(channels, d_time_emb),
        #     nn.SiLU(),
        #     nn.Linear(d_time_emb, d_time_emb),
        # )
        
        self.attention_levels = []
        
        self.conv = nn.Conv2d(in_channels, channels, 3, padding=1)
        
        self.input_blocks = nn.ModuleList()
        
        # self.input_blocks.append(TimestepEmbedSequential(
        #     nn.Conv2d(in_channels, channels, 3, padding=1))) # maps the input to channels
               
        input_block_channels = [channels]
        
        channels_list = [channels * m for m in self.channels_multipliers]
        
        for i in range(levels):
            sub_net = nn.ModuleList()
            # sub_net = mySequential()

            for j in range(n_res_blocks):
                
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                
                if i in self.attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                    
                sub_net.append(TimestepEmbedSequential(*layers))
                # sub_net.add_module(str(j),TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            if i != levels - 1:
                sub_net.append(TimestepEmbedSequential(DownSample(channels)))
                # sub_net.add_module(str(i),TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)
            self.input_blocks.append(sub_net)
            
            
        # self.middle_block = TimestepEmbedSequential(
        #     ResBlock(channels, d_time_emb),
        #     SpatialTransformer(channels, n_heads, tf_layers, d_cond),
        #     ResBlock(channels, d_time_emb),
        # )
        
        self.output_blocks = nn.ModuleList()
        
        cnt = 0
        for i in reversed(range(levels)):
            cnt += 1
            sub_net = nn.ModuleList()
            # sub_net = mySequential()
            for j in range(n_res_blocks + 1):
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                
                if i in self.attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                    
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                    
                sub_net.append(TimestepEmbedSequential(*layers))
                # sub_net.add_module(str(i),TimestepEmbedSequential(*layers))
                
            self.output_blocks.append(sub_net)
            if cnt > 1: break
        
    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channels // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
    def forward(self, x: torch.Tensor, cat_t_embs: torch.Tensor):
        
        self.x_input_block = []
        res = []
        
        x = self.conv(x)
        
        # t_emb = self.time_step_embedding(time_steps)
        # t_emb = self.time_embed(t_emb)
        
        # input half
        ###################
        # 1 block 1
        # 2 block 2
        # 3 block 3 whose output is needed
        ###################
        cnt = 0
        cond = None
        for module in self.input_blocks:
            cnt += 1
            # pdb.set_trace()
            if isinstance(module, nn.ModuleList):
                x = self._forward_nested(module, x, cat_t_embs, cond)
            else:
                x = module(x, cat_t_embs, cond)
                    
            self.x_input_block.append(x)
            if(cnt > 2): res.append(x)
            
        pdb.set_trace()   
        for module in self.output_blocks:
            # x = torch.cat([x,x_input_block.pop()],dim=1)
            # x:[1,2560,32,32] 1280+1280
            if isinstance(module, nn.ModuleList):
                x = self._forward_nested_output(module, x, cat_t_embs, cond)            
            # for sub_module in module:
            #     x = sub_module(x, cat_t_embs, cond)
            res.append(x)
        
        return res
    
    def _forward_nested(self,module_list,x,cat_t_embs,cond):
        # pdb.set_trace()
        for layer in module_list:
            # x = torch.cat([x,self.x_input_block.pop()],dim=1)
            x = layer(x, cat_t_embs, cond)
        return x
    
    def _forward_nested_output(self,module_list,x,cat_t_embs,cond):
        # pdb.set_trace()
        for layer in module_list:
            x = torch.cat([x,self.x_input_block.pop()],dim=1)
            x = layer(x, cat_t_embs, cond)
        return x
        
        
