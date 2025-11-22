import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
#from models.helpers.utils import make_divisible
from enum import Enum
import torch.nn.functional as F
import math
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# Frequency attention module
class FrequencyAttention(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, n_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Project inputs to hidden_dim before attention
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Residual connection with layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x_proj = self.input_projection(x)  # (B, 28, 64)
        
        # Self attention
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)
        
        # Residual connection with layer normalization
        x_out = self.norm(x_proj + self.dropout(attn_out))
        
        return x_out  # (B, 28, 64)


# Cross-modal audio attention module
class AudioCrossAttention(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, n_heads=8):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Cross attention with a single global query
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):  # (B, 28, 16)
        x_proj = self.input_projection(x)  # (B, 28, 64)
        
        # Build a global query via mean pooling
        global_query = x_proj.mean(dim=1, keepdim=True)  # (B, 1, 64)
        
        # Cross-attention: the global query attends to every channel
        cross_out, _ = self.cross_attention(
            query=global_query,           # (B, 1, 64)
            key=x_proj,                   # (B, 28, 64)
            value=x_proj,                 # (B, 28, 64)
        )  # (B, 1, 64)
        
        # Broadcast the attended features back to every position
        cross_out = cross_out.expand(-1, x_proj.size(1), -1)  # (B, 28, 64)
        
        # Residual connection
        output = self.norm(x_proj + self.dropout(cross_out))
        
        return output  # (B, 28, 64)


# Convolutional transform path
class ConvT_Path(nn.Module):
    def __init__(self, base_channels, channels_multiplier, expansion_rate):
        super().__init__()
        
        mid_channels = int(base_channels * channels_multiplier)
        out_channels = int(mid_channels * expansion_rate)
        
        # Pointwise convolution
        self.conv2d = Conv2dNormActivation(
            in_channels=base_channels,
            out_channels=mid_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            inplace=False
        )
        
        # Left branch depthwise convolution
        self.conv2d_L = Conv2dNormActivation(
            in_channels=mid_channels,
            out_channels=out_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            groups=mid_channels,
            kernel_size=(5,1),
            stride=(1,1),
            padding=(2,0),
            inplace=False
        )
        
        # Right branch depthwise convolution
        self.conv2d_R = Conv2dNormActivation(
            in_channels=out_channels,
            out_channels=out_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            groups=out_channels,
            kernel_size=(1,5),
            stride=(2,2),
            padding=(0,2),
            inplace=False
        )
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.conv2d_L(x)
        x = F.silu(self.conv2d_R(x))
        return x


# Convolutional transform block with channel shuffle
class ConvT_HaveShuffle(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        
        # 1x1 Pointwise Conv
        self.conv2d_1_1 = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=int(base_channels/2),
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            kernel_size=1,
            stride=1,
            padding=0,
            inplace=False
        )
        
        # Channel shuffle
        self.cs_1 = ChannelShuffle(int(base_channels/4))
        
        # Left branch kernel_size=(3,1)
        self.codw2d_1L3 = Conv2dNormActivation(
            in_channels=int(base_channels/2),
            out_channels=base_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            groups=int(base_channels/2),
            kernel_size=(3,1),
            stride=(1,1),
            padding=(1,0),
            inplace=False
        )
        
        # Right branch kernel_size=(1,3)
        self.codw2d_1R1 = Conv2dNormActivation(
            in_channels=base_channels,
            out_channels=base_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            groups=int(base_channels/2),
            kernel_size=(1,3),
            stride=(1,1),
            padding=(0,1),
            inplace=False
        )
        
    def forward(self, x):
        x = self.conv2d_1_1(x)
        x = self.cs_1(x)
        x = self.codw2d_1L3(x)
        x = F.silu(self.codw2d_1R1(x))
        
        return x

    
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: Num of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel (B, C, H, W) -> (B, C, H*W) -> (B, C)
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        # Scale the input tensor by the channel score
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class ChannelShuffle(nn.Module):
    def __init__(self, groups = 1):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.groups == 0, "The number of channels must be divisible by the number of groups"
        
        #if torch.rand(1).item() < self.p:
        channels_per_group = num_channels // self.groups

        # Reshape the tensor to (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # Permute the tensor to (batch_size, channels_per_group, groups, height, width)
        # [0,1,2,...,63] -> [0,16,32,48, 1,17,33,49, ..., 15,31,47,63]
        x = x.permute(0, 2, 1, 3, 4).contiguous()      

        # Reshape the tensor back to (batch_size, num_channels, height, width)
        x = x.view(batch_size, num_channels, height, width)

        return x  
                 

class Network_test(nn.Module):
    def __init__(self, config):
        super(Network_test, self).__init__()
        n_classes = config['n_classes']  # Number of classes from the config
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        divisor = config['divisor']


        # 2D Conv (1-7)
        self.conv2d_1_0 = Conv2dNormActivation(in_channels = in_channels,
                                             out_channels = 7,
                                             norm_layer = nn.BatchNorm2d,
                                             activation_layer = None, #torch.nn.ReLU,
                                             kernel_size = 3,
                                             stride = 1,
                                             padding= 1,
                                             inplace=False)

        # Channel shuffle branch
        self.ConvT_HS = ConvT_HaveShuffle(7, base_channels)  
                                                        
        # SE block (24-24)
        self.se_1_1      = ChannelSELayer(base_channels)
        
        # Parallel convolutional transform paths
        self.convt_path_left = ConvT_Path(base_channels, channels_multiplier, expansion_rate)
        self.convt_path_right = ConvT_Path(base_channels, channels_multiplier, expansion_rate)

        # SE block (72-72)
        self.se_2_1     = ChannelSELayer(int(base_channels * channels_multiplier * expansion_rate))
        self.se_2_2     = ChannelSELayer(int(base_channels * channels_multiplier * expansion_rate))

        # 2D Pointwise Conv (72-72)
        self.conv2d_3_1 = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               out_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, # adding relu degrades perf                              
                                               kernel_size = (1,1),
                                               stride = 1,
                                               padding = 0,
                                               inplace = False)
        
        self.cs_3 = ChannelShuffle(int(base_channels * channels_multiplier * expansion_rate/2))
        
        # 2D Depthwise Conv (72-144)
        self.codw2d_3_1 = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               out_channels = int(base_channels * channels_multiplier * expansion_rate * expansion_rate),
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, #torch.nn.ReLU,   
                                               groups = int(base_channels * channels_multiplier * expansion_rate),
                                               kernel_size = (5,1), #change from 7 to 3 as kernel size is bigger than input size
                                               stride = (2,1),
                                               padding = (2,0),
                                               inplace = False)    
        
        # 2D Conv (144-28)        
        self.conv2d_4_1 = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier * expansion_rate  * expansion_rate),
                                             out_channels = int(base_channels * channels_multiplier * expansion_rate * expansion_rate / divisor),
                                             norm_layer = nn.BatchNorm2d,
                                             activation_layer = None, #torch.nn.ReLU,
                                             kernel_size = (5,1),
                                             stride = (1,1),
                                             padding = (2,0),
                                             inplace = False)   
        # (B, 28, 16) -> (B, 28, 64)
        # Frequency attention branches
        self.frequency_attention_1 = FrequencyAttention(input_dim=16, hidden_dim=32, n_heads=4)
        self.frequency_attention_2 = AudioCrossAttention(input_dim=16, hidden_dim=32, n_heads=4)
             
        self.drop2d_0_3 = nn.Dropout2d(p = 0.3)
        self.drop2d_0_5 = nn.Dropout2d(p = 0.5)
        self.drop1d_0_5 = nn.Dropout1d(p = 0.5)
        self.drop1d_0_3 = nn.Dropout1d(p = 0.3)  

        self.maxp2d_2_2 = nn.MaxPool2d(kernel_size = (2,2))
        self.avgp2d_2_2 = nn.AvgPool2d(kernel_size = (2,2))   
  
        self.param1 = nn.Parameter(torch.tensor(0.5))
        self.param2 = nn.Parameter(torch.tensor(0.5))
        self.param3 = nn.Parameter(torch.tensor(0.5))
        self.param4 = nn.Parameter(torch.tensor(0.5))
        self.param5 = nn.Parameter(torch.tensor(0.5))

        # (B, 1, 28) -> (B, n_classes, 28)
        self.conv1d_5_2 = nn.Conv1d(1, #int(base_channels * channels_multiplier * expansion_rate * channels_multiplier * expansion_rate/ divisor),
                                             out_channels = n_classes,  # Uses dynamically configured classes
                                             kernel_size = int(base_channels * channels_multiplier * expansion_rate * expansion_rate / divisor),
                                             stride = 1,
                                             padding = 0
                                             )     
        
        #self.apply(initialize_weights) # default pytorch initializatin seens to work better

    def forward(self, x, return_attn_features=False): # x's shape (batch_size, channels, mel-bins, frames)
        x = self.conv2d_1_0(x) 

        # Channel shuffle feature extraction
        x = self.ConvT_HS(x)

        x = self.se_1_1(x)

        xm = self.maxp2d_2_2(x)
        xa = self.avgp2d_2_2(x)

        # Parallel convolutional transform paths
        xm = self.convt_path_left(xm)
        xa = self.convt_path_right(xa)

        xm = self.se_2_1(xm)
        xa = self.se_2_2(xa)   
           
        xm1 = self.maxp2d_2_2(xm)
        xm2 = self.avgp2d_2_2(xm)
        xa1 = self.avgp2d_2_2(xa)
        xa2 = self.maxp2d_2_2(xa)

        xa = self.param1 * xa1 + (1 - self.param1) * xm2
        xm = self.param2 * xm1 + (1- self.param2) * xa2
        x1 = self.param4 * xa + (1 - self.param4) * xm
        
        x = self.conv2d_3_1(x1) + x1
        x = self.cs_3(x)
        x = F.silu(self.codw2d_3_1(x))  
        x = self.drop2d_0_3(x) 
        
        x = F.silu(self.conv2d_4_1(x))
        x1  = torch.mean(x, dim = 3)

        attn_output_1 = self.frequency_attention_1(x1)
        attn_output_2 = self.frequency_attention_2(x1)
        attn_output = self.param5 * attn_output_1 + (1 - self.param5) * attn_output_2
        x2 = (self.drop1d_0_5(attn_output))

        x1 = torch.mean(x1, dim = 2)
        x2 = torch.mean(x2, dim = 2)
        
        x = F.silu(self.param3 * x1 + (1 - self.param3) * x2) # better than concat

        x = x.unsqueeze(1)
        logits = self.conv1d_5_2(x)
        logits = logits.view(logits.size(0), -1)      

        if return_attn_features:
            return logits, attn_output  # attn_output: (B, C, 64) where C=28 or 43
        return logits        
                 
def get_ntu_model(n_classes=10, in_channels=1, n_blocks=(3, 2, 1), 
                    base_channels = 16, channels_multiplier = 1.5, expansion_rate = 2, divisor = 3, strides=None):
    
    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "divisor": divisor,
        "n_blocks": n_blocks,
        "strides": strides
    }
    m = Network_test(model_config)
    return m