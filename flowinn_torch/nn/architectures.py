
"""
Author of the tensorflow repo is Jon Errasti Odriozola (tensorflow) | github-id: https://github.com/errasti13
Vishnu Sankar has converted tf to pytorch and re-built the same package but in pytroch
"""


"""
Neural network architectures for Physics-Informed Neural Networks (PINNs).

This module implements various neural network architectures that can be used
as the backbone for PINNs, including:
- Modified Fourier Networks (MFN): A network with Fourier feature mapping to address spectral bias
- Adaptive activation functions: Allow the network to learn activation function shapes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from typing import List


def get_fresh_activation(activation_name: str) -> nn.Module:
    act_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'silu': nn.SiLU
    }
    return act_map.get(activation_name.lower(), nn.ReLU)()

class FourierFeatureLayer(nn.Module):
    def __init__(self, fourier_dim=32, scale=10.0, temporal_scale=None, trainable=False, **kwargs):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.scale = scale
        self.temporal_scale = temporal_scale
        self.trainable = trainable
      
    def _initialize_B(self, input_dim: int, device: torch.device):
        """
        Helper function to create and initialize the B matrix.
        """
        # Create B_data tensor first
        B_data = torch.empty(input_dim, self.fourier_dim, device=device) # Ensure B_data is created
        if self.temporal_scale is not None and input_dim >= 3:
            scales_vec = torch.ones(input_dim, dtype=torch.float32, device=device)
            scales_vec[input_dim - 1] = self.temporal_scale/self.scale
            nn.init.normal_(B_data, mean=0.0, std=1.0) # Initialize B_data
            
            B_data_scaled = torch.einsum('i,ij->ij', scales_vec, B_data)*self.scale
        else:
            nn.init.normal_(B_data, mean=0.0, std=self.scale)
            B_data_scaled = B_data 
        
        if self.trainable:
            self.B = nn.Parameter(B_data_scaled) 
        else:
            self.register_buffer('B', B_data_scaled)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'B'): 
            input_dim = inputs.shape[-1]
            self._initialize_B(input_dim, inputs.device)
        
        x_proj = torch.matmul(inputs, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def get_config(self):
        return {
            "fourier_dim": self.fourier_dim,
            "scale": self.scale,
            "temporal_scale": self.temporal_scale,
            "trainable": self.trainable
        }


class AdaptiveActivation(nn.Module):
    """
    Adaptive activation function layer.
    
    This layer implements an activation function with trainable parameters
    that can adapt during training. It combines a base activation function
    with a learnable linear component.
    
    References:
        - Jagtap et al. (2020) - "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks"
    """
    def __init__(self, base_activation='gelu', initial_param=0.1, **kwargs):
        super().__init__()
        self.base_activation = base_activation
        self.initial_param = initial_param
        self.a = nn.Parameter(torch.tensor(float(self.initial_param), dtype=torch.float32), requires_grad=True) # Ensure float

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.base_activation == 'tanh':
            base = torch.tanh(inputs)

        elif self.base_activation == 'gelu':
            base = F.gelu(inputs)

        elif self.base_activation == 'swish':
            base = inputs*torch.sigmoid(inputs)

        else:
            base = torch.tanh(inputs)
            
        return self.a*inputs + (1.0 - self.a)*base 
    
    def get_config(self):
        return {"base_activation": self.base_activation,
            "initial_param": self.initial_param}

def create_mfn_model(input_shape, output_shape, eq, activation='gelu', fourier_dim=32, 
                    layer_size=None, learning_rate=0.001, use_adaptive_activation=False):
    """
    Create a Modified Fourier Network (MFN) model.
    
    This architecture helps overcome spectral bias in standard MLPs by using Fourier
    feature mapping to better capture high-frequency components of the solution.
    
    Args:
        input_shape: Shape of the input tensor (e.g., (3,) for x,y,t)
        output_shape: Number of output variables
        eq: Equation name or identifier
        activation: Activation function to use ('gelu', 'tanh', 'swish')
        fourier_dim: Dimension of the Fourier feature mapping
        layer_size: List of hidden layer sizes, defaults to [128, 128, 128, 128]
        learning_rate: Initial learning rate for Adam optimizer
        use_adaptive_activation: Whether to use adaptive activation functions
        
    Returns:
        A PINN model with MFN architecture
    """
    if layer_size is None:
        layer_size = [128, 128, 128, 128]
    
    from flowinn.models.model import PINN
    if isinstance(input_shape, tuple):
        num_input_features = input_shape[-1]
    else:
        num_input_features = input_shape
    
    pytorch_layers = []

    pytorch_layers.append(FourierFeatureLayer(fourier_dim=fourier_dim))
    current_features = 2*fourier_dim # output of fourierFeatureLayer
    
    for units in layer_size:
        x = nn.Linear(current_features, units, bias=True)
        nn.init.xavier_normal_(x.weight)
        if x.bias is not None:
            nn.init.zeros_(x.bias)
        pytorch_layers.append(x)
        
        if use_adaptive_activation:
            pytorch_layers.append(AdaptiveActivation(base_activation=activation))

        else:
            if activation == 'gelu':
                pytorch_layers.append(nn.GELU())
            elif activation == 'tanh':
                pytorch_layers.append(nn.Tanh())
            elif activation == 'swish':
                pytorch_layers.append(nn.SiLU()) # pytorch SiLU is swish: x*sigmoid(x)
            else:
                pytorch_layers.append(nn.Tanh())

        current_features = units 
    output = nn.Linear(current_features, output_shape)
    nn.init.xavier_uniform_(output.weight)
    if output.bias is not None:
        nn.init.zeros_(output.bias)
    pytorch_layers.append(output)

    mfn_model = nn.Sequential(*pytorch_layers)

    pinn_model = PINN(input_shape=input_shape,
        output_shape=output_shape,
        eq=eq,
        layers=layer_size,
        activation=activation,
        learning_rate=learning_rate)
    
    pinn_model.model = mfn_model
    
    return pinn_model 

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation_module: nn.Module):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim) 
        self.activation = activation_module
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out=self.activation(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += residual 
        out = self.activation(out)
        return out

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer_size: List[int], activation_module: nn.Module):
        super().__init__()
        layers = []
        current_dim = input_dim

        for hidden_units in layer_size:
            linear_layer = nn.Linear(current_dim, hidden_units)
            layers.append(linear_layer)
            layers.append(activation_module)
            current_dim = hidden_units
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SimpleMLPResidual(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer_size: List[int], activation_name: str):
        super().__init__()
        network_layers = []
        current_dim = input_dim

        if not layer_size: 
            network_layers.append(nn.Linear(input_dim, output_dim))
        
        else:
            for i, hidden_units in enumerate(layer_size):
                activation_module = get_fresh_activation(activation_name)
                network_layers.append(nn.Linear(current_dim, hidden_units))
                network_layers.append(nn.BatchNorm1d(hidden_units))
                # network_layers.append(activation_module)
                current_dim = hidden_units 
                network_layers.append(ResidualBlock(dim=current_dim, activation_module=activation_module))

            network_layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*network_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Conv1dResidualBlock(nn.Module):
    """
    A Residual Block using 1D Convolutions.
    Structure: Conv1d -> BN -> Activation -> Conv1d -> BN -> + shortcut -> Activation
    """
    def __init__(self, channels: int, activation_module: nn.Module, kernel_size: int = 3, 
                 drop_out: float = 0.0, stride: int = 1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for simple 'same' padding calculation.")
        
        padding = (kernel_size - 1)//2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding if stride == 1 else (kernel_size -1)//2*stride, # more careful padding if stride changes length
                               bias=False) # false because batchNorm is just after it
        self.bn1 = nn.BatchNorm1d(channels)
        self.activation = activation_module
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.dropout = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1:
            self.shortcut = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1, stride=stride, bias=False),nn.BatchNorm1d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        if not isinstance(self.dropout, nn.Identity):
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  
        out = self.activation(out) 
        return out


class SimpleConv1dResidualNet(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 initial_conv_channels: int,
                 initial_sequence_length: int,
                 conv_block_channels: List[int],
                 activation_name: str,
                 conv_kernel_size: int = 3,
                 drop_out: float = 0.0,
                 final_mlp_layers: List[int] = []
                ):
        super().__init__()

        if initial_conv_channels <= 0 or initial_sequence_length <= 0:
            raise ValueError("initial_conv_channels and initial_sequence_length must be positive.")

        self.initial_conv_channels = initial_conv_channels
        self.initial_sequence_length = initial_sequence_length

        self.initial_fc = nn.Linear(input_dim, initial_conv_channels*initial_sequence_length)
        self.initial_bn_fc = nn.BatchNorm1d(initial_conv_channels*initial_sequence_length)
        self.initial_activation = get_fresh_activation(activation_name)

        conv_layers_list = []
        current_channels = initial_conv_channels

        for stage_channels in conv_block_channels:
            if current_channels != stage_channels:
                conv_layers_list.append(nn.Conv1d(current_channels, stage_channels, kernel_size=1, bias=False))
                conv_layers_list.append(nn.BatchNorm1d(stage_channels))
                conv_layers_list.append(get_fresh_activation(activation_name))
                current_channels = stage_channels

            conv_layers_list.append(Conv1dResidualBlock(
                channels=current_channels,
                activation_module=get_fresh_activation(activation_name),
                kernel_size=conv_kernel_size,
                drop_out=drop_out
            ))

        self.convolutional_middle = nn.Sequential(*conv_layers_list)
        self._final_conv_output_channels = current_channels

        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()

        mlp_head_layers = []
        mlp_input_dim = self._final_conv_output_channels

        if not final_mlp_layers:
            mlp_head_layers.append(nn.Linear(mlp_input_dim, output_dim))
        else:
            current_mlp_dim = mlp_input_dim
            for hidden_units in final_mlp_layers:
                mlp_head_layers.append(nn.Linear(current_mlp_dim, hidden_units))
                mlp_head_layers.append(nn.BatchNorm1d(hidden_units))
                mlp_head_layers.append(get_fresh_activation(activation_name))
                if drop_out > 0.0:
                    mlp_head_layers.append(nn.Dropout(drop_out))
                current_mlp_dim = hidden_units
            mlp_head_layers.append(nn.Linear(current_mlp_dim, output_dim))

        self.mlp_head = nn.Sequential(*mlp_head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_fc(x)
        out = self.initial_bn_fc(out)
        out = self.initial_activation(out)
        out = out.view(-1, self.initial_conv_channels, self.initial_sequence_length)
        out = self.convolutional_middle(out)
        out = self.adaptive_pool(out)
        out = self.flatten(out)
        out = self.mlp_head(out)
        return out
