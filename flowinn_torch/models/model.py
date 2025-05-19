
"""
Author of the tensorflow repo is Jon Errasti Odriozola (tensorflow) | github-id: https://github.com/errasti13
Vishnu Sankar has converted tf to pytorch and re-built the same package but in pytroch
"""

import os
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Any
from flowinn_torch.plot.boundary_visualization import BoundaryVisualization 

from torchsummary import summary
from flowinn_torch.nn.architectures import FourierFeatureLayer, SimpleMLP, SimpleMLPResidual, SimpleConv1dResidualNet 
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class with a Modified Fourier Network (MFN) architecture.
    """

    def __init__(self, input_shape: int = 2, output_shape: int = 1, layers: List[int] = [20, 40, 60, 40, 20],
                 activation: str = 'silu', learning_rate: float = 0.01, eq: str = None, model_type: str = 'conv1d_residual_net', # other models 'mfn' or 'simple_mlp' or 'simple_residual_mlp'
                 mfn_fourier_dim_steady: int = 32,
                 mfn_fourier_dim_unsteady: int = 64,
                 mfn_scale_steady: float = 10.0,
                 mfn_scale_unsteady_spatial: float = 5.0,
                 mfn_temporal_scale_unsteady: float = 25.0,
                 mfn_fourier_trainable: bool = False, 
                 conv_initial_channels: int = 64,
                 conv_initial_length: int = 50, 
                 conv_block_channels: List[int] = [64, 64, 64],
                 conv_kernel_size: int = 3,
                 drop_out: float = 0.0,
                 conv_final_mlp_layers: List[int] = [64]) -> None:
        """
        Initialize a Physics-Informed Neural Network.
        
        Args:
            input_shape: Shape of input tensor - either an integer (number of dimensions) or a tuple
            output_shape: Number of output variables
            layers: List of hidden layer sizes
            activation: Activation function to use
            learning_rate: Initial learning rate
            eq: Equation identifier
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.activation_name = activation 
        
        if isinstance(input_shape, tuple):
            self.input_dim = input_shape[0] if len(input_shape) > 0 else 1
        else:
            self.input_dim = input_shape
            
        self.output_shape = output_shape 
        self.hidden_layers_config = layers

        self.model_type = model_type.lower()
        self.eq: str = eq if eq is not None else "default_eq"

        if self.model_type == 'simple_mlp':
            self.model: nn.Module = self.create_simple_mlp_model(
                input_dim=self.input_dim, 
                output_dim=self.output_shape, 
                layer_size=self.hidden_layers_config,
                activation_name=self.activation_name)
        
        elif self.model_type == 'mfn':
            self.model: nn.Module = self.create_mfn_style_model(
                input_dim=self.input_dim, 
                output_shape=self.output_shape, 
                layers_config=self.hidden_layers_config,
                activation_name=self.activation_name, 
                fourier_dim_steady=mfn_fourier_dim_steady,
                fourier_dim_unsteady=mfn_fourier_dim_unsteady,
                scale_steady=mfn_scale_steady,
                scale_unsteady_spatial=mfn_scale_unsteady_spatial,
                temporal_scale_unsteady=mfn_temporal_scale_unsteady,
                fourier_trainable=mfn_fourier_trainable)
        
        elif self.model_type == 'simple_residual_mlp':
            self.model: nn.Module = self.create_simple_residual_mlp_model(
                input_dim = self.input_dim, 
                output_dim=self.output_shape, 
                layer_size=self.hidden_layers_config,
                activation_name=self.activation_name)

        elif self.model_type == 'conv1d_residual_net': 
            self.model: nn.Module = self.create_conv1d_residual_net_model(
                input_dim=self.input_dim,
                output_dim=self.output_shape,
                initial_conv_channels=conv_initial_channels,
                initial_sequence_length=conv_initial_length,
                conv_block_channels=conv_block_channels,
                activation_name=self.activation_name,
                conv_kernel_size=conv_kernel_size,
                drop_out=drop_out,
                final_mlp_layers=conv_final_mlp_layers)

        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'. Choose 'simple_mlp' or 'mfn'.")
        
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=50, min_lr=1e-5)
        self.boundary_visualizer: Optional[BoundaryVisualization] = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print("-"*70)
        print(f"Model Summary (Input Dim: {self.input_dim}, Output Dim: {self.output_shape}):")
        try:
            summary(self.model, input_size=(self.input_dim,), device=str(self.device))
        except Exception as e:
            print(f"Could not print model summary using torchsummary: {e}")
            print("Printing basic model structure instead:")
            print(self.model)
        print("-"*70)

    def create_simple_mlp_model(self, input_dim: int, output_dim: int, 
    layer_size: List[int], activation_name: str) -> SimpleMLP:
        """
        Creates the simple MLP model using the SimpleMLP class.
        The activation module is now created inside this method.
        """
        if activation_name.lower() == 'relu':
            activation_module = nn.ReLU()
        elif activation_name.lower() == 'tanh':
            activation_module = nn.Tanh()
        elif activation_name.lower() == 'gelu':
            activation_module = nn.GELU()
        elif activation_name.lower() == 'silu' or activation_name.lower() == 'swish':
            activation_module = nn.SiLU()
        else:
            print(f"Warning: Activation '{activation_name}' not recognized for SimpleMLP. Defaulting to ReLU.")
            activation_module = nn.ReLU()

        return SimpleMLP(input_dim, output_dim, layer_size, activation_module)



    def create_simple_residual_mlp_model(self, input_dim: int, output_dim: int, 
    layer_size: List[int], activation_name: str) -> SimpleMLPResidual:
        """
        Creates the simple MLP model using the SimpleMLP class.
        The activation module is now created inside this method.
        """

        if activation_name.lower() == 'relu':
            activation_module = nn.ReLU()
        elif activation_name.lower() == 'tanh':
            activation_module = nn.Tanh()
        elif activation_name.lower() == 'gelu':
            activation_module = nn.GELU()
        elif activation_name.lower() == 'silu' or activation_name.lower() == 'swish':
            activation_module = nn.SiLU()
        else:
            print(f"Warning: Activation '{activation_name}' not recognized for SimpleMLP. Defaulting to ReLU.")
            activation_module = nn.ReLU()

        return SimpleMLPResidual(input_dim, output_dim, layer_size, activation_name)



    def create_conv1d_residual_net_model(self, input_dim: int, output_dim: int,
                                          initial_conv_channels: int,
                                          initial_sequence_length: int,
                                          conv_block_channels: List[int],
                                          activation_name: str,
                                          conv_kernel_size: int,
                                          drop_out: float,
                                          final_mlp_layers: List[int]) -> SimpleConv1dResidualNet:
        """Creates the SimpleConv1dResidualNet model."""

        if activation_name.lower() == 'relu':
            activation_module = nn.ReLU()
        elif activation_name.lower() == 'tanh':
            activation_module = nn.Tanh()
        elif activation_name.lower() == 'gelu':
            activation_module = nn.GELU()
        elif activation_name.lower() == 'silu' or activation_name.lower() == 'swish':
            activation_module = nn.SiLU()
        else:
            print(f"Warning: Activation '{activation_name}' not recognized for SimpleMLP. Defaulting to ReLU.")
            activation_module = nn.ReLU()

        return SimpleConv1dResidualNet(
            input_dim=input_dim,
            output_dim=output_dim,
            initial_conv_channels=initial_conv_channels,
            initial_sequence_length=initial_sequence_length,
            conv_block_channels=conv_block_channels,
            activation_name=activation_name,
            conv_kernel_size=conv_kernel_size,
            drop_out=drop_out,
            final_mlp_layers=final_mlp_layers
        )

    def create_mfn_style_model(self, input_dim: int, output_shape: int, layers_config: List[int], activation_name: str) -> nn.Sequential:
        """
        Creates an MFN model (PyTorch nn.Sequential) that begins with a Fourier feature mapping layer.
        """
        pytorch_layers = []
        
        is_unsteady = input_dim >= 3
        
        current_features = input_dim
        
        if is_unsteady:
            fourier_layer = FourierFeatureLayer(fourier_dim=64, scale=5.0, temporal_scale=25.0, trainable=False)
            pytorch_layers.append(fourier_layer)

            current_features = 2*64 
        else:
            fourier_layer = FourierFeatureLayer(fourier_dim=32, scale=10.0, trainable=False)
            pytorch_layers.append(fourier_layer)
            current_features = 2*32

        for i, units in enumerate(layers_config):
            if i > 0 and i % 2 == 0 and is_unsteady:
                pytorch_layers.append(nn.Dropout(0.1))

            linear_layer = nn.Linear(current_features, units)
            nn.init.xavier_normal_(linear_layer.weight)
            if linear_layer.bias is not None:
                nn.init.zeros_(linear_layer.bias)
            pytorch_layers.append(linear_layer)
            
            pytorch_layers.append(nn.BatchNorm1d(units))
            if activation_name == 'gelu':
                pytorch_layers.append(nn.GELU())
            elif activation_name == 'tanh':
                pytorch_layers.append(nn.Tanh())
            elif activation_name == 'silu':
                pytorch_layers.append(nn.SiLU())
            else:
                pytorch_layers.append(nn.GELU()) 
            current_features = units
            
        output_linear_layer = nn.Linear(current_features, output_shape)
        nn.init.xavier_uniform_(output_linear_layer.weight)
        if output_linear_layer.bias is not None:
            nn.init.zeros_(output_linear_layer.bias)
        pytorch_layers.append(output_linear_layer)
        
        return nn.Sequential(*pytorch_layers)

    def _create_learning_rate_schedule(self, optimizer: optim.Optimizer, learning_rate: float) -> optim.lr_scheduler._LRScheduler:
        """
        Creates a learning rate schedule for PyTorch.
        """
        if hasattr(self, 'input_dim') and self.input_dim >= 3: 
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000, eta_min=learning_rate*0.01)
        else: 
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=(0.9 ** (1/1000)))

    @staticmethod
    def _as_numpy(data: Any) -> np.ndarray:
        """ Helper to convert tensors to numpy arrays """
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        elif hasattr(data, 'numpy') and callable(data.numpy) and not isinstance(data, np.ndarray):
            return np.array(data)
        return np.asarray(data)

    def _add_training_noise(self, batch_coords: torch.Tensor, noise_level=0.01) -> torch.Tensor:
        coords_np = self._as_numpy(batch_coords) 
        
        scales = []
        for dim in range(coords_np.shape[1]):
            dim_range = np.max(coords_np[:, dim]) - np.min(coords_np[:, dim])
            if dim_range == 0: dim_range = 1.0
            scales.append(dim_range*noise_level)
        
        noise_np = np.zeros_like(coords_np)
        for dim in range(coords_np.shape[1]):
            if dim == coords_np.shape[1] - 1 and coords_np.shape[1] >= 3: 
                continue
            noise_np[:, dim] = np.random.normal(0, scales[dim], size=coords_np.shape[0])
        
        noisy_coords_np = coords_np + noise_np
        return torch.tensor(noisy_coords_np, dtype=torch.float32, device=batch_coords.device)


    def generate_batches(self, mesh, num_batches, time_window_size=3, add_noise=True, noise_level=0.01):
        batches = []
        x_flat = self._as_numpy(mesh.x).ravel()
        y_flat = self._as_numpy(mesh.y).ravel()
        z_flat = None
        if hasattr(mesh, 'z') and mesh.z is not None and not mesh.is2D : # check if z exists
            z_flat = self._as_numpy(mesh.z).ravel()

        if hasattr(mesh, 'is_unsteady') and mesh.is_unsteady and hasattr(mesh, 't'):
            mesh_t_np = self._as_numpy(mesh.t)
            spatial_points = len(x_flat) 
            time_points = len(mesh_t_np)
            time_window_size = min(time_window_size, time_points)
            
            for batch_idx in range(num_batches):
                num_batch_samples = spatial_points*time_window_size//num_batches if num_batches > 0 else spatial_points*time_window_size
                if num_batch_samples == 0: num_batch_samples = 10

                idx_x = np.random.choice(len(x_flat), num_batch_samples)
                idx_y = np.random.choice(len(y_flat), num_batch_samples)
                idx_t = np.random.choice(len(mesh_t_np), num_batch_samples)

                batch_coords_list = []
                if z_flat is not None: 
                     idx_z = np.random.choice(len(z_flat), num_batch_samples)
                     for i in range(num_batch_samples):
                        batch_coords_list.append([x_flat[idx_x[i]], y_flat[idx_y[i]], z_flat[idx_z[i]], mesh_t_np[idx_t[i]]])
                else: 
                    for i in range(num_batch_samples):
                        batch_coords_list.append([x_flat[idx_x[i]], y_flat[idx_y[i]], mesh_t_np[idx_t[i]]])
                
                batch_coords_np = np.array(batch_coords_list, dtype=np.float32)
                
                batch_tensor = torch.tensor(batch_coords_np, dtype=torch.float32)
                if add_noise:
                    batch_tensor = self._add_training_noise(batch_tensor, noise_level)
                
                batches.append(batch_tensor)

        else: 
            total_points = len(x_flat)
            points_per_batch = total_points//num_batches if num_batches > 0 else total_points
            if points_per_batch == 0: points_per_batch = 10

            for _ in range(num_batches):
                indices = np.random.choice(total_points, size=points_per_batch, replace=True)
                if z_flat is not None:
                     batch_coords_np = np.stack([x_flat[indices], y_flat[indices], z_flat[indices]], axis=-1).astype(np.float32)
                else: 
                    batch_coords_np = np.stack([x_flat[indices], y_flat[indices]], axis=-1).astype(np.float32)
              
                batch_tensor = torch.tensor(batch_coords_np, dtype=torch.float32)
                if add_noise:
                    batch_tensor = self._add_training_noise(batch_tensor, noise_level)
                batches.append(batch_tensor)
        return batches

    def train_step(self, loss_function, batch_data: torch.Tensor, print_gradients=False) -> torch.Tensor:
        self.model.train() 
        self.optimizer.zero_grad() 
        batch_data = batch_data.to(self.device)
        loss = loss_function(batch_data=batch_data) 
        loss.backward() 

        if print_gradients:
            print(f"\n--- Gradients after loss.backward() for loss: {loss.item():.4e} ---")
            total_grad_norm = 0.0
            num_params_with_grad = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    num_params_with_grad +=1
                    if grad_norm < 1e-9 and param.requires_grad: 
                        print(f"WARNING: Very small gradient norm for layer: {name} ({grad_norm:.2e})")
                else:
                    if param.requires_grad: 
                        print(f"WARNING: No gradient for trainable layer: {name}")
            
            if num_params_with_grad > 0:
                print(f"Overall gradient norm (sum of individual norms): {total_grad_norm:.6f}")
            else:
                print("WARNING: No parameters received gradients!")
        
            max_grad_norm_overall = 0.0
            min_grad_norm_overall = float('inf')
            avg_grad_norm_overall = 0.0
            num_params_with_grad = 0

            print(f"{'Layer Name':<60} | {'Param Shape':<15} | {'Grad Norm':<12} | {'Grad Mean':<12} | {'Grad Std':<12} | {'Param Norm':<12}")
            print("-"*130)

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_norm = param.data.norm().item() 
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    
                    print(f"{name:<60} | {str(list(param.shape)):<15} | {grad_norm:<12.3e} | {grad_mean:<12.3e} | {grad_std:<12.3e} | {param_norm:<12.3e}")

                    if param.requires_grad: 
                        total_grad_norm += grad_norm 
                        max_grad_norm_overall = max(max_grad_norm_overall, grad_norm)
                        min_grad_norm_overall = min(min_grad_norm_overall, grad_norm)
                        avg_grad_norm_overall += grad_norm
                        num_params_with_grad +=1
                        if grad_norm < 1e-8:
                            print(f"    ---> WARNING: Vanishingly small gradient norm for layer: {name} ({grad_norm:.2e})")
                
                else:
                    if param.requires_grad:
                        print(f"{name:<60} | {str(list(param.shape)):<15} | {'None':<12} | {'None':<12} | {'None':<12} | {param.data.norm().item():<12.3e}")
                        print(f"-------> CRITICAL WARNING: No gradient for trainable layer: {name}")
            
            if num_params_with_grad > 0:
                avg_grad_norm_overall /= num_params_with_grad
                print("-"*130)
                print(f"Total Grad Norm (sum of layer norms): {total_grad_norm:.3e}") 
                print(f"Max Layer Grad Norm: {max_grad_norm_overall:.3e}")
                print(f"Min Layer Grad Norm: {min_grad_norm_overall:.3e}")
                print(f"Avg Layer Grad Norm: {avg_grad_norm_overall:.3e}")
            else:
                print("CRITICAL WARNING: No parameters received gradients!")
            print("--- End Gradients Check ---\n")
        
            self.optimizer.step()

            print(f"\n--- Gradients Detail for Loss: {loss.item():.4e} (Epoch {getattr(self, 'current_epoch', 'N/A')}, Batch {getattr(self, 'current_batch_idx', 'N/A')}) ---")
            print(f"{'Layer Name':<70} | {'Param Shape':<18} | {'Grad Norm':<12} | {'Grad Mean':<12} | {'Grad Std':<12} | {'Data Norm':<12}")
            print("-"*150)
            any_grad_missing = False
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue 

                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    param_norm = param.data.norm().item()
                    print(f"{name:<70} | {str(list(param.shape)):<18} | {grad_norm:<12.2e} | {grad_mean:<12.2e} | {grad_std:<12.2e} | {param_norm:<12.2e}")
                    if grad_norm < 1e-8:
                        print(f"-------> WARNING: Vanishingly small gradient norm for: {name}")
                else:
                    param_norm = param.data.norm().item()
                    print(f"{name:<70} | {str(list(param.shape)):<18} | {'None':<12} | {'None':<12} | {'None':<12} | {param_norm:<12.2e}")
                    print(f"-------> CRITICAL WARNING: No gradient computed for trainable layer: {name}")
                    any_grad_missing = True
            if any_grad_missing:
                print("    ---> One or more trainable layers did not receive gradients!")
            print("--- End Gradients Detail ---\n")

        return loss 

    def train(self, loss_function, mesh, epochs: int = 1000, num_batches: int = 1,
              print_interval: int = 100, autosave_interval: int = 100,
              plot_loss: bool = False, bc_plot_interval: Optional[int] = None, 
              domain_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
              airfoil_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              output_dir: str = 'bc_plots', patience: int = 10000, min_delta: float = 1e-6,
              time_window_size: int = 5, add_noise: bool = True, noise_level: float = 0.01,
              use_cpu: bool = False, batch_data: Optional[List[torch.Tensor]] = None) -> dict :
        
        if use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Training on device: {self.device}")

        return self._train_implementation(
            loss_function, mesh, epochs, num_batches, 
            print_interval, autosave_interval, plot_loss, 
            bc_plot_interval, domain_range, airfoil_coords, 
            output_dir, patience, min_delta, time_window_size, 
            add_noise, noise_level, batch_data
        )

    def _train_implementation(self, loss_function, mesh, epochs, num_batches,
                              print_interval, autosave_interval, plot_loss, 
                              bc_plot_interval, domain_range, airfoil_coords,
                              output_dir, patience, min_delta, time_window_size, 
                              add_noise, noise_level, batch_data=None) -> dict: 
        import time
        import gc
        
        loss_history = []
        lr_history = []
        epoch_history = []
        time_history = []
        last_loss = float('inf')
        patience_counter = 0
        total_training_time = 0.0

        if bc_plot_interval is not None:
            self.boundary_visualizer = BoundaryVisualization(output_dir=output_dir)

        live_fig, live_ax1, live_ax2, live_line1, live_line2 = None, None, None, None, None
        if plot_loss:
            plt.ion()
            live_fig, (live_ax1, live_ax2) = plt.subplots(1, 2, figsize=(15, 5))
            live_ax1.set_xlabel('Epoch')
            live_ax1.set_ylabel('Loss')
            live_line1, = live_ax1.semilogy([], [], label='Training Loss')
            live_ax1.legend()
            
            live_ax2.set_xlabel('Epoch')
            live_ax2.set_ylabel('Time per Epoch (s)')
            live_line2, = live_ax2.plot([], [], label='Time/Epoch')
            live_ax2.legend()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']

            if (epoch + 1) % print_interval == 0 or epoch == 0:
                 lr_history.append(current_lr) 
            current_batches = []

            if batch_data is not None and len(batch_data) > 0:
                current_batches = batch_data
                np.random.shuffle(current_batches) 

            else:
                current_batches = self.generate_batches(
                    mesh, num_batches, 
                    time_window_size=time_window_size,
                    add_noise=add_noise, noise_level=noise_level
                )
            
            epoch_loss_sum = 0.0
            num_processed_batches = 0
            for batch_idx, batch_tensor in enumerate(current_batches):
                try:
                    if batch_tensor.shape[0] > 5000: 
                        sub_batch_size = 2500
                        num_sub_batches = (batch_tensor.shape[0] + sub_batch_size - 1)//sub_batch_size
                        
                        current_batch_total_loss = 0.0
                        for i in range(num_sub_batches):
                            start_idx = i*sub_batch_size
                            end_idx = min(start_idx + sub_batch_size, batch_tensor.shape[0])
                            sub_batch = batch_tensor[start_idx:end_idx]
                            
                            loss_val_tensor = self.train_step(loss_function, sub_batch)
                            current_batch_total_loss += loss_val_tensor.item()*sub_batch.shape[0] 
                        epoch_loss_sum += current_batch_total_loss/batch_tensor.shape[0]
                        
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache() 
                        gc.collect()
                    else:
                        loss_val_tensor = self.train_step(loss_function, batch_tensor)
                        epoch_loss_sum += loss_val_tensor.item()
                    num_processed_batches +=1

                except RuntimeError as e: 
                    print(f"Error in batch {batch_idx+1}/{len(current_batches)}: {e}")
                    if "CUDA out of memory" in str(e):
                        print("CUDA OOM. Try reducing batch size or model complexity.")
                     
                        if self.device.type == 'cuda': torch.cuda.empty_cache()
                        gc.collect()
                    print("Skipping batch and continuing...")
                    continue 
            
            epoch_avg_loss = epoch_loss_sum/num_processed_batches if num_processed_batches > 0 else float('inf')
            epoch_time = time.time() - epoch_start_time
            total_training_time += epoch_time

            if epoch_avg_loss < last_loss - min_delta:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            last_loss = epoch_avg_loss

            if (epoch + 1) % print_interval == 0:
                loss_history.append(epoch_avg_loss)
                epoch_history.append(epoch + 1)
                time_history.append(epoch_time)
                avg_time_so_far = total_training_time/(epoch + 1)

                if plot_loss and live_fig is not None:
                    live_line1.set_xdata(epoch_history)
                    live_line1.set_ydata(loss_history)
                    live_ax1.relim()
                    live_ax1.autoscale_view()
                    
                    live_line2.set_xdata(epoch_history)
                    live_line2.set_ydata(time_history)
                    live_ax2.relim()
                    live_ax2.autoscale_view()
                    
                    live_fig.canvas.draw()
                    live_fig.canvas.flush_events()
                    plt.pause(0.001)

                print(f"Epoch {epoch + 1}: LR = {current_lr:.6f}, Loss = {epoch_avg_loss:.6f}, Time = {epoch_time:.2f}s, Avg Time = {avg_time_so_far:.2f}s")
            
            if self.boundary_visualizer is not None and bc_plot_interval is not None and (epoch + 1) % bc_plot_interval == 0:
                self.boundary_visualizer.plot_boundary_conditions(self, mesh, epoch + 1, domain_range, airfoil_coords)

            if (epoch + 1) % autosave_interval == 0:
                self.save(f"{self.eq}_epoch_{epoch+1}") 

            self.scheduler.step(epoch_avg_loss)
            if self.optimizer.param_groups[0]['lr'] < current_lr:
                print(f"Epoch {epoch + 1}: Learning rate reduced to {self.optimizer.param_groups[0]['lr']:.6f}")


        final_avg_time = total_training_time/(epoch + 1) if epoch >=0 else 0
        print(f"\nTraining completed (or stopped early):")
        print(f"Total training time: {total_training_time:.2f}s")
        if epoch >=0 : print(f"Average time per epoch: {final_avg_time:.2f}s")
        if time_history:
             print(f"Min epoch time: {min(time_history):.2f}s")
             print(f"Max epoch time: {max(time_history):.2f}s")

        if self.boundary_visualizer is not None:
            self.boundary_visualizer.plot_error_evolution()

        if plot_loss and live_fig is not None:
            plt.ioff()
            plt.close(live_fig)

        training_history = {'loss': loss_history,
            'learning_rate': lr_history,
            'time_per_epoch': time_history,
            'total_time': total_training_time,
            'avg_time_per_epoch': final_avg_time}

        return training_history


    def predict(self, X: Any, use_cpu: bool = False) -> np.ndarray: 
        self.model.eval() 
        current_device = torch.device("cpu") if use_cpu else self.device
        self.to(current_device) 
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.to(dtype=torch.float32)
        else:
            raise TypeError("Input X must be a NumPy array or PyTorch tensor.")

        X_tensor = X_tensor.to(current_device)
        
        with torch.no_grad(): 
            predictions_tensor = self.model(X_tensor)
        return predictions_tensor.cpu().numpy()

    def save(self, model_name_prefix: str) -> bool:
        """ Saves the model state_dict and metadata. """
        save_dir = 'trainedModels'
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"{model_name_prefix}.pth")
        metadata_path = os.path.join(save_dir, f"{model_name_prefix}_metadata.json")
        
        try:
            torch.save(self.model.state_dict(), model_path)
            
            metadata = {'activation_name': self.activation_name,
                'learning_rate': self.learning_rate, 
                'input_dim': self.input_dim,
                'output_shape': self.output_shape,
                'hidden_layers_config': self.hidden_layers_config,
                'eq': self.eq,
                'architecture_type': 'pinn_pytorch', 
                'model_version': '1.0_pytorch'}
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Model successfully saved to {model_path}")
            print(f"Model metadata saved to {metadata_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load(self, model_name_prefix: str, map_location:Optional[str]=None) -> None: 
        """ Loads the model state_dict and metadata. """
        import json
        load_dir = 'trainedModels'
        model_path = os.path.join(load_dir, f"{model_name_prefix}.pth")
        metadata_path = os.path.join(load_dir, f"{model_name_prefix}_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The specified model file does not exist: {model_path}")

        try:
           
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.activation_name = metadata.get('activation_name', self.activation_name)
                self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                self.input_dim = metadata.get('input_dim', self.input_dim)
                self.output_shape = metadata.get('output_shape', self.output_shape)
                self.hidden_layers_config = metadata.get('hidden_layers_config', self.hidden_layers_config)
                self.eq = metadata.get('eq', self.eq if self.eq else model_name_prefix)
                print(f"Model metadata loaded from {metadata_path}")
            else:
                print("Warning: No metadata file found. Model architecture might not match if __init__ defaults changed.")
             
           
            self.model = self.create_model(self.input_dim, self.output_shape, self.hidden_layers_config, self.activation_name)
            
            if map_location is None:
                map_location = self.device.type 
            
            self.model.load_state_dict(torch.load(model_path, map_location=map_location))
            self.model.to(self.device)
            self.model.eval() 
            
         
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) 
            self.scheduler = self._create_learning_rate_schedule(self.optimizer, self.learning_rate)

            print(f"Model successfully loaded from {model_path} to device {self.device}")

        except Exception as e:
            print(f"Error loading model: {e}")
            if "CUDA" in str(e).upper(): # basic check for cuda related errors
                print("Trying to load model on CPU...")
                try:
                    self.model = self.create_model(self.input_dim, self.output_shape, self.hidden_layers_config, self.activation_name)
                    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    self.device = torch.device('cpu')
                    self.model.to(self.device)
                    self.model.eval()
                    
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                    self.scheduler = self._create_learning_rate_schedule(self.optimizer, self.learning_rate)
                    print(f"Model successfully loaded on CPU from {model_path}")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model on CPU: {e2}")
            else:
                raise RuntimeError(f"Failed to load model: {e}")