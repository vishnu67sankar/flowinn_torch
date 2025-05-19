
"""
Author of the tensorflow repo is Jon Errasti Odriozola (tensorflow) | github-id: https://github.com/errasti13
Vishnu Sankar has converted tf to pytorch and re-built the same package but in pytroch
"""

import numpy as np
import logging
from typing import Tuple, List
from flowinn_torch.mesh.mesh import Mesh
from flowinn_torch.models.model import PINN
from flowinn_torch.training.loss import NavierStokesLoss
from flowinn_torch.plot.plot import Plot
from flowinn_torch.physics.boundary_conditions import InletBC, OutletBC, WallBC 
import matplotlib.pyplot as plt
import os

class FlowOverAirfoil:
    def __init__(self, caseName: str, xRange: Tuple[float, float], yRange: Tuple[float, float], AoA: float = 0.0, model_type: str = 'conv1d_residual_net', layer_size: List[int] = [20, 40, 60, 40, 20], drop_out=0.0):
        """
        Initialize FlowOverAirfoil simulation.
        
        Args:
            caseName: Name of the simulation case
            xRange: Tuple of (min_x, max_x) domain bounds
            yRange: Tuple of (min_y, max_y) domain bounds
            AoA: Angle of attack in degrees
        """
        if not isinstance(caseName, str):
            raise TypeError("caseName must be a string")
        if not all(isinstance(x, (int, float)) for x in [*xRange, *yRange, AoA]):
            raise TypeError("xRange, yRange and AoA must be numeric")
            
        self.logger = logging.getLogger(__name__)
        self.is2D = True
        self.problemTag = caseName

        self.mesh = Mesh(self.is2D)
        self.model = PINN(input_shape=(2,), output_shape=3, eq = self.problemTag, layers=layer_size, model_type=model_type, drop_out=drop_out)

        self.loss = None
        self.Plot = None
        
        self.xRange = xRange
        self.yRange = yRange

        self.AoA = AoA
        self.c   = 1.0 
        self.x0  = 0.0 
        self.y0  = 0.0 

        self.generate_airfoil_coords()

        self.u_ref = 1.0
        self.P_inf = 101325 
        self.rho = 1.225 
        self.nu = 0.1 
        
        self.u_inf = float(self.u_ref*np.cos(np.radians(self.AoA)))
        self.v_inf = float(self.u_ref*np.sin(np.radians(self.AoA)))

        self.Re = (xRange[1] - xRange[0])*self.u_ref/self.nu 
        print(f"Reynolds number: {self.Re}")

        self.L_ref = xRange[1] - xRange[0] 
        self.normalize_data()
        self.inlet_bc = InletBC("inlet")
        self.outlet_bc = OutletBC("outlet")
        self.wall_bc = WallBC("wall")

        return
    
    def normalize_data(self):
        """
        Normalize the airfoil coordinates and freestream velocity.
        """
        self.xAirfoil = (self.xAirfoil)/self.L_ref
        self.yAirfoil = (self.yAirfoil)/self.L_ref

        self.xRange = (self.xRange[0]/self.L_ref, self.xRange[1]/self.L_ref)
        self.yRange = (self.yRange[0]/self.L_ref, self.yRange[1]/self.L_ref)

        self.u_inf = self.u_inf/self.u_ref
        self.v_inf = self.v_inf/self.u_ref

        self.P_inf = self.P_inf/(self.rho*self.u_ref**2)

        return
    
    def generate_airfoil_coords(self, N=100, thickness=0.12):
        """
        Generate NACA 4-digit airfoil coordinates for both upper and lower surfaces.
        The lower surface is just the negative of the upper surface.
        """
        x = np.linspace(self.x0, self.x0 + self.c, N)
        x_normalized = x/self.c
        
        y_upper = self.y0 + 5*thickness*(0.2969*np.sqrt(x_normalized) - 0.1260*x_normalized - 0.3516*x_normalized**2 + 0.2843*x_normalized**3 - 0.1015*x_normalized**4)
        y_lower = self.y0 - y_upper
        self.xAirfoil = np.concatenate([x, np.flip(x)]).reshape(-1, 1)
        self.yAirfoil = np.concatenate([y_upper, np.flip(y_lower)]).reshape(-1, 1)
        return
    
    def generateMesh(self, Nx: int = 100, Ny: int = 100, NBoundary: int = 100, sampling_method: str = 'random'):
        try:
            if not all(isinstance(x, int) and x > 0 for x in [Nx, Ny, NBoundary]):
                raise ValueError("Nx, Ny, and NBoundary must be positive integers")
            if sampling_method not in ['random', 'uniform']:
                raise ValueError("sampling_method must be 'random' or 'uniform'")
                
            self._initialize_boundaries()
            x_top = np.linspace(self.xRange[0], self.xRange[1], NBoundary)
            y_top = np.full_like(x_top, self.yRange[1])
            
            x_bottom = np.linspace(self.xRange[0], self.xRange[1], NBoundary)
            y_bottom = np.full_like(x_bottom, self.yRange[0])
            
            x_inlet = np.full(NBoundary, self.xRange[0])
            y_inlet = np.linspace(self.yRange[0], self.yRange[1], NBoundary)
            
            x_outlet = np.full(NBoundary, self.xRange[1])
            y_outlet = np.linspace(self.yRange[0], self.yRange[1], NBoundary)
            
            x_airfoil = self.xAirfoil.flatten()
            y_airfoil = self.yAirfoil.flatten()

            all_coords = [x_top, y_top, x_bottom, y_bottom, x_inlet, y_inlet, x_outlet, y_outlet, x_airfoil, y_airfoil]
            
            if any(np.any(np.isnan(coord)) for coord in all_coords):
                raise ValueError("NaN values detected in boundary coordinates")

            for name, coords in [('top', (x_top, y_top)), ('bottom', (x_bottom, y_bottom)), ('Inlet', (x_inlet, y_inlet)), ('Outlet', (x_outlet, y_outlet))]:
                if name not in self.mesh.boundaries:
                    raise KeyError(f"Boundary '{name}' not initialized")

                self.mesh.boundaries[name].update({
                    'x': coords[0].astype(np.float32),
                    'y': coords[1].astype(np.float32)})

            if 'Airfoil' not in self.mesh.interiorBoundaries:
                raise KeyError("Airfoil boundary not initialized")
            
            self.mesh.interiorBoundaries['Airfoil'].update({
                'x': x_airfoil.astype(np.float32),
                'y': y_airfoil.astype(np.float32)})

            self._validate_boundary_conditions()
            self.mesh.generateMesh(Nx=Nx, Ny=Ny, sampling_method=sampling_method)
            
        except Exception as e:
            self.logger.error(f"Mesh generation failed: {str(e)}")
            raise

    def _initialize_boundaries(self):
        """Initialize boundaries with proper BC system."""

        self.mesh.boundaries = {'Inlet': { 'x': None, 'y': None,
        'conditions': {'u': {'value': self.u_inf}, 'v': {'value': self.v_inf}, 'p': {'gradient': 0.0, 'direction': 'x'}}, 'bc_type': self.inlet_bc},
            'Outlet': {'x': None, 'y': None, 'conditions': {'u': None, 'v': None, 'p': None}, 'bc_type': self.outlet_bc},
            'top': {'x': None, 'y': None, 'conditions': {'u': {'value': self.u_inf}, 'v': {'value': self.v_inf}, 'p': {'gradient': 0.0, 'direction': 'y'}}, 'bc_type': self.wall_bc},
            'bottom': {'x': None, 'y': None, 'conditions': { 'u': {'value': self.u_inf}, 'v': {'value': self.v_inf}, 'p': {'gradient': 0.0, 'direction': 'y'}}, 'bc_type': self.wall_bc}}
        
        self.mesh.interiorBoundaries = {'Airfoil': {'x': None,'y': None, 'conditions': { 'u': {'value': 0.0},  # no-slip condition
                    'v': {'value': 0.0},  # no-slip condition
                    'p': {'gradient': 0.0, 'direction': 'normal'}  # zero pressure gradient normal to wall
                },
                'bc_type': self.wall_bc,
                'isInterior': True  # mark as interior boundary
            }
        }

    def _validate_boundary_conditions(self):
        """Validate boundary conditions before mesh generation."""
        for name, boundary in self.mesh.boundaries.items():
            if any(key not in boundary for key in ['x', 'y', 'conditions', 'bc_type']):
                raise ValueError(f"Missing required fields in boundary {name}")
            if boundary['x'] is None or boundary['y'] is None:
                raise ValueError(f"Coordinates not set for boundary {name}")

        for name, boundary in self.mesh.interiorBoundaries.items():
            if any(key not in boundary for key in ['x', 'y', 'conditions', 'bc_type']):
                raise ValueError(f"Missing required fields in interior boundary {name}")
            if boundary['x'] is None or boundary['y'] is None:
                raise ValueError(f"Coordinates not set for interior boundary {name}")

    def getLossFunction(self):
        self.loss = NavierStokesLoss('steady', self.mesh, self.model, Re =self.Re)
    
    def train(self, epochs=10000, print_interval=100, autosaveInterval=10000, num_batches=10):
        self.getLossFunction()
        self.model.train(self.loss.loss_function,
            self.mesh,
            epochs=epochs,
            print_interval=print_interval,
            autosave_interval=autosaveInterval,
            num_batches=num_batches,
            patience=1000,
            min_delta=1e-4)

    def predict(self) -> None:
        """Predict flow solution and generate plots."""
        try:
            X = (np.hstack((self.mesh.x.flatten()[:, None], self.mesh.y.flatten()[:, None])))
            sol = self.model.predict(X)

            self.mesh.solutions['u'] = sol[:, 0]
            self.mesh.solutions['v'] = sol[:, 1]
            self.mesh.solutions['p'] = sol[:, 2]

            self.mesh.solutions['vMag'] = np.sqrt(
                self.mesh.solutions['u']**2 + 
                self.mesh.solutions['v']**2)

            self.generate_plots()
            self.write_solution()

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def write_solution(self, filename=None):
        """Write the solution to a CSV format file."""
        if filename is None:
            filename = f"{self.problemTag}_AoA_{self.AoA:.1f}.csv"
        elif not filename.endswith('.csv'):
            filename += '.csv'
        
        try:
            if any(key not in self.mesh.solutions for key in ['u', 'v', 'p']):
                raise ValueError("Missing required solution components (u, v, p)")
            
            self.mesh.write_tecplot(filename)
            
        except Exception as e:
            print(f"Error writing solution: {str(e)}")
            raise
    
    def generate_plots(self):
        """Initialize plotting object with custom styling."""
        self.Plot = Plot(self.mesh)
        self.Plot.set_style(
            figsize=(12, 8),
            cmap='viridis',
            scatter_size=15,
            scatter_alpha=0.7,
            font_size=12,
            title_size=14,
            colorbar_label_size=10,
            axis_label_size=11)

    def plot(self, solkey='u', plot_type='default', **kwargs):
        """
        Create various types of plots for the solution fields.
        
        Args:
            solkey (str): Solution field to plot ('u', 'v', 'p', 'vMag')
            plot_type (str): Type of plot to create:
                - 'default': Scatter plot with boundaries
                - 'quiver': Vector field plot showing flow direction
                - 'slices': Multiple y-plane slices (3D only)
            **kwargs: Additional plotting parameters
        """
        if not hasattr(self, 'Plot'):
            self.generate_plots()

        if plot_type == 'quiver':
            self.Plot.vectorField(xRange=self.xRange, yRange=self.yRange, **kwargs)
        
        elif plot_type == 'default':
            if solkey not in self.mesh.solutions:
                available_keys = list(self.mesh.solutions.keys())
                raise ValueError(
                    f"Invalid solution key '{solkey}'. "
                    f"Available keys: {available_keys}. "
                    f"Note: 'vMag' is automatically calculated during prediction.")
            self.Plot.scatterPlot(solkey, show=True)
        
        elif plot_type == 'slices':
            if self.mesh.is2D:
                raise ValueError("Slice plotting is only available for 3D meshes")
            self.Plot.plotSlices(solkey, **kwargs)
        else:
            raise ValueError(f"Invalid plot type: {plot_type}")

    def export_plots(self, directory="results", format=".png"):
        """
        Export all solution field plots to files.
        
        Args:
            directory (str): Directory to save plots
            format (str): File format ('.png', '.pdf', '.svg', etc.)
        """
        os.makedirs(directory, exist_ok=True)
        
        if not all(key in self.mesh.solutions for key in ['u', 'v', 'p', 'vMag']):
            raise ValueError("Missing required solution components. "
                "Make sure to run predict() before exporting plots.")
        
        if not hasattr(self, 'Plot'):
            self.generate_plots()
        
        for solkey in ['u', 'v', 'p', 'vMag']:
            plt.figure()
            self.Plot.scatterPlot(solkey)
            filename = os.path.join(directory, f"{self.problemTag}_AoA{self.AoA}_{solkey}{format}")
            plt.savefig(filename, bbox_inches='tight', dpi=self.Plot.style['dpi'])
            plt.close()
        
        plt.figure(figsize=self.Plot.style['figsize'])
        self.plot(plot_type='quiver')
        filename = os.path.join(directory, f"{self.problemTag}_AoA{self.AoA}_quiver{format}")
        plt.savefig(filename, bbox_inches='tight', dpi=self.Plot.style['dpi'])
        plt.close()

    def load_model(self):
        self.model.load(self.problemTag)
