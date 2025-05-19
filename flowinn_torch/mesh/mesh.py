
"""
author: Jon Errasti Odriozola
github-id: https://github.com/errasti13
"""
import numpy as np
from typing import Dict, Optional, Tuple
from flowinn_torch.mesh.meshio import MeshIO
from flowinn_torch.mesh.visualizer import MeshVisualizer
from flowinn_torch.mesh.sampler import Sampler 
from flowinn_torch.mesh.boundary_handler import BoundaryConditionHandler as bc

class Mesh:
    """
    A class for generating and managing computational meshes.

    Attributes:
        is2D (bool): Flag indicating whether the mesh is 2D or 3D.
        meshio (MeshIO): Instance of MeshIO for handling mesh I/O operations.
    """

    def __init__(self, is2D: bool = True) -> None:
        """
        Initializes a new Mesh object.

        Args:
            is2D (bool): Flag indicating whether the mesh is 2D or 3D. Defaults to True.
        """
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None
        self._t: Optional[np.ndarray] = None
        self._solutions: Dict[str, np.ndarray] = {}
        self._boundaries: Dict[str, Dict[str, np.ndarray]] = {}
        self._interiorBoundaries: Dict[str, Dict[str, np.ndarray]] = {}
        self._periodicBoundaries: Dict[str, Dict[str, np.ndarray]] = {}
        self._initialConditions: Dict[str, np.ndarray] = {}
        self._is2D: bool = is2D
        self.meshio: Optional[MeshIO] = None
        self.is_unsteady = False

    def _create_meshio(self) -> None:
        """
        Creates a MeshIO instance if it doesn't exist.
        """
        if self.meshio is None:
            self.meshio = MeshIO(self)

    @property
    def x(self) -> Optional[np.ndarray]:
        """
        Returns the x-coordinates of the mesh points.
        """
        return self._x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        """
        Sets the x-coordinates of the mesh points.

        Args:
            value (np.ndarray): A numpy array containing the x-coordinates.

        Raises:
            TypeError: If value is not a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("x must be a numpy array")
        self._x = value

    @property
    def y(self) -> Optional[np.ndarray]:
        """
        Returns the y-coordinates of the mesh points.
        """
        return self._y

    @y.setter
    def y(self, value: np.ndarray) -> None:
        """
        Sets the y-coordinates of the mesh points.

        Args:
            value (np.ndarray): A numpy array containing the y-coordinates.

        Raises:
            TypeError: If value is not a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("y must be a numpy array")
        self._y = value

    @property
    def z(self) -> Optional[np.ndarray]:
        """
        Returns the z-coordinates of the mesh points.
        """
        return self._z

    @z.setter
    def z(self, value: np.ndarray) -> None:
        """
        Sets the z-coordinates of the mesh points.

        Args:
            value (np.ndarray): A numpy array containing the z-coordinates.

        Raises:
            TypeError: If value is not a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("z must be a numpy array")
        self._z = value

    @property
    def t(self) -> Optional[np.ndarray]:
        """
        Returns the time values of the mesh points.
        """
        return self._t
    
    @t.setter
    def t(self, value: np.ndarray) -> None:
        """
        Sets the time values of the mesh points.

        Args:
            value (np.ndarray): A numpy array containing the time values.

        Raises:
            TypeError: If value is not a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("t must be a numpy array")
        self._t = value

    @property
    def solutions(self) -> Dict[str, np.ndarray]:
        """
        Returns the solutions dictionary.
        """
        return self._solutions

    @solutions.setter
    def solutions(self, value: Dict[str, np.ndarray]) -> None:
        """
        Sets the solutions dictionary.

        Args:
            value (dict): A dictionary containing solution data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("solutions must be a dictionary")
        self._solutions = value

    @property
    def boundaries(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the boundaries dictionary.
        """
        return self._boundaries

    @boundaries.setter
    def boundaries(self, value: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Sets the boundaries dictionary.

        Args:
            value (dict): A dictionary containing boundary data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("boundaries must be a dictionary")
        self._boundaries = value

    @property
    def interiorBoundaries(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the interior boundaries dictionary.
        """
        return self._interiorBoundaries

    @interiorBoundaries.setter
    def interiorBoundaries(self, value: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Sets the interior boundaries dictionary.

        Args:
            value (dict): A dictionary containing interior boundary data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("interiorBoundaries must be a dictionary")
        self._interiorBoundaries = value

    @property
    def periodicBoundaries(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the interior boundaries dictionary.
        """
        return self._periodicBoundaries

    @periodicBoundaries.setter
    def periodicBoundaries(self, value: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Sets the periodic boundaries dictionary.

        Args:
            value (dict): A dictionary containing interior boundary data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("interiorBoundaries must be a dictionary")
        self._periodicBoundaries = value
    
    @property
    def initialConditions(self) -> Dict[str, np.ndarray]:
        """
        Returns the initial conditions dictionary.
        """
        return self._initialConditions
    
    @initialConditions.setter
    def initialConditions(self, value: Dict[str, np.ndarray]) -> None:
        """
        Sets the initial conditions dictionary.

        Args:
            value (dict): A dictionary containing initial condition data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("initialConditions must be a dictionary")
        self._initialConditions = value

    @property
    def is2D(self) -> bool:
        """
        Returns the is2D flag.
        """
        return self._is2D

    @is2D.setter
    def is2D(self, value: bool) -> None:
        """
        Sets the is2D flag.

        Args:
            value (bool): A boolean value indicating if the mesh is 2D.

        Raises:
            TypeError: If value is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError("is2D must be a boolean")
        self._is2D = value

    def generateMesh(self, Nx: int = 100, Ny: int = 100, Nz: Optional[int] = None, sampling_method: str = 'random') -> None:
        """
        Generates a mesh within a domain defined by boundary data.

        Args:
            Nx (int): Number of points in the x-dimension for structured sampling. Defaults to 100.
            Ny (int): Number of points in the y-dimension for structured sampling. Defaults to 100.
            Nz (Optional[int]): Number of points in the z-dimension for structured sampling. Defaults to None.
            sampling_method (str): Sampling method ('random', 'uniform'). Defaults to 'random'.

        Raises:
            ValueError: If input parameters are invalid or mesh generation fails.
        """
        if not self.boundaries:
            raise ValueError("No boundaries defined. Use setBoundary() to define boundaries before generating mesh")

        try:
            self._generateMeshFromBoundary(sampling_method, Nx, Ny, Nz)
        except Exception as e:
            raise ValueError(f"Mesh generation failed: {str(e)}")

    def _generateMeshFromBoundary(self, sampling_method: str, Nx: int, Ny: int, Nz: Optional[int]) -> None:
        """
        Generates the mesh from the defined boundaries using the specified sampling method.

        Args:
            sampling_method (str): Sampling method ('random', 'uniform').
            Nx (int): Number of points in the x-dimension.
            Ny (int): Number of points in the y-dimension.
            Nz (Optional[int]): Number of points in the z-dimension.

        Raises:
            ValueError: If boundary data is invalid or sampling method is unsupported.
        """
        for boundary_name, boundary_data in self.boundaries.items():
            if 'x' not in boundary_data or 'y' not in boundary_data:
                raise ValueError(f"Boundary '{boundary_name}' must contain 'x' and 'y' coordinates.")
            if not self.is2D and 'z' not in boundary_data:
                raise ValueError(f"3D mesh requires z coordinate for boundary {boundary_name}")

        try:
            x_boundary = np.concatenate([np.asarray(boundary_data['x'], dtype=np.float32).flatten()
                                         for boundary_data in self.boundaries.values()])
            y_boundary = np.concatenate([np.asarray(boundary_data['y'], dtype=np.float32).flatten()
                                         for boundary_data in self.boundaries.values()])

            if not self.is2D:
                z_boundary = np.concatenate([np.asarray(boundary_data['z'], dtype=np.float32).flatten()
                                             for boundary_data in self.boundaries.values()])
            else:
                z_boundary = None

        except Exception as e:
            print(f"Debug: Error during boundary concatenation: {str(e)}")
            raise

        if sampling_method == 'random':
            Sampler._sampleRandomlyWithinBoundary(self, x_boundary, y_boundary, z_boundary, Nx, Ny, Nz)
        elif sampling_method == 'uniform':
            Sampler._sampleUniformlyWithinBoundary(self, x_boundary, y_boundary, z_boundary, Nx, Ny, Nz)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")
        
    def generate_time_discretization(self, Nt: int = 100, t_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        """
        Generates a time mesh.

        Args:
            Nt (int): Number of time points. Defaults to 100.
            t_range (Tuple[float, float]): Time range. Defaults to (0.0, 1.0).
        """
        if t_range is None:
            self._t = None
            self.is_unsteady = False
            return
            
        self._t = np.linspace(t_range[0], t_range[1], Nt)
        self.is_unsteady = True
        return


    def setBoundary(self, boundary_name: str, xBc: np.ndarray, yBc: np.ndarray, interior: bool = False,
                    **boundary_conditions: Dict[str, np.ndarray]) -> None:
        """
        Sets multiple boundary conditions at once for non-periodic boundaries.

        Args:
            boundary_name (str): Name of the boundary.
            xBc (np.ndarray): x-coordinates of the boundary.
            yBc (np.ndarray): y-coordinates of the boundary.
            interior (bool): Flag indicating if this is an interior boundary. Defaults to False.
            **boundary_conditions (Dict[str, np.ndarray]): Variable names and their values.
        """
        for var_name, values in boundary_conditions.items():
            bc.setBoundaryCondition(self, xBc, yBc, values, var_name, boundary_name, interior=interior)

    def setPeriodicBoundary(self, boundary_name: str, coupled_boundary: str) -> None:
        """
        Sets a periodic boundary pair.

        Args:
            boundary_name (str): Name of the periodic boundary.
            coupled_boundary (str): Name of the boundary to which this boundary is coupled.
        """
        bc.setPeriodicBoundary(self, boundary_name, coupled_boundary)

    def showMesh(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Displays the mesh visualization.
        
        Args:
            figsize (Tuple[int, int]): Size of the figure. Defaults to (8, 6).
        """
        visualizer = MeshVisualizer()
        visualizer.showMesh(self)


    def write_tecplot(self, filename: str) -> None:
        """
        Writes the solution to a Tecplot file using MeshIO.

        Args:
            filename (str): The name of the Tecplot file to write.
        """
        self._create_meshio()
        self.meshio.write_tecplot(filename)

    def print_mesh_info(self) -> None:
        """
        Prints information about the mesh.
        """
        print("Mesh information:")
        print(f'Number of points in x-direction: {len(self.x)}')
        print(f'Min and max x-coordinates: {np.min(self.x)} and {np.max(self.x)}')
        print(f'Number of points in y-direction: {len(self.y)}')
        print(f'Min and max y-coordinates: {np.min(self.y)} and {np.max(self.y)}')
        if self.is2D:
            pass
        else:
            print(f'Number of points in z-direction: {len(self.z)}')
            print(f'Min and max z-coordinates: {np.min(self.z)} and {np.max(self.z)}')
        if self.t is not None:
            print(f'Number of time points: {len(self.t)}')
            print(f'Min and max time values: {np.min(self.t)} and {np.max(self.t)}')

