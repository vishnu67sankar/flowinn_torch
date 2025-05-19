
"""
Author of the tensorflow repo is Jon Errasti Odriozola (tensorflow) | github-id: https://github.com/errasti13
Vishnu Sankar has converted tf to pytorch and re-built the same package but in pytroch
"""

from typing import Any
from flowinn_torch.training.steady_loss import SteadyNavierStokesLoss
from flowinn_torch.training.unsteady_loss import UnsteadyNavierStokesLoss

class NavierStokesLoss:
    def __init__(self, loss_type: str = 'steady', mesh: Any = None, model: Any = None, **kwargs: Any) -> None:
        """
        Initialize the NavierStokesLoss instance.
        Note: The primary factory logic is in __new__ and create.
        This __init__ is called if a direct instance of NavierStokesLoss is created
        (e.g., when mesh or model is None in the __new__ method's fallback).
        """
        if mesh is not None and model is not None:
            return self.create(loss_type, mesh, model, **kwargs)
        return None
        
    @staticmethod
    def create(loss_type: str, mesh: Any, model: Any, **kwargs: Any) -> Any: # Return type Any for flexibility
        """
        Factory method to create appropriate loss function.
        
        Returns:
            A callable loss function object
        """
        if loss_type.lower() == 'steady':
            loss_obj = SteadyNavierStokesLoss(mesh, model, **kwargs)
        
        elif loss_type.lower() == 'unsteady':
            loss_obj = UnsteadyNavierStokesLoss(mesh, model, **kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        if not callable(loss_obj):
            original_loss_obj = loss_obj
            
            class CallableLoss:
                def __init__(self, loss_obj_inner: Any): 
                    self.loss_obj = loss_obj_inner
                    if hasattr(self.loss_obj, 'loss_function'):
                        self.loss_function = self.loss_obj.loss_function
                    
                def __call__(self, batch_data=None) -> Any:
                    if hasattr(self, 'loss_function') and callable(self.loss_function):
                        return self.loss_function(batch_data)
                    elif hasattr(self.loss_obj, 'loss_function') and callable(self.loss_obj.loss_function):
                        return self.loss_obj.loss_function(batch_data)
                    else:
                        raise AttributeError(f"Loss object {type(self.loss_obj).__name__} or its 'loss_function' attribute is not callable or does not exist.")
                        
            loss_obj = CallableLoss(original_loss_obj)
            
        return loss_obj

    def __new__(cls, loss_type: str = 'steady', mesh: Any = None, model: Any = None, **kwargs: Any) -> Any:
        """Override new to return the actual loss object"""
        if mesh is not None and model is not None:
            return cls.create(loss_type, mesh, model, **kwargs)
        return super().__new__(cls)