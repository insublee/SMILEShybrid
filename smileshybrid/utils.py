from typing import Dict, List, Union
import torch

def to(obj: Union[List, Dict, torch.Tensor], device: str):
    """
    Move object to gpu
    Args:
        obj (Union[List, Dict, torch.Tensor]): object
        device (str): device, one of 'gpu', 'cpu'
    Returns:
        cuda object
    """
    assert device in ["gpu", "cpu"], "device must be one of [gpu, cpu]"

    if device == "gpu":
        device = torch.cuda.current_device()

    if isinstance(obj, list):
        for i in range(len(obj)):
            if torch.is_tensor(obj[i]):
                obj[i] = obj[i].to(device)

    elif isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                obj[k] = v.to(device)

    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)

    return obj
