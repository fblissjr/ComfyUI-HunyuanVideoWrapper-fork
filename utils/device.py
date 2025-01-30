import torch
import comfy.model_management as mm

def get_available_device():
    """
    Returns the most appropriate torch device based on availability.
    """
    if mm.is_device_type_cpu():
        return "cpu"
    return mm.get_torch_device()

def get_autocast_device():
    return mm.get_autocast_device(mm.get_torch_device())

def set_default_device(device):
    """
    Sets the default torch device.
    """
    torch.set_default_device(device)

# other device-related utility functions can be added here