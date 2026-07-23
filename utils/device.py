import torch

# Global device setting. Default to CUDA if available, otherwise CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_device(device_str: str):
    """Set the global device. Call this before loading any models.

    Args:
        device_str: 'cuda', 'cpu', or 'auto' (auto selects cuda if available)
    """
    global DEVICE
    if device_str == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(device_str)
    return DEVICE

def get_device():
    """Get the current global device."""
    return DEVICE
