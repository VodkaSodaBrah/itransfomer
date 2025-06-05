"""
Module for advanced MPS (Metal Performance Shaders) optimizations for PyTorch models.
This includes model compilation utilities, custom MPS operations, and chip-specific
tuning for Apple M1/M2/M3 series chips.
"""
import os
import sys
import torch
import platform
import subprocess
import time
import importlib
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any

# Check if we're running on macOS
is_mac = platform.system() == 'Darwin'

# Try to import psutil for memory monitoring
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False

def detect_mac_chip_details() -> Dict[str, Any]:
    """
    Get detailed information about the Mac's M-series chip.
    
    Returns:
        dict: Dictionary with chip details including:
            - chip_type: String like "M1", "M2", "M3"
            - is_pro: Boolean if it's a Pro model
            - is_max: Boolean if it's a Max model
            - is_ultra: Boolean if it's an Ultra model
            - core_count: Number of performance cores if available
            - memory_gb: System memory in GB if available
    """
    if not is_mac:
        return {
            "is_apple_silicon": False,
            "chip_type": "Unknown",
            "is_pro": False,
            "is_max": False, 
            "is_ultra": False,
            "core_count": None,
            "memory_gb": None
        }
    
    try:
        # Get chip brand string
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], 
            capture_output=True, text=True, check=True
        )
        chip_info = result.stdout.strip()
        
        # Get memory info
        memory_gb = None
        if has_psutil:
            memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Parse the chip info
        is_apple_silicon = "Apple M" in chip_info
        
        # Detect chip type
        chip_type = "Unknown"
        if "Apple M1" in chip_info:
            chip_type = "M1"
        elif "Apple M2" in chip_info:
            chip_type = "M2"
        elif "Apple M3" in chip_info:
            chip_type = "M3"
        
        # Detect chip variant
        is_pro = "Pro" in chip_info
        is_max = "Max" in chip_info
        is_ultra = "Ultra" in chip_info
        
        # Try to get core count (this may not work on all systems)
        core_count = None
        try:
            core_result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"], 
                capture_output=True, text=True, check=True
            )
            core_count = int(core_result.stdout.strip())
        except:
            pass
        
        return {
            "is_apple_silicon": is_apple_silicon,
            "chip_type": chip_type,
            "is_pro": is_pro,
            "is_max": is_max,
            "is_ultra": is_ultra,
            "core_count": core_count,
            "memory_gb": memory_gb
        }
    except Exception as e:
        print(f"Error detecting Mac chip details: {e}")
        return {
            "is_apple_silicon": True if "Apple" in platform.processor() else False,
            "chip_type": "Unknown",
            "is_pro": False,
            "is_max": False,
            "is_ultra": False,
            "core_count": None,
            "memory_gb": None
        }

def optimize_model_for_chip(
    model: torch.nn.Module,
    chip_details: Dict[str, Any]
) -> torch.nn.Module:
    """
    Apply chip-specific optimizations to the model.
    
    Args:
        model: The PyTorch model to optimize
        chip_details: Dictionary with chip details from detect_mac_chip_details()
        
    Returns:
        Optimized model
    """
    if not is_mac or not torch.backends.mps.is_available():
        return model
    
    # Ensure MPS fallback is enabled for compatibility
    if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Apply optimizations based on chip type
    if chip_details['is_apple_silicon']:
        print(f"Applying optimizations for {chip_details['chip_type']} chip...")
        
        # Enable float16 computations if available
        if hasattr(torch.backends.mps, 'enable_float16_computations'):
            torch.backends.mps.enable_float16_computations(True)
            print("  - Enabled float16 computations for MPS")
        
        # Different compilation strategies based on chip
        if hasattr(torch, 'compile'):
            try:
                # Set different compilation options based on chip
                compile_options = get_optimal_compile_options(chip_details)
                
                print(f"  - Compiling model with backend: {compile_options['backend']}")
                compiled_model = torch.compile(model, **compile_options)
                return compiled_model
            except Exception as e:
                print(f"  - Model compilation failed: {e}")
                print("    Proceeding without compilation")
    
    return model

def get_optimal_compile_options(chip_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the optimal compilation options for the specific Mac chip.
    
    Args:
        chip_details: Dictionary with chip details
        
    Returns:
        Dictionary with torch.compile options
    """
    # Default options for most chips
    options = {
        "backend": "aot_eager",
        "mode": "reduce-overhead",
        "fullgraph": False,
        "options": {
            "allow_rnn": True,
            "allow_dropout": True,
        }
    }
    
    # Adjust based on chip capabilities
    if chip_details["chip_type"] == "M3" or (chip_details["chip_type"] == "M2" and (chip_details["is_pro"] or chip_details["is_max"])):
        # More advanced chips can use more aggressive optimizations
        options["backend"] = "inductor"
        options["mode"] = "max-autotune"
        options["options"]["max_autotune"] = True
    
    # Add more memory for larger chips
    if chip_details["is_max"] or chip_details["is_ultra"]:
        if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.9"
    
    return options

def create_mps_optimized_dropout(p: float = 0.5, inplace: bool = False) -> torch.nn.Module:
    """
    Create a dropout layer that works efficiently on MPS.
    
    This addresses known issues with native_dropout on MPS by providing a custom
    implementation that automatically falls back to a manual implementation
    when running on MPS without CPU fallback enabled.
    
    Args:
        p: Dropout probability
        inplace: Whether to perform dropout in-place
        
    Returns:
        Dropout module compatible with MPS
    """
    class MPSFriendlyDropout(torch.nn.Module):
        def __init__(self, p=0.5, inplace=False):
            super().__init__()
            self.p = p
            self.inplace = inplace
            # Standard PyTorch dropout for non-MPS devices
            self.dropout = torch.nn.Dropout(p=p, inplace=inplace)
            
        def forward(self, x):
            if not self.training or self.p == 0.0:
                return x
                
            device_type = x.device.type
            if device_type == 'mps' and 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
                # Use manual dropout implementation for MPS when fallback isn't enabled
                if self.inplace:
                    mask = torch.rand_like(x) > self.p
                    x.mul_(mask / (1 - self.p))
                    return x
                else:
                    mask = torch.rand_like(x) > self.p
                    return x * mask / (1 - self.p)
            else:
                # Use standard dropout for other devices or when fallback is enabled
                return self.dropout(x)
    
    return MPSFriendlyDropout(p=p, inplace=inplace)

def replace_dropout_layers(model: torch.nn.Module) -> torch.nn.Module:
    """
    Replace all dropout layers in the model with MPS-friendly versions.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model with replaced dropout layers
    """
    if not is_mac or not torch.backends.mps.is_available():
        return model
        
    # Only replace if MPS fallback is not enabled
    if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
        for name, module in list(model.named_children()):
            if isinstance(module, torch.nn.Dropout):
                setattr(model, name, create_mps_optimized_dropout(p=module.p, inplace=module.inplace))
            else:
                # Recursively replace in child modules
                replace_dropout_layers(module)
    
    return model

def optimize_parameters_for_mps(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize model parameters for MPS by ensuring they're in the right format.
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    if not is_mac or not torch.backends.mps.is_available():
        return model
    
    # Ensure parameters are contiguous and in the right format
    with torch.no_grad():
        for param in model.parameters():
            if param.data.dim() > 0:  # Skip scalar parameters
                param.data = param.data.contiguous()
                
    return model

def prepare_model_for_mps(model: torch.nn.Module) -> torch.nn.Module:
    """
    Comprehensive preparation of a model for MPS execution.
    
    This function:
    1. Replaces dropout layers with MPS-friendly versions
    2. Optimizes parameters for MPS
    3. Applies chip-specific optimizations
    
    Args:
        model: PyTorch model
        
    Returns:
        MPS-optimized model
    """
    if not is_mac or not torch.backends.mps.is_available():
        return model
    
    # Get chip details
    chip_details = detect_mac_chip_details()
    
    # Apply each optimization in sequence
    model = replace_dropout_layers(model)
    model = optimize_parameters_for_mps(model)
    model = optimize_model_for_chip(model, chip_details)
    
    return model
