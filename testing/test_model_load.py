import pytest
import os
import sys

# Assuming the structure provided, add the necessary paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

import torch
from core.model_loader import (
    HyVideoModelLoader,
    standardize_lora_key_format,
    filter_state_dict_by_blocks,
)
from core.vae import HyVideoVAELoader
from modules.models import HYVideoDiffusionTransformer
import folder_paths
from comfy.utils import load_torch_file

# Fixtures are defined in conftest.py or a similar appropriately named file
# They will be used in the tests to provide necessary data or objects

def test_model_loader_loads(model_path, base_precision, load_device, quantization, attention_mode, compile_args, block_swap_args, lora, auto_cpu_offload, upcast_rope):
    # Initialize the model loader class
    model_loader = HyVideoModelLoader()

    # Ensure we can load a model without raising any errors
    # Use non-default values that still represent a valid configuration
    loaded_model = model_loader.loadmodel(
        model=model_path,
        base_precision=base_precision,
        load_device=load_device,
        quantization=quantization,
        attention_mode=attention_mode,
        compile_args=compile_args,
        block_swap_args=block_swap_args,
        lora=lora,
        auto_cpu_offload=auto_cpu_offload,
        upcast_rope=upcast_rope
    )

    assert loaded_model is not None, "Model loading failed"
    assert isinstance(loaded_model[0].model, dict)
    assert isinstance(loaded_model[0].model["comfy_model"], HyVideoModelConfig)
    assert isinstance(loaded_model[0].model["diffusion_model"], HYVideoDiffusionTransformer)
    # Add more assertions here to validate the model

def test_vae_loader_loads(vae_model_path, vae_precision, compile_args):
    # Initialize the vae loader class
    vae_loader = HyVideoVAELoader()

    # Ensure we can load a vae without raising any errors
    loaded_vae = vae_loader.loadmodel(
        model_name=vae_model_path,
        precision=vae_precision,
        compile_args=compile_args
    )

    assert loaded_vae is not None, "VAE loading failed"
    assert isinstance(loaded_vae[0], AutoencoderKLCausal3D)
    # Add more assertions here to validate the vae

def test_standardize_lora_key_format(lora_sd):
    standardized_sd = standardize_lora_key_format(lora_sd)
    assert standardized_sd["test_key_that_should_be_loaded"].shape == torch.Size([1, 1])
    assert standardized_sd["test_key_that_should_be_loaded"].dtype == torch.bfloat16
    
    assert isinstance(standardized_sd, dict)

def test_filter_state_dict_by_blocks(valid_sd):
    # blocks mapping should take any key starting with "double_blocks." or "single_blocks.",
    blocks_mapping = {"double_blocks.0.": True, "single_blocks.0.": True}
    filtered_sd = filter_state_dict_by_blocks(valid_sd, blocks_mapping)

    # Check that keys starting with the specified prefixes are present and others are filtered out
    for key in filtered_sd:
        assert key.startswith("double_blocks.0.") or key.startswith("single_blocks.0.")

    for key in valid_sd:
        if not key.startswith("double_blocks.0.") and not key.startswith("single_blocks.0."):
            assert key not in filtered_sd

# ensure that the lora path can be located correctly and that we return a correct path for a dummy lora file
def test_lora_path(lora_name, model_path_for_lora):
    lora_path = get_lora_path(lora_name)
    assert lora_path == os.path.join(folder_paths.models_dir, "loras", lora_name)