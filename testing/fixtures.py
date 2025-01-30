# comfyui_hunyuanvideowrapper/testing/fixtures.py
import pytest
import torch
import os

# Determine the absolute path to the ComfyUI directory
comfyui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Construct the path to the checkpoints directory
ckpts_path = os.path.join(comfyui_path, "models", "checkpoints")

# Construct the path to the checkpoints directory
models_path = os.path.join(comfyui_path, "models")
vae_path = os.path.join(comfyui_path, "models", "vae")

@pytest.fixture
def model_path():
    return "your_model.safetensors"  # Replace with your actual model filename

@pytest.fixture
def base_precision():
    return "bf16"

@pytest.fixture
def load_device():
    return "main_device"

@pytest.fixture
def quantization():
    return "disabled"

@pytest.fixture
def attention_mode():
    return "sdpa"

@pytest.fixture
def compile_args():
    return None

@pytest.fixture
def block_swap_args():
    return None

@pytest.fixture
def lora():
    return None

@pytest.fixture
def auto_cpu_offload():
    return False

@pytest.fixture
def upcast_rope():
    return True

@pytest.fixture
def vae_model_path():
    return "hy_vae_config.json"  # replace with an actual model if possible, using a config for now

@pytest.fixture
def vae_precision():
    return "bf16"

@pytest.fixture
def valid_sd():
    return {
        "test_key_that_should_be_loaded": torch.zeros(1, 1, dtype=torch.bfloat16)
    }

@pytest.fixture
def model_path_for_lora():
    return "your_model_path_lora"

@pytest.fixture
def lora_name():
    return "test_lora.safetensors"