import os
import tempfile
import shutil  # Import the shutil module
import pytest

# Assuming your ComfyUI directory structure is as follows:
# ComfyUI/
#   models/
#       checkpoints/
#       loras/
#       vae/
#       llm/
#       diffusion_models/
#       hyvid_embeds/
#       vae_configs/
#   custom_nodes/
#       ComfyUI-HunyuanVideoWrapper/

@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown():
    # Create temporary directories
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    loras_dir = os.path.join(models_dir, "loras")
    vae_dir = os.path.join(models_dir, "vae")
    llm_dir = os.path.join(models_dir, "llm")
    diffusion_models_dir = os.path.join(models_dir, "diffusion_models")
    hyvid_embeds_dir = os.path.join(models_dir, "hyvid_embeds")
    vae_configs_dir = os.path.join(models_dir, "vae_configs")
    # Ensure paths are absolute and normalized
    models_dir = os.path.abspath(models_dir)
    checkpoints_dir = os.path.abspath(checkpoints_dir)
    loras_dir = os.path.abspath(loras_dir)
    vae_dir = os.path.abspath(vae_dir)
    llm_dir = os.path.abspath(llm_dir)
    diffusion_models_dir = os.path.abspath(diffusion_models_dir)
    hyvid_embeds_dir = os.path.abspath(hyvid_embeds_dir)
    vae_configs_dir = os.path.abspath(vae_configs_dir)
    
    # Setup: Create directories and dummy files
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(loras_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(llm_dir, exist_ok=True)
    os.makedirs(diffusion_models_dir, exist_ok=True)
    os.makedirs(hyvid_embeds_dir, exist_ok=True)
    os.makedirs(vae_configs_dir, exist_ok=True)

    # Create dummy files for testing
    with open(os.path.join(checkpoints_dir, "dummy.pt"), "w") as f:
        f.write("Dummy checkpoint")
    with open(os.path.join(loras_dir, "dummy.safetensors"), "w") as f:
        f.write("Dummy lora")
    with open(os.path.join(vae_dir, "dummy.safetensors"), "w") as f:
        f.write("Dummy vae")
    with open(os.path.join(llm_dir, "dummy"), "w") as f:
        f.write("Dummy llm")
    with open(os.path.join(diffusion_models_dir, "dummy.safetensors"), "w") as f:
        f.write("Dummy diffusion model")
    with open(os.path.join(hyvid_embeds_dir, "dummy.safetensors"), "w") as f:
        f.write("Dummy hyvid_embeds")
    with open(os.path.join(vae_configs_dir, "dummy.json"), "w") as f:
        f.write("Dummy vae config")

    yield  # This is where the testing happens

    # Teardown: Remove the directories and files
    shutil.rmtree(checkpoints_dir)
    shutil.rmtree(loras_dir)
    shutil.rmtree(vae_dir)
    shutil.rmtree(llm_dir)
    shutil.rmtree(diffusion_models_dir)
    shutil.rmtree(hyvid_embeds_dir)
    shutil.rmtree(vae_configs_dir)