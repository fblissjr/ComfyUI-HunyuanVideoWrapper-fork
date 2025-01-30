import pytest
import torch

from core.vae import HyVideoVAELoader
from modules.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D

@pytest.fixture
def vae_loader():
    return HyVideoVAELoader()

def test_load_vae_success(vae_loader, vae_model_path, config_path, vae_precision):
    # Ensure we can load a vae without raising any errors
    loaded_vae = vae_loader.loadmodel(
        model_name=vae_model_path,
        vae_config=config_path,
        precision=vae_precision,
    )

    assert loaded_vae is not None, "VAE loading failed"
    assert isinstance(loaded_vae[0], AutoencoderKLCausal3D)