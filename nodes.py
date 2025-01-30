import torch
import os  # for script directory and the existing HyVideoModelConfig class
import gc
import json

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file
import comfy.model_base
import comfy.latent_formats

from .core.model_loader import (
    HyVideoModelLoader,
    filter_state_dict_by_blocks,
    standardize_lora_key_format,
)
from .core.sampler import HyVideoSampler
from .core.vae import HyVideoVAELoader, HyVideoEncode, HyVideoDecode
from .modules.attention import attention, get_cu_seqlens
from .modules.models import HYVideoDiffusionTransformer
from .samplers.dpm import DPMSolverMultistepScheduler
from .samplers.flow_match import FlowMatchDiscreteScheduler
from .samplers.sa import SASolverScheduler
from .samplers.base import BaseSampler
from .samplers.enhance import HyVideoEnhanceAVideo, get_feta_scores
from .samplers.teacache import HyVideoTeaCache
from .utils.torch_utils import print_memory
from .utils.data_utils import align_to, save_videos_grid
from .utils.path_utils import (
    get_model_path,
    get_lora_path,
    get_text_encoder_path,
    get_vae_path,
    get_hyvid_embeds_path,
)
from .text.prompt import HyVideoCustomPromptTemplate, get_rewrite_prompt
from .text.text_encoder import (
    DownloadAndLoadHyVideoTextEncoder,
    TextEncoder,
    TextEncoderModelOutput,
)
from .text.embeddings import HyVideoTextEmbedsSave, HyVideoTextEmbedsLoad
from .utils.log import setup_logger
from .modules.context import get_context_scheduler, get_total_steps

log = setup_logger(__name__)

# nodes that are currently not used by the repo
# HyVideoLoraBlockEdit
# HyVideoBlockSwap
# HyVideoTorchCompileSettings
# HyVideoContextOptions

NODE_CLASS_MAPPINGS = {
    "HyVideoSampler": HyVideoSampler,
    "HyVideoDecode": HyVideoDecode,
    "HyVideoEncode": HyVideoEncode,
    "HyVideoModelLoader": HyVideoModelLoader,
    "HyVideoVAELoader": HyVideoVAELoader,
    "DownloadAndLoadHyVideoTextEncoder": DownloadAndLoadHyVideoTextEncoder,
    "HyVideoTextEncode": HyVideoTextEncode,
    "HyVideoCustomPromptTemplate": HyVideoCustomPromptTemplate,
    "HyVideoLoraSelect": HyVideoLoraSelect,
    "HyVideoTextEmbedsSave": HyVideoTextEmbedsSave,
    "HyVideoTextEmbedsLoad": HyVideoTextEmbedsLoad,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyVideoSampler": "HunyuanVideo Sampler",
    "HyVideoDecode": "HunyuanVideo Decode",
    "HyVideoEncode": "HunyuanVideo Encode",
    "HyVideoModelLoader": "HunyuanVideo Model Loader",
    "HyVideoVAELoader": "HunyuanVideo VAE Loader",
    "DownloadAndLoadHyVideoTextEncoder": "(Down)Load HunyuanVideo TextEncoder",
    "HyVideoTextEncode": "HunyuanVideo TextEncode",
    "HyVideoCustomPromptTemplate": "HunyuanVideo Custom Prompt Template",
    "HyVideoLoraSelect": "HunyuanVideo Lora Select",
    "HyVideoTextEmbedsSave": "HunyuanVideo Text Embeds Save",
    "HyVideoTextEmbedsLoad": "HunyuanVideo Text Embeds Load",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]