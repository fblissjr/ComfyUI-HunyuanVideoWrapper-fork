import torch
import os # for script directory and the existing HyVideoModelConfig class
import gc
import json

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file
import comfy.model_base
import comfy.latent_formats

from .core.model_loader import HyVideoModelLoader, filter_state_dict_by_blocks, standardize_lora_key_format
from .core.sampler import HyVideoSampler
from .core.vae import HyVideoVAELoader, HyVideoEncode, HyVideoDecode

# from .modules.models import (
#     HYVideoDiffusionTransformer,
# )  # leave this out of our first version of this update

from .samplers.dpm import (
    DPMSolverMultistepScheduler
)

from .samplers.base import BaseSampler
from .samplers.enhance import HyVideoEnhanceAVideo, get_feta_scores
from .samplers.teacache import HyVideoTeaCache

from .utils.torch_utils import (
    print_memory,
    get_fp_maxval,
    quantize_to_fp8,
    fp8_tensor_quant,
    fp8_activation_dequant,
    fp8_linear_forward,
    convert_fp8_linear
)

from .utils.data_utils import (
    align_to,
    save_videos_grid
)

from .utils.path_utils import (
    get_model_path,
    get_lora_path,
    get_text_encoder_path,
    get_vae_path,
    get_hyvid_embeds_path,
)

from .text.prompts import HyVideoCustomPromptTemplate, get_rewrite_prompt
from .text.text_encoder import (
    DownloadAndLoadHyVideoTextEncoder,
    TextEncoder,
)
from .text.embeddings import (
    HyVideoTextEmbedsSave,
    HyVideoTextEmbedsLoad
)

from .utils.log import setup_logger
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
    # "HyVideoSTG": HyVideoSTG, # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoCFG": HyVideoCFG, # these are nodes that are currently unavailable due to reliance on removed files
    "HyVideoCustomPromptTemplate": HyVideoCustomPromptTemplate,
    # "HyVideoLatentPreview": HyVideoLatentPreview, # this requires VAE to be imported from diffusers - maybe we should install as editable?
    "HyVideoLoraSelect": HyVideoLoraSelect,
    "HyVideoTextEmbedsSave": HyVideoTextEmbedsSave,
    "HyVideoTextEmbedsLoad": HyVideoTextEmbedsLoad,
    # "HyVideoEnhanceAVideo": HyVideoEnhanceAVideo, # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoTeaCache": HyVideoTeaCache, # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoRichSpaceTextEncode": HyVideoRichSpaceTextEncode, # these are nodes that are currently unavailable due to reliance on removed files
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyVideoSampler": "HunyuanVideo Sampler",
    "HyVideoDecode": "HunyuanVideo Decode",
    "HyVideoEncode": "HunyuanVideo Encode",
    "HyVideoModelLoader": "HunyuanVideo Model Loader",
    "HyVideoVAELoader": "HunyuanVideo VAE Loader",
    "DownloadAndLoadHyVideoTextEncoder": "(Down)Load HunyuanVideo TextEncoder",
    "HyVideoTextEncode": "HunyuanVideo TextEncode",
    # "HyVideoBlockSwap": "HunyuanVideo BlockSwap", # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoTorchCompileSettings": "HunyuanVideo Torch Compile Settings", # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoSTG": "HunyuanVideo STG", # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoCFG": "HunyuanVideo CFG", # these are nodes that are currently unavailable due to reliance on removed files
    "HyVideoCustomPromptTemplate": "HunyuanVideo Custom Prompt Template",
    # "HyVideoLatentPreview": "HunyuanVideo Latent Preview", # this requires VAE to be imported from diffusers - maybe we should install as editable?
    "HyVideoLoraSelect": "HunyuanVideo Lora Select",
    # "HyVideoLoraBlockEdit": "HunyuanVideo Lora Block Edit", # these are nodes that are currently unavailable due to reliance on removed files
    "HyVideoTextEmbedsSave": "HunyuanVideo Text Embeds Save",
    "HyVideoTextEmbedsLoad": "HunyuanVideo Text Embeds Load",
    # "HyVideoContextOptions": "HunyuanVideo Context Options", # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoEnhanceAVideo": "HunyuanVideo Enhance A Video", # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoTeaCache": "HunyuanVideo TeaCache", # these are nodes that are currently unavailable due to reliance on removed files
    # "HyVideoRichSpaceTextEncode": "HunyuanVideo RichSpace TextEncode", # these are nodes that are currently unavailable due to reliance on removed files
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]