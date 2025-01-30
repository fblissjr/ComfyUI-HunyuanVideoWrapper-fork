import os
import json

from comfy.utils import load_torch_file
import comfy.model_management as mm
import torch

from ..utils.log import log

# imports needed for when we add these back later:
from diffusers.video_processor import VideoProcessor
from typing import List, Dict, Any, Tuple
from ..hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D

class HyVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
                "compile_args":("COMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision, compile_args=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        # After updating the directory structure, this will no longer be needed
        # with open(os.path.join(script_directory, 'configs', 'hy_vae_config.json')) as f:
        #     vae_config = json.load(f)
        # Instead, we should put the config json alongside other models so it can be used the same way,
        # i.e. we will modify this line later to use something like this:
        vae_config_path = folder_paths.get_full_path("vae_configs", "hy_vae_config.json")
        with open(vae_config_path) as f:
            vae_config = json.load(f)
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        vae = AutoencoderKLCausal3D.from_config(vae_config)
        vae.load_state_dict(vae_sd)
        del vae_sd
        vae.requires_grad_(False)
        vae.eval()
        vae.to(device = device, dtype = dtype)

        #compile
        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            vae = torch.compile(vae, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            

        return (vae,)

# at the moment these are equivalent to just using the VAE loader/encoder/decoder nodes provided by comfyui. we will
# modify these and add functionality in later steps, after we are done refactoring.

class HyVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "temporal_tiling_sample_size": ("INT", {"default": 64, "min": 4, "max": 256, "tooltip": "Smaller values use less VRAM, model default is 64, any other value will cause stutter"}),
                    "spatial_tile_sample_min_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32, "tooltip": "Spatial tile minimum size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Automatically set tile size based on defaults, above settings are ignored"}),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "HunyuanVideoWrapper"

    def encode(self, vae, image, enable_vae_tiling, temporal_tiling_sample_size, auto_tile_size, spatial_tile_sample_min_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)
        if not auto_tile_size:
            vae.tile_latent_min_tsize = temporal_tiling_sample_size // 4
            vae.tile_sample_min_size = spatial_tile_sample_min_size
            vae.tile_latent_min_size = spatial_tile_sample_min_size // 8
            if temporal_tiling_sample_size != 64:
                vae.t_tile_overlap_factor = 0.0
            else:
                vae.t_tile_overlap_factor = 0.25
        else:
            #defaults
            vae.tile_latent_min_tsize = 16
            vae.tile_sample_min_size = 256
            vae.tile_latent_min_size = 32

        image = (image * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        if enable_vae_tiling:
            vae.enable_tiling()
        latents = vae.encode(image).latent_dist.sample(generator)
        #latents = latents * vae.config.scaling_factor
        vae.to(offload_device)
        print("encoded latents shape",latents.shape)

        return ({"samples": latents},)

class HyVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "temporal_tiling_sample_size": ("INT", {"default": 64, "min": 4, "max": 256, "tooltip": "Smaller values use less VRAM, model default is 64, any other value will cause stutter"}),
                    "spatial_tile_sample_min_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32, "tooltip": "Spatial tile minimum size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Automatically set tile size based on defaults, above settings are ignored"}),
                    },
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "HunyuanVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, temporal_tiling_sample_size, spatial_tile_sample_min_size, auto_tile_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()
        latents = samples["samples"]
        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)
        if not auto_tile_size:
            vae.tile_latent_min_tsize = temporal_tiling_sample_size // 4
            vae.tile_sample_min_size = spatial_tile_sample_min_size
            vae.tile_latent_min_size = spatial_tile_sample_min_size // 8
            if temporal_tiling_sample_size != 64:
                vae.t_tile_overlap_factor = 0.0
            else:
                vae.t_tile_overlap_factor = 0.25
        else:
            #defaults
            vae.tile_latent_min_tsize = 16
            vae.tile_sample_min_size = 256
            vae.tile_latent_min_size = 32

        expand_temporal_dim = False
        if len(latents.shape) == 4:
            if isinstance(vae, AutoencoderKLCausal3D):
                latents = latents.unsqueeze(2)
                expand_temporal_dim = True
        elif len(latents.shape) == 5:
            pass
        else:
            raise ValueError(
                f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
            )
        #latents = latents / vae.config.scaling_factor
        latents = latents.to(vae.dtype).to(device)

        if enable_vae_tiling:
            vae.enable_tiling()
            video = vae.decode(
                latents, return_dict=False, generator=generator
            )[0]
        else:
            video = vae.decode(
                latents, return_dict=False, generator=generator
            )[0]

        if expand_temporal_dim or video.shape[2] == 1:
            video = video.squeeze(2)

        vae.to(offload_device)
        mm.soft_empty_cache()

        if len(video.shape) == 5:
            video_processor = VideoProcessor(vae_scale_factor=8)
            video_processor.config.do_resize = False

            video = video_processor.postprocess_video(video=video, output_type="pt")
            out = video[0].permute(0, 2, 3, 1).cpu().float()
        else:
            out = (video / 2 + 0.5).clamp(0, 1)
            out = out.permute(0, 2, 3, 1).cpu().float()

        return (out,)