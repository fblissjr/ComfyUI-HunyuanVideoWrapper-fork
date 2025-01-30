from ..utils.log import log
from ..modules.models import HYVideoDiffusionTransformer
from ..diffusion.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from ..diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoPipeline
import torch
import gc

class HyVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYVIDEOMODEL",),
                "hyvid_embeds": ("HYVIDEMBEDS",),
                "width": (
                    "INT",
                    {"default": 512, "min": 64, "max": 4096, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 64, "max": 4096, "step": 16},
                ),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 1024, "step": 4}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "guidance scale",
                    },
                ),
                "sampler_name": (
                    ["flow_match"],
                    {
                        "tooltip": "Schedule name (different schedulers have different behaviour in terms of steps)"
                    },
                ),
                "scheduler": (
                    ["flow_match"],
                    {
                        "tooltip": "Schedule name (different schedulers have different behaviour in terms of steps)"
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, hyvid_embeds, steps, cfg, seed, width, height, num_frames, sampler_name, scheduler, samples=None, stg_args=None, context_options=None, feta_args=None, teacache_args=None):
        pass