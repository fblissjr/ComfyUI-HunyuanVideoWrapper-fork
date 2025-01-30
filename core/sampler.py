from ..utils.log import log
from ..modules.models import HYVideoDiffusionTransformer
from ..pipelines.pipeline_hunyuan_video import HunyuanVideoPipeline
from ..samplers.dpm import DPMSolverMultistepScheduler
from ..samplers.flow_match import FlowMatchDiscreteScheduler
from ..samplers.sa import SASolverScheduler
from ..samplers.base import BaseSampler
import torch
import gc
import comfy.samplers

class HyVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYVIDEOMODEL",),
                "hyvid_embeds": ("HYVIDEMBEDS", ),
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
                "sampler_name": (["dpmsolver++", "flow_match", "sa_solver"],),
                "scheduler": (["euler"], # for now, only euler is supported - will update later to be more comprehensive
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

    def process(self, model, hyvid_embeds, flow_shift, steps, cfg, seed, width, height, num_frames, sampler_name, scheduler, samples=None, stg_args=None, context_options=None, feta_args=None, teacache_args=None):
        pass