import torch
from .base import BaseSampler
from diffusers.schedulers import SASolverScheduler

class SASolver(BaseSampler):
    def __init__(self):
        super().__init__()
        # this is a workaround to make this code work with ComfyUI
        # I would like to remove this and make our code base fully compatible with diffusers
        # but for now this is a quick fix to get things working, in a future refactor we'll follow
        # the standard approach.
        self.scheduler = SASolverScheduler()
        self.scheduler.alphas_cumprod = torch.tensor([0])
        self.scheduler.sigmas = torch.tensor([0])
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": ("sa_solver",),
                "scheduler": ("sgm_uniform", ), # we will update these names later
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
            }
        }

    def sample(self, model, add_noise, seed, steps, start_at_step, end_at_step, cfg, sampler_name, scheduler, positive, negative, latent_image, **kwargs):
        # this does not match diffusers functionality exactly, but aims to be close enough
        # that we can eventually refactor the two to be fully compatible in a later iteration
        # when doing so, it will be possible to delete our fork of DPMSolverMultistepScheduler
        # and use theirs instead.
        generator = torch.manual_seed(seed)

        latents = {}
        latents["samples"] = self.scheduler.sample(
            model,
            input_latents=latent_image,
            generator=generator,
            steps=steps,
            order=4,
            skip_type="time_uniform",
            method="multistep",
            lower_order_final=False,
        ).float()

        return (latents, )