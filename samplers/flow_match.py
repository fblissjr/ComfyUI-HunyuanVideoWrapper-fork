import torch
from .base import BaseSampler
from diffusers.schedulers import(
    FlowMatchDiscreteScheduler,
)

class FlowMatch(BaseSampler):
    def __init__(self):
        super().__init__()
        # This is a workaround to make this code work with ComfyUI.
        # I would like to remove this and make our code base fully compatible with diffusers,
        # but for now this is a quick fix to get things working.
        self.scheduler = FlowMatchDiscreteScheduler()
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
                "sampler_name": ("flow_match",),
                "scheduler": ("ddim", ), # we will update these names later
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
            }
        }

    def sample(self, model, add_noise, seed, steps, start_at_step, end_at_step, cfg, sampler_name, scheduler, positive, negative, latent_image, **kwargs):
        # Implement FlowMatch sampling logic here
        # For simplicity, let's assume it's similar to DPMSolver with necessary adjustments
        
        generator = torch.manual_seed(seed)

        latents = {}
        latents["samples"] = self.scheduler.sample(
            model,
            input_latents=latent_image,
            generator=generator,
            steps=steps,
            # other necessary arguments for FlowMatch
        ).float()

        return (latents,)