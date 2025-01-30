import gc

import torch
import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file
from comfy.model_patcher import ModelPatcher
import comfy.model_base
import comfy.latent_formats

from ..modules.models import HYVideoDiffusionTransformer
from .model_config import HyVideoModelConfig, HUNYUAN_VIDEO_CONFIG

from ..utils.log import log

# this function is the only one that needs to stay, since it ensures our lora loading method is preserved.
# we need to find another home for this eventually, too. i would like to change how loras are loaded for a
# future pr, and this is where the changes should be made to do that
def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # Diffusers format
        if k.startswith("transformer."):
            k = k.replace("transformer.", "diffusion_model.")
        new_sd[k] = v
    return new_sd

# moved from nodes.py
def filter_state_dict_by_blocks(state_dict, blocks_mapping):
    filtered_dict = {}

    for key in state_dict:
        if "double_blocks." in key or "single_blocks." in key:
            block_pattern = key.split("diffusion_model.")[1].split(".", 2)[0:2]
            block_key = f"{block_pattern[0]}.{block_pattern[1]}."

            if block_key in blocks_mapping:
                filtered_dict[key] = state_dict[key]

    return filtered_dict

# mostly code moved from nodes.py
# original comments preserved
class HyVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {
                        "tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",
                    },
                ),
                "base_precision": (["fp32", "bf16"], {"default": "bf16"}),
                "quantization": (
                    [
                        "disabled",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_scaled",
                        "torchao_fp8dq",
                        "torchao_fp8dqrow",
                        "torchao_int8dq",
                        "torchao_fp6",
                        "torchao_int4",
                        "torchao_int8",
                    ],
                    {
                        "default": "disabled",
                        "tooltip": "optional quantization method",
                    },
                ),
                "load_device": (
                    ["main_device", "offload_device"],
                    {"default": "main_device"},
                ),
            },
            "optional": {
                "attention_mode": (
                    ["sdpa", "flash_attn_varlen", "sageattn_varlen", "comfy"],
                    {"default": "sdpa"},
                ),
                "compile_args": ("COMPILEARGS",),
                "block_swap_args": ("BLOCKSWAPARGS",),
                "lora": ("HYVIDLORA", {"default": None}),
                "auto_cpu_offload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable auto offloading for reduced VRAM usage, implementation from DiffSynth-Studio, slightly different from block swapping and uses even less VRAM, but can be slower as you can't define how much VRAM to use",
                    },
                ),
                "upcast_rope": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Upcast RoPE to fp32 for better accuracy, this is the default behaviour, disabling can improve speed and reduce memory use slightly",
                    },
                ),
            },
        }

    RETURN_TYPES = ("HYVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"

    def loadmodel(
        self,
        model,
        base_precision,
        load_device,
        quantization,
        compile_args=None,
        attention_mode="sdpa",
        block_swap_args=None,
        lora=None,
        auto_cpu_offload=False,
        upcast_rope=True,
    ):
        transformer = None
        # mm.unload_all_models()
        mm.soft_empty_cache()
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn_varlen
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        manual_offloading = True
        transformer_load_device = (
            device if load_device == "main_device" else offload_device
        )

        base_dtype = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e4m3fn_fast": torch.float8_e4m3fn,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[base_precision]

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        in_channels = out_channels = 16
        factor_kwargs = {"device": transformer_load_device, "dtype": base_dtype}

        # note: these values need to match those defined in model_config.py!
        HYVIDEOMODEL_CONFIG = {
            "mm_double_blocks_depth": 20,
            "mm_single_blocks_depth": 40,
            "rope_dim_list": [16, 56, 56],
            "hidden_size": 3072,
            "heads_num": 24,
            "mlp_width_ratio": 4,
            "guidance_embed": True,
        }
        transformer = HYVideoDiffusionTransformer(
            in_channels=in_channels,
            out_channels=out_channels,
            attention_mode=attention_mode,
            main_device=device,
            offload_device=offload_device,
            **HYVIDEOMODEL_CONFIG,
            **factor_kwargs,
        )
        transformer.eval()

        transformer.upcast_rope = upcast_rope

        comfy_model = HyVideoModel(
            HyVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )

        scheduler_config = {
            "flow_shift": 9.0,
            "reverse": True,
            "solver": "euler",
            "use_flow_sigmas": True,
            "prediction_type": "flow_prediction",
        }
        scheduler = FlowMatchDiscreteScheduler.from_config(scheduler_config)

        pipe = HunyuanVideoPipeline(
            transformer=transformer,
            scheduler=scheduler,
            progress_bar_config=None,
            base_dtype=base_dtype,
            comfy_model=comfy_model,
        )

        if not "torchao" in quantization:
            log.info("Using accelerate to load and assign model weights to device...")
            if (
                quantization == "fp8_e4m3fn"
                or quantization == "fp8_e4m3fn_fast"
                or quantization == "fp8_scaled"
            ):
                dtype = torch.float8_e4m3fn
            else:
                dtype = base_dtype
            params_to_keep = {
                "norm",
                "bias",
                "time_in",
                "vector_in",
                "guidance_in",
                "txt_in",
                "img_in",
            }
            for name, param in transformer.named_parameters():
                dtype_to_use = (
                    base_dtype
                    if any(keyword in name for keyword in params_to_keep)
                    else dtype
                )
                set_module_tensor_to_device(
                    transformer,
                    name,
                    device=transformer_load_device,
                    dtype=dtype_to_use,
                    value=sd[name],
                )

            comfy_model.diffusion_model = transformer
            patcher = ModelPatcher(comfy_model, device, offload_device)
            pipe.comfy_model = patcher

            del sd
            gc.collect()
            mm.soft_empty_cache()

            if lora is not None:
                from comfy.sd import load_lora_for_models

                for l in lora:
                    log.info(
                        f"Loading LoRA: {l['name']} with strength: {l['strength']}"
                    )
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)
                    lora_sd = standardize_lora_key_format(lora_sd)
                    if l["blocks"]:
                        lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"])

                    patcher, _ = load_lora_for_models(
                        patcher, None, lora_sd, lora_strength, 0
                    )

            comfy.model_management.load_models_gpu([patcher])
            if load_device == "offload_device":
                patcher.model.diffusion_model.to(offload_device)

            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear

                convert_fp8_linear(
                    patcher.model.diffusion_model,
                    base_dtype,
                    params_to_keep=params_to_keep,
                )
            elif quantization == "fp8_scaled":
                from .hyvideo.modules.fp8_optimization import convert_fp8_linear

                convert_fp8_linear(patcher.model.diffusion_model, base_dtype)

            if auto_cpu_offload:
                transformer.enable_auto_offload(dtype=dtype, device=device)

            # compile
            if compile_args is not None:
                torch._dynamo.config.cache_size_limit = compile_args[
                    "dynamo_cache_size_limit"
                ]
                if compile_args["compile_single_blocks"]:
                    for i, block in enumerate(
                        patcher.model.diffusion_model.single_blocks
                    ):
                        patcher.model.diffusion_model.single_blocks[i] = torch.compile(
                            block,
                            fullgraph=compile_args["fullgraph"],
                            dynamic=compile_args["dynamic"],
                            backend=compile_args["backend"],
                            mode=compile_args["mode"],
                        )
                if compile_args["compile_double_blocks"]:
                    for i, block in enumerate(
                        patcher.model.diffusion_model.double_blocks
                    ):
                        patcher.model.diffusion_model.double_blocks[i] = torch.compile(
                            block,
                            fullgraph=compile_args["fullgraph"],
                            dynamic=compile_args["dynamic"],
                            backend=compile_args["backend"],
                            mode=compile_args["mode"],
                        )
                if compile_args["compile_txt_in"]:
                    patcher.model.diffusion_model.txt_in = torch.compile(
                        patcher.model.diffusion_model.txt_in,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )
                if compile_args["compile_vector_in"]:
                    patcher.model.diffusion_model.vector_in = torch.compile(
                        patcher.model.diffusion_model.vector_in,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )
                if compile_args["compile_final_layer"]:
                    patcher.model.diffusion_model.final_layer = torch.compile(
                        patcher.model.diffusion_model.final_layer,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )
        elif "torchao" in quantization:
            try:
                from torchao.quantization import (
                    quantize_,
                    fpx_weight_only,
                    float8_dynamic_activation_float8_weight,
                    int8_dynamic_activation_int8_weight,
                    int8_weight_only,
                    int4_weight_only,
                )
            except:
                raise ImportError("torchao is not installed")

            if "fp6" in quantization:
                quant_func = fpx_weight_only(3, 2)
            elif "int4" in quantization:
                quant_func = int4_weight_only()
            elif "int8" in quantization:
                quant_func = int8_weight_only()
            elif "fp8dq" in quantization:
                quant_func = float8_dynamic_activation_float8_weight()
            elif "fp8dqrow" in quantization:
                from torchao.quantization.quant_api import PerRow

                quant_func = float8_dynamic_activation_float8_weight(
                    granularity=PerRow()
                )
            elif "int8dq" in quantization:
                quant_func = int8_dynamic_activation_int8_weight()

            log.info(f"Quantizing model with {quant_func}")
            comfy_model.diffusion_model = transformer
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)

            if lora is not None:
                from comfy.sd import load_lora_for_models

                for l in lora:
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)
                    lora_sd = standardize_lora_key_format(lora_sd)
                    patcher, _ = load_lora_for_models(
                        patcher, None, lora_sd, lora_strength, 0
                    )

            comfy.model_management.load_models_gpu([patcher])

            for i, block in enumerate(patcher.model.diffusion_model.single_blocks):
                log.info(f"Quantizing single_block {i}")
                for name, _ in block.named_parameters(prefix=f"single_blocks.{i}"):
                    set_module_tensor_to_device(
                        patcher.model.diffusion_model,
                        name,
                        device=transformer_load_device,
                        dtype=base_dtype,
                        value=sd[name],
                    )
                if compile_args is not None:
                    patcher.model.diffusion_model.single_blocks[i] = torch.compile(
                        block,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )
                quantize_(block, quant_func)
                print(block)
                block.to(offload_device)
            for i, block in enumerate(patcher.model.diffusion_model.double_blocks):
                log.info(f"Quantizing double_block {i}")
                for name, _ in block.named_parameters(prefix=f"double_blocks.{i}"):
                    set_module_tensor_to_device(
                        patcher.model.diffusion_model,
                        name,
                        device=transformer_load_device,
                        dtype=base_dtype,
                        value=sd[name],
                    )
                if compile_args is not None:
                    patcher.model.diffusion_model.double_blocks[i] = torch.compile(
                        block,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )
                quantize_(block, quant_func)
            for name, param in patcher.model.diffusion_model.named_parameters():
                if "single_blocks" not in name and "double_blocks" not in name:
                    set_module_tensor_to_device(
                        patcher.model.diffusion_model,
                        name,
                        device=transformer_load_device,
                        dtype=base_dtype,
                        value=sd[name],
                    )

            manual_offloading = False  # to disable manual .to(device) calls
            log.info(f"Quantized transformer blocks to {quantization}")
            for name, param in patcher.model.diffusion_model.named_parameters():
                print(name, param.dtype)
                # param.data = param.data.to(self.vae_dtype).to(device)

            del sd
            mm.soft_empty_cache()

        patcher.model["pipe"] = pipe
        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = manual_offloading
        patcher.model["quantization"] = "disabled"
        patcher.model["block_swap_args"] = block_swap_args
        patcher.model["auto_cpu_offload"] = auto_cpu_offload
        patcher.model["scheduler_config"] = scheduler_config

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)

        return (patcher,)