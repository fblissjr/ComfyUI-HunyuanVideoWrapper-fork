import os
import json
from comfy.utils import save_torch_file, load_torch_file
import folder_paths

# these functions were previously defined as HyVideoTextEmbedsSave and HyVideoTextEmbedsLoad
class HyVideoTextEmbedsSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hyvid_embeds": ("HYVIDEMBEDS",),
                "filename_prefix": ("STRING", {"default": "hyvid_embeds/hyvid_embed"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "save"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Save the text embeds"

    def save(self, hyvid_embeds, prompt, filename_prefix, extra_pnginfo=None):
        from comfy.cli_args import args

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )
        file = f"{filename}_{counter:05}_.safetensors"
        file = os.path.join(full_output_folder, file)
        tensors_to_save = {}
        for key, value in hyvid_embeds.items():
            if value is not None:
                tensors_to_save[key] = value
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)
        metadata = None
        if not args.disable_metadata:
            metadata = {"prompt": prompt_info}
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        save_torch_file(tensors_to_save, file, metadata=metadata)
        return (file,)

class HyVideoTextEmbedsLoad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embeds": (folder_paths.get_filename_list("hyvid_embeds"), {
                    "tooltip": "The saved embeds to load from output/hyvid_embeds."
                })
            }
        }

    RETURN_TYPES = ("HYVIDEMBEDS",)
    RETURN_NAMES = ("hyvid_embeds",)
    FUNCTION = "load"
    CATEGORY = "HunyuanVideoWrapper"
    DESCTIPTION = "Load the saved text embeds"

    def load(self, embeds):
        embed_path = folder_paths.get_full_path_or_raise("hyvid_embeds", embeds)
        loaded_tensors = load_torch_file(embed_path, safe_load=True)
        # Reconstruct original dictionary with None for missing keys
        prompt_embeds_dict = {
            "prompt_embeds": loaded_tensors.get("prompt_embeds", None),
            "negative_prompt_embeds": loaded_tensors.get("negative_prompt_embeds", None),
            "attention_mask": loaded_tensors.get("attention_mask", None),
            "negative_attention_mask": loaded_tensors.get("negative_attention_mask", None),
            "prompt_embeds_2": loaded_tensors.get("prompt_embeds_2", None),
            "negative_prompt_embeds_2": loaded_tensors.get("negative_prompt_embeds_2", None),
            "attention_mask_2": loaded_tensors.get("attention_mask_2", None),
            "negative_attention_mask_2": loaded_tensors.get("negative_attention_mask_2", None),
            "cfg": loaded_tensors.get("cfg", None),
            "start_percent": loaded_tensors.get("start_percent", None),
            "end_percent": loaded_tensors.get("end_percent", None),
        }

        return (prompt_embeds_dict,)