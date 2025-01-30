import torch
from ..constants import PROMPT_TEMPLATE

# from ..utils.log import log

def get_rewrite_prompt(prompt, mode="Normal"):
    # Placeholder: Just return the original prompt for now
    # we will add support for custom models later
    print(f"prompt rewriting mode: {mode}, we are not loading a model yet")
    return prompt

def get_prompt_template(template_name):
    """
    Placeholder: Retrieves a prompt template by name.

    For now, this simply returns pre-defined templates from constants.py
    """
    if template_name in PROMPT_TEMPLATE:
        return PROMPT_TEMPLATE[template_name]
    else:
        # log.warning(f"Prompt template '{template_name}' not found. Using default.")
        return None

class HyVideoCustomPromptTemplate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "custom_prompt_template": (
                    "STRING",
                    {
                        "default": f"{PROMPT_TEMPLATE['dit-llm-encode-video']['template']}",
                        "multiline": True,
                    },
                ),
                "crop_start": (
                    "INT",
                    {
                        "default": PROMPT_TEMPLATE["dit-llm-encode-video"][
                            "crop_start"
                        ],
                        "tooltip": "To cropt the system prompt",
                    },
                ),
            },
        }

    RETURN_TYPES = ("PROMPT_TEMPLATE",)
    RETURN_NAMES = ("hyvid_prompt_template",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, custom_prompt_template, crop_start):
        prompt_template_dict = {
            "template": custom_prompt_template,
            "crop_start": crop_start,
        }
        return (prompt_template_dict,)

def rewrite_prompt_with_llm(prompt, **kwargs):
    """
    Placeholder: Will eventually rewrite the prompt using an LLM.

    For now, it just returns the original prompt.
    """
    # extract model arguments from kwargs to use during initialization
    # llm_model = kwargs.get("llm_model", "default_llm_model")
    # llm_config = kwargs.get("llm_config", {})
    mode = kwargs.get("mode", "local")

    # TODO: Add logic to load the prompt rewriting model or use an external API
    # if mode is local, load and use a local model for inference
    # if mode is api, make a request to an external API

    print(f"Rewriting prompt using mode: {mode}")
    return prompt