import folder_paths
import os

# right now we just move the code over, but later we can make this more sophisticated by managing config data in the
# same way as we manage models
def get_model_path(subfolder, model_name):
    return folder_paths.get_full_path(subfolder, model_name)

def get_lora_path(lora_name):
    return folder_paths.get_full_path("loras", lora_name)

def get_text_encoder_path(text_encoder_name):
    return folder_paths.get_full_path("llm", text_encoder_name)

def get_vae_path(vae_name):
    return folder_paths.get_full_path("vae", vae_name)

def get_hyvid_embeds_path(embed_name):
    return folder_paths.get_full_path("hyvid_embeds", embed_name)

# we can add more helper functions here for managing paths to other resources as needed