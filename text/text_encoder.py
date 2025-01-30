from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel, LlavaForConditionalGeneration, AutoProcessor

from ..utils.log import log
from ..constants import TEXT_ENCODER_PATH, TOKENIZER_PATH
from ..utils.token_helper import find_subsequence, multi_slice_to_mask
from PIL import Image

def use_default(value, default):
    return value if value is not None else default

def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
    dtype=None,
    quantization_config=None,
):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        text_encoder.final_layer_norm = text_encoder.norm
    elif text_encoder_type == "vlm":
        text_encoder = LlavaForConditionalGeneration.from_pretrained(
            text_encoder_path, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    if text_encoder_precision is not None and quantization_config is None and dtype != torch.float8_e4m3fn:
        text_encoder = text_encoder.to(dtype)

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {dtype}")

    if device is not None and quantization_config is None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path

def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llm" or tokenizer_type == "vlm":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path
    

@dataclass
class TextEncoderModelOutput:
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None