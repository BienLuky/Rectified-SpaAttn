
import torch
from typing import Any, List, Tuple, Optional, Union, Dict
from diffusers.models.attention_processor import AttentionProcessor

def get_attn_processors(module) -> Dict[str, AttentionProcessor]:
    r"""
    Returns:
        `dict` of attention processors: A dictionary containing all attention processors used in the model with
        indexed by its weight name.
    """
    # set recursively
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
        if hasattr(module, "get_processor"):
            processors[f"{name}.processor"] = module.get_processor()

        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in module.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors

# Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
def set_attn_processor(module, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
    r"""
    Sets the attention processor to use to compute attention.

    Parameters:
        processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
            The instantiated processor class or a dictionary of processor classes that will be set as the processor
            for **all** `Attention` layers.

            If `processor` is a dict, the key needs to define the path to the corresponding cross attention
            processor. This is strongly recommended when setting trainable attention processors.

    """
    count = len(get_attn_processors(module).keys())

    if isinstance(processor, dict) and len(processor) != count:
        raise ValueError(
            f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
            f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
        )

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor"):
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in module.named_children():
        fn_recursive_attn_processor(name, module, processor)
