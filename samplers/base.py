# content of samplers/base.py
# defines common methods or attributes shared by sampler classes.

class BaseSampler:
    """
    Base class for samplers.

    Subclasses should implement the specific sampling logic,
    inheriting common functionality from this class.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for ComfyUI input node.

        """
        raise NotImplementedError("Subclasses must define input config")

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def __init__(self):
        pass  # you can initialize any common attributes here

    def sample(self, *args, **kwargs):
        """
        This method should implement the core sampling logic,
        common to all samplers.

        Specific sampler implementations should override this method
        if they need different behavior.
        """
        raise NotImplementedError("Subclasses must implement sampling logic")

# any common helper functions can be added here in the future