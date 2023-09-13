# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)

import math

import numpy as np
from pydantic import BaseModel

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.model import ModelInfo, UNetField, VaeField
from invokeai.backend.model_management import BaseModelType


@invocation_output("ideal_size_output")
class IdealSizeOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    width: int = OutputField(description="The ideal width of the image in pixels")
    height: int = OutputField(description="The ideal height of the image in pixels")


@invocation("ideal_size", title="Ideal Size", tags=["math", "ideal_size"], version="1.0.0")
class IdealSizeInvocation(BaseInvocation):
    """Calculates the ideal size for generation to avoid duplication"""

    width: int = InputField(default=1024, description="Target width")
    height: int = InputField(default=576, description="Target height")
    unet: UNetField = InputField(default=None, description="UNet submodel")
    vae: VaeField = InputField(default=None, description="Vae submodel")
    multiplier: float = InputField(default=1.0, description="Dimensional multiplier")

    def trim_to_multiple_of(self, *args, multiple_of=8):
        return tuple((x - x % multiple_of) for x in args)

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        aspect = self.width / self.height
        dimension = 512  # self.model.unet.config.sample_size * self.model.vae_scale_factor
        if self.unet.unet.base_model == BaseModelType.StableDiffusion2:
            dimension = 768
        elif self.unet.unet.base_model == BaseModelType.StableDiffusionXL:
            dimension = 1024
        dimension = dimension * self.multiplier
        min_dimension = math.floor(dimension * 0.5)
        model_area = dimension * dimension  # hardcoded for now since all models are trained on square images

        if aspect > 1.0:
            init_height = max(min_dimension, math.sqrt(model_area / aspect))
            init_width = init_height * aspect
        else:
            init_width = max(min_dimension, math.sqrt(model_area * aspect))
            init_height = init_width / aspect

        scaled_width, scaled_height = self.trim_to_multiple_of(
            math.floor(init_width),
            math.floor(init_height),
        )

        return IdealSizeOutput(width=scaled_width, height=scaled_height)
