# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)

from typing import Literal

from pydantic import BaseModel, Field
import numpy as np

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)

from .model import ModelInfo, UNetField, VaeField

import math

from ...backend.model_management import BaseModelType

class IdealSizeOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    # fmt: off
    type: Literal["ideal_size_output"] = "ideal_size_output"
    width:             int = Field(description="The ideal width of the image in pixels")
    height:            int = Field(description="The ideal height of the image in pixels")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "width", "height"]}

class IdealSizeInvocation(BaseInvocation):
    """Calculates the ideal size for generation to avoid duplication"""

    # fmt: off
    type: Literal["ideal_size"] = "ideal_size"
    width: int = Field(default=1024, description="Target width")
    height: int = Field(default=576, description="Target height")
    unet: UNetField = Field(default=None, description="UNet submodel")
    vae: VaeField = Field(default=None, description="Vae submodel")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Ideal Size",
                "tags": ["math", "ideal_size"]
            },
        }

    def trim_to_multiple_of(self, *args, multiple_of=8):
        return tuple((x - x % multiple_of) for x in args)

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        aspect = self.width / self.height
        dimension = 512 # self.model.unet.config.sample_size * self.model.vae_scale_factor
        if self.unet.unet.base_model == BaseModelType.StableDiffusion2:
            dimension = 768
        elif self.unet.unet.base_model == BaseModelType.StableDiffusionXL:
            dimension = 1024
        min_dimension = math.floor(dimension * 0.5)
        model_area = dimension * dimension # hardcoded for now since all models are trained on square images

        if aspect > 1.0:
            init_height = max(min_dimension, math.sqrt(model_area / aspect))
            init_width = init_height * aspect
        else:
            init_width = max(min_dimension, math.sqrt(model_area * aspect))
            init_height = init_width / aspect

        scaled_width, scaled_height = self.trim_to_multiple_of(
            math.floor(init_width),
            math.floor(init_height)
        )

        return IdealSizeOutput(width=scaled_width, height=scaled_height)
