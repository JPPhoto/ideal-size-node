# ideal-size-node
An InvokeAI node for calculating ideal image sizes to avoid duplication.
This InvokeAI node takes in your target dimensions, Unet, and VAE, and outputs a width and height for initial generation using the model you've fed it. For example, asking for a 2048x2048 image would give you 512x512 when using a SD 1.x model. You can generate with that resolution to avoid duplication and other artifacts that plague models when images get too big, then upscale and enhance the resulting latents however you want.
