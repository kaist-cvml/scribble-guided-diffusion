import os
import torch
import numpy as np

from PIL import Image

from data import PromptInput
from scribble_propagation import SelfAttnAggregator, ScribblePropagator


def run(
        models,
        steps,
        scale,
        opt,
        sampler,
        prompt_input: PromptInput,
        x_T=None,
        self_attn_aggregator:SelfAttnAggregator = None,
        scribble_propagator:ScribblePropagator = None,
        device=torch.device("cpu")
    ):
    '''
    Run the model to generate an images.
    '''
    if x_T is None:
        x_T = torch.randn(prompt_input.batch_size, 4, 64, 64).to(device)

    model = models["model"]
    autoencoder = models["autoencoder"]
    text_encoder = models["text_encoder"]
    
    batch_size = prompt_input.batch_size
    
    if scribble_propagator is not None:
        scribble_propagator.initialize_propagation(
            prompt_input.individual_scribbles
        )

    context = text_encoder.encode(prompt_input.prompts)
    uncond = text_encoder.encode([""] * batch_size)

    if opt.negative_prompt is not None:
        uncond = text_encoder.encode([opt.negative_prompt] * batch_size)

    shape = (batch_size, model.in_channels, model.image_size, model.image_size)

    z_0 = sampler.sample(
        x_t=x_T,
        S=steps, 
        shape=shape, 
        prompt_input=prompt_input,
        c=context,
        uc=uncond, 
        guidance_scale=scale,
        scribble_propagator=scribble_propagator,
        self_attn_aggregator=self_attn_aggregator, 
    )
    
    with torch.no_grad():
        x_0 = autoencoder.decode(z_0)

    return x_0
