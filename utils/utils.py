import torch
import numpy as np

from PIL import Image
from ldm.util import instantiate_from_config
from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense

def load_ckpt(ckpt_path, device, models_to_load=None):
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model_names = ["model", "autoencoder", "text_encoder", "diffusion"]
    loaded_models = {}

    for model_name in model_names:
        if not models_to_load or model_name in models_to_load:
            instance = instantiate_from_config(config[model_name]).to(device)
            instance.load_state_dict(saved_ckpt[model_name])

            if model_name in ["autoencoder", "text_encoder"]:
                instance.eval()

            loaded_models[model_name] = instance
        else:
            loaded_models[model_name] = None

    return loaded_models, config


def get_text_clip_feature(clip_model, clip_processor, phrase, config):
    txt_embeds = None

    if phrase is not None:
        inputs = clip_processor(text=phrase, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(config.device)
        inputs['pixel_values'] = torch.ones(1, 3, 224, 224).to(config.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(config.device)

        outputs = clip_model(**inputs)
        txt_embeds = outputs['text_model_output']['pooler_output']

    return txt_embeds


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def set_alpha_scale(model, alpha_scale):
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timesteps needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear decay stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas
