import os
import json
import torch
import argparse
import numpy as np

from PIL import Image
from functools import partial
from omegaconf import OmegaConf

from run import run
from args import parse_args
from data import PromptInput
from utils.utils import load_ckpt, set_alpha_scale, alpha_generator
from pytorch_lightning import seed_everything
from transformers import CLIPModel, CLIPProcessor

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

from losses.loss_config import ScribbleLossConfig
from losses.loss_scheduler import ScribbleLossScheduler
from scribble_propagation import ScribblePropagator, SelfAttnAggregator


def preprocess_prompt_input(
        prompt_inputs: PromptInput, 
        text_encoder,
        clip_model,
        clip_processor,
        grounding_tokenizer_input
    ):
    prompt_inputs.valid_check()
    
    # get phrases and strokes
    prompt_inputs.get_phrases_and_strokes_from_inputs()

    # get scribbles
    prompt_inputs.get_scribbles_from_strokes()

    # get masks
    # prompt_inputs.get_masks_from_scribbles()
    
    prompt_inputs.get_tensors_from_lists(text_encoder)
    
    # save scribbles
    prompt_inputs.save_scribbles()
    prompt_inputs.save_scribbles(save_individual=True)
    
    # save masks
    prompt_inputs.save_masks()
    prompt_inputs.save_masks(save_individual=True)
    
    prompt_inputs.get_grounding_input(
        clip_model=clip_model,
        clip_processor=clip_processor,
        grounding_tokenizer_input=grounding_tokenizer_input
    )
    
    return


def main(opt, config_file):
    seed_everything(opt.seed)
    
    assert os.path.exists(opt.ckpt), "Please specify the path to the checkpoint."
    device = torch.device(opt.device)
    
    loss_config = config_file["loss_config"]
    loss_scheduler_config = config_file["loss_scheduler"]
    prop_config = config_file["propagation"]
    
    # - - - - - load models - - - - - #
    models, config = load_ckpt(opt.ckpt, device)
    config = OmegaConf.create(config)
    
    text_encoder = models["text_encoder"]

    grounding_tokenizer_input = instantiate_from_config(config["grounding_tokenizer_input"])
    models["model"].grounding_tokenizer_input = grounding_tokenizer_input
    
    clip_ver = text_encoder.tokenizer.name_or_path
    clip_model = CLIPModel.from_pretrained(clip_ver).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_ver)

    alpha_generator_func = partial(alpha_generator, type=opt.alpha_type)

    loss_scheduler = ScribbleLossScheduler(loss_scheduler_config)
    loss_config = ScribbleLossConfig(loss_config)

    loss_type = loss_config.loss_type

    if opt.sampler_type == "PLMSSampler":
        sampler = PLMSSampler(
            models=models,
            loss_scheduler=loss_scheduler,
            loss_config=loss_config,
            loss_type=loss_type,
            schedule="linear",
            alpha_generator_func=alpha_generator_func, 
            set_alpha_scale=set_alpha_scale,
            sd_weights_path=opt.sd_weights_path if "sd_weights_path" in opt else None,
            verbose=opt.verbose,
            save_vis=opt.save_vis,
            generate_source=opt.generate_source,
            vis_cross_res=opt.vis_cross_res,
            vis_self_res=opt.vis_self_res,
            device=device
        )
    
    num_repeat = opt.n_repeat
    
    if not opt.prompt_from_file:
        print("Only one prompt is specified.")
        prompt = opt.prompt
        if opt.prompt == None:
            prompt = input("Please enter a single prompt: ")
            
        num_repeat = opt.n_repeat
        prompt_input = PromptInput(
            batch_size=num_repeat,
            prompts=[prompt] * num_repeat,
            stroke_dirs=[opt.stroke_dir] * num_repeat,
            output_dirs=[opt.output_dir] * num_repeat,
            save_scribble_dirs=[opt.save_scribble_dir] * num_repeat,
            save_mask_dirs=[opt.save_mask_dir] * num_repeat,
            vis_dirs=[opt.vis_dir] * num_repeat,
            scribble_res=opt.scribble_res,
            device=device
        )
        prompt_inputs = [prompt_input]
    else:
        print(f"Reading prompts from {opt.prompt_from_file}") if opt.verbose else None
        with open(opt.prompt_from_file, "r") as f:
            inputs = json.load(f)
            
            prompt_inputs = []
            for input_ in inputs:
                prompt_input = PromptInput(
                    num_repeat, 
                    prompts=[input_["prompt"]] * num_repeat,
                    stroke_dirs=[input_["stroke_dir"]] * num_repeat,
                    output_dirs=[input_["output_dir"]] * num_repeat,
                    save_scribble_dirs=[input_["save_scribble_dir"]] * num_repeat,
                    save_mask_dirs=[input_["save_mask_dir"]] * num_repeat,
                    vis_dirs=[input_["vis_dir"]] * num_repeat,
                    scribble_res=opt.scribble_res,
                    device=device
                )
                prompt_inputs.append(prompt_input)
    
    
    self_attn_aggregator, scribble_propagator = None, None
    if opt.do_propagation:
        self_attn_aggregator = SelfAttnAggregator(
            src_res=opt.agg_src_res,
            tgt_res=opt.agg_tgt_res,
            device=device
        )
        scribble_propagator = ScribblePropagator(
            prop_config_file=prop_config,
            device=device
        )
    
    for prompt_input in prompt_inputs:
        preprocess_prompt_input(
            prompt_input, 
            text_encoder, 
            clip_model, 
            clip_processor,
            grounding_tokenizer_input
        )
        
        if opt.do_propagation:
            self_attn_aggregator.batch_size = prompt_input.batch_size
            scribble_propagator.batch_size = prompt_input.batch_size
            scribble_propagator.individual_num = prompt_input.max_num_individuals
        
        x_T = torch.randn(num_repeat, 4, 64, 64).to(opt.device)

        x_0 = run(
            models=models,
            steps=opt.steps,
            scale=opt.scale,
            opt=opt,
            sampler=sampler,
            prompt_input=prompt_input,
            x_T=x_T,
            self_attn_aggregator=self_attn_aggregator,
            scribble_propagator=scribble_propagator,
            device=device
        )
        
        for batch in range(prompt_input.batch_size):
            output_dir = prompt_input.output_dirs[batch]
            # count only '.png' extension file
            start_idx = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
            img_id = start_idx + 1
            output_name = prompt_input.prompts[batch].replace(" ", "_")

            sample = torch.clamp(x_0[batch], -1, 1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
            sample = Image.fromarray(sample.astype(np.uint8))
            
            sample_path = os.path.join(output_dir, "{}({}).png".format(output_name[:100], img_id))
            sample.save(sample_path)

            # save config file
            save_config_path = os.path.join(output_dir, "configs")
            os.makedirs(save_config_path, exist_ok=True)
            
            with open(os.path.join(save_config_path, f"config({img_id}).json"), "w") as f:
                json.dump(config_file, f, indent=4)

    
if __name__ == "__main__":
    opt = parse_args()
    assert os.path.exists(opt.config_from_file), "Please specify the path to the config file."
    
    config_file = json.load(open(opt.config_from_file, "r"))
    
    for k, v in config_file["config"].items():
        setattr(opt, k, v)
    
    main(opt, config_file)