import os
import gc
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from data import PromptInput

from losses.loss_config import ScribbleLossConfig
from losses.loss_scheduler import ScribbleLossScheduler

from losses.loss import compute_cross_attn_loss, compute_loss_self_attn, compute_moment_loss, compute_feature_loss
from losses.utils import get_all_self_attn, get_all_cross_attn
from utils.vis_utils import vis_all_cross_attn, vis_all_self_attn


class Sampler(object):
    def __init__(
            self, 
            models, 
            loss_scheduler:ScribbleLossScheduler=None,
            loss_config:ScribbleLossConfig=None,
            loss_type=None, 
            verbose=False,
            save_vis=False,
            generate_source=False,
            vis_cross_res=[8, 16, 32, 64],
            vis_self_res=[16],
            device=torch.device("cpu")
        ):
        super().__init__()

        self.model = models['model']
        self.diffusion = models['diffusion']
        self.autoencoder = models['autoencoder']
        
        self.loss_scheduler = loss_scheduler
        self.loss_config = loss_config
        self.loss_type = loss_type
        self.verbose = verbose
        self.device = device
        
        self.save_vis = save_vis
        self.vis_cross_res = vis_cross_res
        self.vis_self_res = vis_self_res
        
        self.generate_source = generate_source
        
        self._update_attributes(loss_config)
        self._update_attributes(loss_scheduler)
        
        
    def _update_attributes(self, source_obj):
        if source_obj is not None:
            for attr, value in vars(source_obj).items():
                setattr(self, attr, value)

    def compute_loss(
            self, 
            self_attns, 
            cross_attns, 
            features,
            step_num,
            masks, 
            scribbles, 
            token_indices,
            individual_masks=None, 
            individual_scribbles=None, 
            individual_to_phrase=None, 
            phrase_to_obj=None,
            loss_self_scale=1., 
            loss_cross_scale=1., 
            loss_moment_scale=1.,
            loss_feature_scale=1.
        ):
        loss = torch.tensor(0.).to(self.device)

        in_self_attn_list, mid_self_attn_list, out_self_attn_list = self_attns
        in_cross_attn_list, mid_cross_attn_list, out_cross_attn_list = cross_attns
        
        all_self_attn = get_all_self_attn(in_self_attn_list, mid_self_attn_list, out_self_attn_list, self.self_attn_loss_res)
        all_cross_attn = get_all_cross_attn(in_cross_attn_list, mid_cross_attn_list, out_cross_attn_list, self.cross_attn_loss_res)

        if self.self_loss_start <= step_num and step_num < self.self_loss_end:
            if 'self' in self.loss_type or 'all' in self.loss_type:
                self_attn_loss = compute_loss_self_attn(
                    all_self_attn=all_self_attn,
                    masks=masks,
                    scribbles=scribbles,
                    res=self.self_attn_loss_res,
                    use_outside_mask_loss=self.use_outside_mask_self_loss,
                    top_k=self.top_k,
                    min_top_k_obj_num=self.min_top_k_obj_num,
                    device=self.device
                ) 
                loss += self_attn_loss * loss_self_scale
                if loss_self_scale > 0:
                    print('self attention loss: ', self_attn_loss.item() / loss_self_scale) if self.verbose else None

        if (self.cross_loss_start <= step_num and step_num < self.cross_loss_end) or \
            (self.focal_loss_start <= step_num and step_num < self.focal_loss_end):
            if 'cross' in self.loss_type or 'all' in self.loss_type:
                use_outside_mask_loss =  self.cross_loss_start <= step_num and step_num < self.cross_loss_end and self.use_outside_mask_loss
                use_outside_scribble_loss = self.cross_loss_start <= step_num and step_num < self.cross_loss_end and self.use_outside_scribble_loss
                use_inside_scribble_loss = self.cross_loss_start <= step_num and step_num < self.cross_loss_end and self.use_inside_scribble_loss

                use_focal_loss = self.focal_loss_start <= step_num and step_num < self.focal_loss_end

                cross_attn_loss = compute_cross_attn_loss(
                    all_cross_attn=all_cross_attn, 
                    masks=masks,
                    scribbles=scribbles,
                    token_indices=token_indices,
                    cross_attn_res=self.cross_attn_loss_res,
                    focal_loss_alpha=self.cross_focal_loss_alpha,
                    focal_loss_beta=self.cross_focal_loss_beta,
                    focal_loss_weight=self.cross_focal_loss_weight,
                    smooth_attn=self.smooth_attn,
                    sigma=self.sigma,
                    kernel_size=self.kernel_size,
                    top_k=self.top_k,
                    min_top_k_obj_num=self.min_top_k_obj_num,
                    use_outside_mask_loss=use_outside_mask_loss,
                    use_outside_scribble_loss=use_outside_scribble_loss,
                    use_outside_scribble_loss_with_mask=self.use_outside_scribble_loss_with_mask,
                    outside_not_largest=self.outside_not_largest,
                    use_inside_scribble_loss=use_inside_scribble_loss,
                    use_focal_loss=use_focal_loss,
                ) 
                loss += cross_attn_loss * loss_cross_scale
                if loss_cross_scale > 0:
                    print('cross attention loss: ', cross_attn_loss.item() / loss_cross_scale) if self.verbose else None

        if self.moment_loss_start <= step_num and step_num < self.moment_loss_end: 
            if 'moment' in self.loss_type or 'all' in self.loss_type:
                moment_loss = compute_moment_loss(
                    all_cross_attn=all_cross_attn,
                    individual_scribbles=individual_scribbles,
                    individual_masks=individual_masks,
                    token_indices=token_indices,
                    individual_to_phrase=individual_to_phrase,
                    phrase_to_obj=phrase_to_obj,
                    cross_attn_res=self.cross_attn_loss_res,
                    alpha=self.moment_loss_alpha,
                    smooth_attn=self.smooth_attn,
                    sigma=self.sigma,
                    kernel_size=self.kernel_size,
                )
                loss += moment_loss * loss_moment_scale
                if loss_moment_scale > 0:
                    print('moment loss: ', moment_loss.item() / loss_moment_scale) if self.verbose else None

        if self.feature_loss_start <= step_num and step_num < self.feature_loss_end:
            if 'feature' in self.loss_type or 'all' in self.loss_type:
                feature_loss = compute_feature_loss(
                    features=features,
                    all_cross_attn=all_cross_attn,
                    individual_scribbles=individual_scribbles,
                    individual_masks=individual_masks,
                    token_indices=token_indices,
                    individual_to_phrase=individual_to_phrase,
                    phrase_to_obj=phrase_to_obj,
                    feat_indices=[6, 9],
                    cross_attn_res=self.cross_attn_loss_res,
                )
                loss += feature_loss * loss_feature_scale
                if loss_feature_scale > 0:
                    print('feature loss: ', feature_loss.item() / loss_feature_scale) if self.verbose else None

        return loss

    def update_loss(
            self, 
            x_t, 
            cond, 
            timestep, 
            step_num, 
            prompt_input: PromptInput, 
            loss=None
        ):
        assert self.loss_type is not None, "loss_type is not specified."
        
        x = deepcopy(x_t) 
        iteration = 0
        
        loss_schedule = self.loss_scheduler.schedule(step_num)
        loss_self_scale = loss_schedule['loss_self_scale']
        loss_cross_scale = loss_schedule['loss_cross_scale']
        loss_moment_scale = loss_schedule['loss_moment_scale']
        loss_feature_scale = loss_schedule['loss_feature_scale']
        
        max_iter = loss_schedule['max_iter']
        step_size = loss_schedule['step_size']
        
        masks = prompt_input.masks
        scribbles = prompt_input.scribbles
        token_indices = prompt_input.token_indices
        individual_scribbles = prompt_input.individual_scribbles
        individual_masks = prompt_input.individual_masks
        individual_to_phrase = prompt_input.individual_to_phrase
        phrase_to_obj = prompt_input.phrase_to_obj
        
        loss = torch.tensor(0.).to(self.device).requires_grad_(True) if loss is None else loss
        
        x_grad = None
        
        while (loss.item() > self.loss_threshold or step_num == 0) and iteration < max_iter:
            print('Iteration', iteration) if self.verbose else None
            
            x = x.requires_grad_(True)
            
            # optimizer = torch.optim.Adam([x], lr=0.001 * step_size)
            
            self.model.zero_grad()
  
            e_t, self_attns, cross_attns, feats = self.model(
                    x, timestep, cond, grounding_inputs=prompt_input.grounding_inputs, ret_attn_list=True
            )

            loss = self.compute_loss(
                self_attns, cross_attns, feats, step_num, masks, scribbles, token_indices,
                individual_masks, individual_scribbles, individual_to_phrase, phrase_to_obj,
                loss_self_scale, loss_cross_scale, loss_moment_scale, loss_feature_scale,
            )

            print('loss: ', loss.item()) if self.verbose else None
            
            if loss.item() > self.loss_threshold:
                torch.autograd.backward(loss, retain_graph=True)
                x_grad = x.grad
                # optimizer.step()
                x = x - step_size * x_grad
                # optimizer.zero_grad()
                x = x.detach()

                iteration += 1
 
            gc.collect()
            torch.cuda.empty_cache()
            
        return x, loss, x_grad
    
    
    def visualize_process(
            self, 
            batch_size, 
            pred_z0, 
            prompt_input:PromptInput, 
            step, 
            self_attns, 
            cross_attns,
            is_source=False,
            save_scribbles=False,
            save_masks=False
        ):
        # visualizing predicted x_0
        with torch.no_grad():
            pred_x0 = self.autoencoder.decode(pred_z0)

        pred_img = torch.clamp(pred_x0, -1, 1) * 0.5 + 0.5
            
        vis_dir = prompt_input.vis_dirs[0]
        
        if is_source:
            vis_dir = f'{vis_dir}/source'
        else:
            vis_dir = f'{vis_dir}/target'
        
        if not is_source:
            if save_scribbles:
                vis_scribble_dir = f'{vis_dir}/scribbles'
                prompt_input.save_scribbles(save_individual=True, save_scribble_dir=vis_scribble_dir, timestep=step)
            if save_masks:
                vis_mask_dir = f'{vis_dir}/masks'
                prompt_input.save_masks(save_individual=True, save_mask_dir=vis_mask_dir, timestep=step)
            
        for batch in range(batch_size):
            sample_save_dir = f'{vis_dir}/pred_x0/{batch}'
            if not os.path.exists(sample_save_dir):
                os.makedirs(sample_save_dir)
        
            sample_batch = pred_img[batch].detach().cpu().numpy().transpose(1, 2, 0) * 255
            Image.fromarray(sample_batch.astype(np.uint8)).save(f'{sample_save_dir}/{step}.png')

        in_self_attn_list, mid_self_attn_list, out_self_attn_list = self_attns
        in_cross_attn_list, mid_cross_attn_list, out_cross_attn_list = cross_attns
        
        # visualizing self attention maps with PCA
        vis_all_self_attn(in_self_attn_list, mid_self_attn_list, out_self_attn_list, 
                        self.vis_self_res, vis_dir, step)
        
        # visualizing cross attention maps
        vis_all_cross_attn(in_cross_attn_list, mid_cross_attn_list,out_cross_attn_list,
                        prompt_input.token_indices, self.vis_cross_res, vis_dir, prompt_input.tokens, step)

    