import gc
import torch
import numpy as np
import torch.nn.functional as F

from copy import deepcopy

from losses.loss_scheduler import ScribbleLossScheduler
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from losses.utils import get_all_self_attn
from ldm.models.diffusion.sampler import Sampler

from data import PromptInput
from scribble_propagation import ScribblePropagator, SelfAttnAggregator


class PLMSSampler(Sampler):
    def __init__(
            self, 
            models,
            loss_scheduler:ScribbleLossScheduler=None,
            loss_config=None,
            loss_type=None,
            schedule="linear",        
            alpha_generator_func=None, 
            set_alpha_scale=None ,
            sd_weights_path=None,
            verbose=False,
            save_vis=False,
            generate_source=False,
            vis_cross_res=[8, 16, 32, 64],
            vis_self_res=[16],
            device=torch.device("cpu")
        ):
       
        super().__init__(
            models=models,
            loss_scheduler=loss_scheduler,
            loss_config=loss_config,
            loss_type=loss_type,
            verbose=verbose,
            save_vis=save_vis,
            generate_source=generate_source,
            vis_cross_res=vis_cross_res,
            vis_self_res=vis_self_res,
            device=device
        )
        
        self.ddpm_num_timesteps = self.diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale
        self.sd_weights_path = sd_weights_path

            
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self, x_t, S, shape, prompt_input, 
               c=None, uc=None, guidance_scale=1., mask=None, x0=None, 
               scribble_propagator=None, self_attn_aggregator=None, callback=None):
        self.make_schedule(ddim_num_steps=S)
        return self.plms_sampling(
            x_t, shape, prompt_input, c, uc, guidance_scale, 
            mask=mask, x0=x0, scribble_propagator=scribble_propagator, 
            self_attn_aggregator=self_attn_aggregator, callback=callback
        )


    def plms_sampling(
            self, 
            x_t,
            shape,
            prompt_input: PromptInput,
            c=None,
            uc=None, 
            guidance_scale=1, 
            mask=None, 
            x0=None,
            scribble_propagator:ScribblePropagator=None,
            self_attn_aggregator:SelfAttnAggregator=None,
            callback=None
        ):
        
        batch_size = shape[0]
        
        if x_t is not None:
            x = x_t
        else:
            x = torch.randn(shape, device=self.device)

        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        
        old_eps = []
        
        if self.generate_source:
            src_old_eps = []
            src_x = deepcopy(x)

        if self.alpha_generator_func != None:
            alphas = self.alpha_generator_func(len(time_range))
            
        loss = torch.tensor(10000.).to(self.device).requires_grad_(True)
            
        for step_num, step in enumerate(time_range):
            print(f"step: {step_num} / {total_steps}")
            # set alpha and restore first conv layer 
            if self.alpha_generator_func != None:
                self.set_alpha_scale(self.model, alphas[step_num])
                if alphas[step_num] == 0:
                    self.model.restore_first_conv_from_SD(sd_weights_path=self.sd_weights_path)

            # run 
            index = total_steps - step_num - 1
            timestep = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            next_timestep = torch.full((batch_size,), time_range[min(step_num + 1, len(time_range) - 1)], device=self.device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.diffusion.q_sample(x0, timestep) 
                src_x = img_orig * mask + (1. - mask) * src_x if self.generate_source else None
                x = img_orig * mask + (1. - mask) * x
            
            if self.generate_source:
                src_x, src_pred_z0, src_e_t, src_self_attns, src_cross_attns, src_feats = self.p_sample_plms(
                    x_t=src_x, cond=c, timestep=timestep, step_num=step_num, index=index, prompt_input=prompt_input, 
                    uc=uc, guidance_scale=guidance_scale, old_eps=src_old_eps, t_next=next_timestep, modulate=False
                )
            
            x_grad = None
            
            # if self.loss_type != None:
            if self.loss_type:
                if step_num < self.loss_end:
                    x, loss, x_grad = self.update_loss(x, c, timestep, step_num, prompt_input, loss=loss)
                    
            x, pred_z0, e_t, self_attns, cross_attns, feats = self.p_sample_plms(
                x_t=x, cond=c, timestep=timestep, step_num=step_num, index=index, prompt_input=prompt_input, 
                x_grad=x_grad, uc=uc, guidance_scale=guidance_scale, old_eps=old_eps, t_next=next_timestep, modulate=True
            )
            
            old_eps.append(e_t)
            src_old_eps.append(src_e_t) if self.generate_source else None
            
            if scribble_propagator != None:
                if self.propagation_start <= step_num and step_num < self.propagation_end:
                    in_self_attn_list, mid_self_attn_list, out_self_attn_list = self_attns
                    all_self_attn = get_all_self_attn(in_self_attn_list, mid_self_attn_list, out_self_attn_list)
                    agg_self_attn = self_attn_aggregator.aggregate_self_attn(all_self_attn)
                    
                    prompt_input.individual_scribbles = scribble_propagator.propagate_scribble(
                        agg_self_attn,
                        prompt_input.individual_scribbles, 
                        timestep,
                        prompt_input.individual_masks,
                    )
                    
                    prompt_input.update_token_tensors()

            if self.save_vis:
                self.visualize_process(
                    batch_size, src_pred_z0, prompt_input, step,
                    src_self_attns, src_cross_attns, is_source=True
                )  if self.generate_source else None
                
                self.visualize_process(
                    batch_size, pred_z0, prompt_input, step,
                    self_attns, cross_attns, is_source=False,
                    save_scribbles=self.propagation_start <= step_num and step_num < self.propagation_end,
                    save_masks=self.propagation_start <= step_num and step_num < self.propagation_end
                )
                
            if len(old_eps) >= 4:
                src_old_eps.pop(0) if self.generate_source else None
                old_eps.pop(0)

        return x


    @torch.no_grad()
    def p_sample_plms(self, x_t, cond, timestep, step_num, index, prompt_input:PromptInput=None,
                      x_grad=None, guidance_scale=1., uc=None, old_eps=None, t_next=None, modulate=False):
        x = deepcopy(x_t)
        
        b = x.shape[0]

        def get_model_output(x, timestep, ret_attn_list=False):
            if prompt_input is not None:
                if self.modulation_start <= step_num and step_num < self.modulation_end and modulate:
                    mod_masks = None
                    if self.mod_with_masks:
                        mod_masks = prompt_input.masks
                    else:
                        mod_masks = prompt_input.scribbles
                    e_t, self_attns, cross_attns, feats = self.model(
                            x, timestep, cond, prompt_input.grounding_inputs, 
                            mod_masks=mod_masks,
                            token_indices=prompt_input.token_indices,
                            mod_res=self.mod_res, mod_pos=self.mod_pos,
                            self_reg=self.mod_self_reg, cross_reg=self.mod_cross_reg,
                            ret_attn_list=True)
                else:
                    e_t, self_attns, cross_attns, feats = self.model(
                        x, timestep, cond, prompt_input.grounding_inputs, ret_attn_list=True
                    )
            else:
                e_t, self_attns, cross_attns, feats = self.model(
                    x, timestep, cond, ret_attn_list=True
                )
            if uc is not None and guidance_scale != 1:
                e_t_uncond, _, _, _ = self.model(
                    x, timestep, uc, ret_attn_list=True
                )
                e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            
            # e_t = e_t + x_grad if x_grad is not None else e_t
            
            if ret_attn_list:
                return e_t, self_attns, cross_attns, feats
            
            return e_t

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
            a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
            sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn_like(x)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        t = timestep 
        e_t, self_attns, cross_attns, feats = get_model_output(
                 x, timestep=t, ret_attn_list=True
            )
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            x = x_prev
            t = t_next
            e_t_next = get_model_output(x, t)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t, self_attns, cross_attns, feats


