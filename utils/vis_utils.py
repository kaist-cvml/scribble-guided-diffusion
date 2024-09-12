import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.nn import functional as F
from sklearn.decomposition import PCA
from typing import Optional, Union, List
from losses.utils import get_all_self_attn, get_all_cross_attn


def resize_and_pad(image, target_size, return_coords=False):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if original_width > original_height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_size, target_size), color="black")

    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    # compute valid coordinates for the resized image
    valid_x_min = paste_x
    valid_y_min = paste_y
    valid_x_max = paste_x + new_width
    valid_y_max = paste_y + new_height

    if return_coords:
        return new_image, (valid_y_min, valid_x_min, valid_y_max, valid_x_max)

    return new_image

def vis_all_cross_attn(in_cross_attn_list,
                   mid_cross_attn_list,
                   out_cross_attn_list,
                   object_positions,
                   res,
                   vis_dir,
                   tokens,
                   step):
    
    if type(res) is int:
        res = [res]

    all_attn = get_all_cross_attn(in_cross_attn_list, mid_cross_attn_list, out_cross_attn_list, res)
    
    for r in res:
        attn_maps = all_attn[r]

        cross_attn = torch.stack(attn_maps, dim=0).mean(0)
        cross_attn_text = cross_attn[:, :, :, 1:-1].detach()
        cross_attn_text = cross_attn_text.permute(0, 3, 1, 2)

        b, c, h, w = cross_attn_text.shape
        _, n = object_positions.shape

        obj_true_pos = object_positions - 1

        obj_pos = obj_true_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        cross_attn_text = torch.gather(cross_attn_text, dim=1, index=obj_pos)

        for batch in range(b):
            for token_idx in range(n):
                cross_attn_map_obj = cross_attn_text[batch, token_idx]
                cross_attn_map = cross_attn_map_obj.detach().cpu().numpy()
                
                cross_attn_map = (cross_attn_map - cross_attn_map.min()) / (cross_attn_map.max() - cross_attn_map.min())

                cross_attn_map_npy = np.expand_dims(cross_attn_map, axis=0).repeat(3, axis=0)
                cross_attn_map_npy = cross_attn_map_npy.transpose(1, 2, 0) * 255

                cross_attn_map_img = Image.fromarray(cross_attn_map_npy.astype(np.uint8))
                cross_attn_map_img = cross_attn_map_img.filter(ImageFilter.GaussianBlur(radius=1))
                cross_attn_map_img = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)(cross_attn_map_img)
                
                obj_idx = object_positions[batch, token_idx].item()

                cross_save_dir = f'{vis_dir}/cross_attn/{batch}/{r}/{tokens[batch][obj_idx]}_{obj_idx}'
                if not os.path.exists(cross_save_dir):
                    os.makedirs(cross_save_dir)

                cross_attn_map_img.save(f'{cross_save_dir}/cross_attn_{step}.png')


def pca_self_attn(
        self_attn_npy, 
        res=64,
        n_components=3
    ):

    # if self attention has multi-heads
    if len(self_attn_npy.shape) == 3:
        mean_self_attn_npy = np.mean(self_attn_npy, axis=0)
        self_attn_npy = mean_self_attn_npy.reshape(self_attn_npy.shape[-2], self_attn_npy.shape[-1])

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(self_attn_npy)

    pca_result = pca_result.reshape(res, res, 3)
    pca_result = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    pca_result = pca_result.astype(np.uint8)

    return pca_result


def vis_all_self_attn(
        in_self_attn_list,
        mid_self_attn_list,
        out_self_attn_list,
        res,
        vis_dir,
        step
    ):
    if type(res) is int:
        res = [res]

    all_self_attn = get_all_self_attn(in_self_attn_list, mid_self_attn_list, out_self_attn_list, res)

    for r in res:
        self_attn_maps = all_self_attn[r * r]

        self_attn_maps = torch.stack(self_attn_maps, dim=0).mean(0)
        for batch in range(self_attn_maps.shape[0]):
            self_attn = self_attn_maps[batch]
            self_attn_npy = self_attn.detach().cpu().numpy()

            pca_result = pca_self_attn(self_attn_npy, r)

            pca_img = Image.fromarray(pca_result)
            pca_img = pca_img.filter(ImageFilter.GaussianBlur(radius=1))
            pca_img = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)(pca_img)
            
            self_save_dir = f'{vis_dir}/self_attn/{batch}/{r}'
            if not os.path.exists(self_save_dir):
                os.makedirs(self_save_dir)

            pca_img.save(f'{self_save_dir}/self_attn_{step}.png')

