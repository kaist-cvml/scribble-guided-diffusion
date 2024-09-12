import math
import torch
import numpy as np

from PIL import Image
from torch.nn import functional as F

from ldm.models.diffusion.gaussian_smoothing import GaussianSmoothing


def loss_self_attn(
        attn_map, 
        masks, 
        scribbles,
        top_k=0,
        use_outside_mask_loss=True,
        min_top_k_obj_num=False,
    ):
    b, obj_num, _, _ = masks.shape
    n, i, j = attn_map.shape
    
    H = W = int(math.sqrt(i))
    
    if top_k > 0:
        if min_top_k_obj_num:
            top_k = min(top_k, obj_num)
        else:
            top_k = max(top_k, obj_num)
    
    loss = torch.tensor(0., device=attn_map.device)

    scribbles_resized = F.interpolate(scribbles, size=(H, W), mode='bilinear', align_corners=False).bool().float()

    scribbles_reshaped = scribbles_resized.reshape(b, obj_num, H * W)
    scribble_indices = torch.nonzero(scribbles_reshaped, as_tuple=False)

    attn_maps = attn_map.unsqueeze(1).repeat(1, obj_num, 1, 1)

    attn_filtered = torch.zeros_like(attn_maps)
    attn_filtered[scribble_indices[:, 0], scribble_indices[:, 1], scribble_indices[:, 2]] = attn_maps[scribble_indices[:, 0], scribble_indices[:, 1], scribble_indices[:, 2]]

    if use_outside_mask_loss:
        masks_resized = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False).bool().float()
        masks_out = 1. - masks_resized
        masks_or = torch.any(masks_resized, dim=1, keepdim=True).float().repeat(1, obj_num, 1, 1)
        masks_out = torch.logical_and(masks_out, masks_or).float()
        masks_out= masks_out.reshape(b, obj_num, H * W).unsqueeze(2).repeat(1, 1, H * W, 1)
        
        attn_filtered = attn_filtered * masks_out
    else: 
        scribbles_out = 1. - scribbles_resized
        scribbles_or = torch.any(scribbles_resized, dim=1, keepdim=True).float().repeat(1, obj_num, 1, 1)
        scribbles_out = torch.logical_and(scribbles_out, scribbles_or).float()
        
        scribbles_out= scribbles_out.reshape(b, obj_num, H * W).unsqueeze(2).repeat(1, 1, H * W, 1)
        
        attn_filtered = attn_filtered * scribbles_out


    if top_k > 0:
        topk_values, _ = torch.topk(attn_filtered, k=top_k, dim=2, largest=True, sorted=False)

        topk_mean = topk_values.mean(dim=2)
        loss = topk_mean.mean()
    else:
        act_val = attn_filtered.sum(dim=(1, 2, 3)) / obj_num
        loss = act_val.mean()
            
    return loss



def compute_loss_self_attn(
        all_self_attn,
        masks,
        scribbles,
        res=[16],
        top_k=0,
        use_outside_mask_loss=True,
        min_top_k_obj_num=False,
        device='cuda'
    ):

    if type(res) == int:
        res = [res]

    res_cnt = 0
    total_loss = torch.tensor(0., device=device)

    for r in res:
        attn_maps = all_self_attn[r * r]

        # compute mean of list of attn_maps
        attn_map = torch.stack(attn_maps, dim=0).mean(0)
        
        total_loss += loss_self_attn(
            attn_map, masks, scribbles, top_k=top_k,
            use_outside_mask_loss=use_outside_mask_loss, min_top_k_obj_num=min_top_k_obj_num    
        )
        res_cnt += 1

    return total_loss / res_cnt


def compute_cross_attn_outside_mask_loss(attn_obj, masks_resized, top_k=0):
    attn_masks_outside = (attn_obj * ~masks_resized)
    
    if top_k > 0:
        attn_masks_outside_flat = attn_masks_outside.flatten(start_dim=2)
        topk_values_outside, _ = torch.topk(attn_masks_outside_flat, k=top_k, dim=2, largest=True, sorted=False)
        outside_mask_loss = topk_values_outside.mean()
    else:
        attn_max_masks_outside = attn_masks_outside.max(dim=3)[0].max(dim=2)[0]
        outside_mask_loss = attn_max_masks_outside.mean()

    return outside_mask_loss


def compute_cross_attn_outside_scribble_loss(attn_obj, scribbles_resized, masks_resized=None, top_k=0, not_largest=True):
    if masks_resized is not None:
        attn_scribbles_outside = (attn_obj * ~scribbles_resized * masks_resized)
    else:
        attn_scribbles_outside = (attn_obj * ~scribbles_resized)
    
    if top_k > 0:
        attn_scribbles_outside_flat = attn_scribbles_outside.flatten(start_dim=2)
        if not_largest:
            topk_values_outside, _ = torch.topk(attn_scribbles_outside_flat[attn_scribbles_outside_flat > 0], k=top_k, dim=0, largest=False, sorted=False)
        else:
            topk_values_outside, _ = torch.topk(attn_scribbles_outside_flat, k=top_k, dim=2, largest=True, sorted=False)
        outside_scribble_loss = topk_values_outside.mean()
    else:
        if not_largest:
            attn_max_scribbles_outside = attn_scribbles_outside[attn_scribbles_outside_flat > 0].min(dim=0)
        else:
            attn_max_scribbles_outside = attn_scribbles_outside.max(dim=3)[0].max(dim=2)[0]
        outside_scribble_loss = attn_max_scribbles_outside.mean()

    return outside_scribble_loss

def compute_cross_attn_inside_scribble_loss(attn_obj, scribbles_resized, top_k=0):
    attn_scribbles_inside = (attn_obj * scribbles_resized)
    
    if top_k > 0:
        attn_scribbles_inside_flat = attn_scribbles_inside.flatten(start_dim=2)
        topk_values_inside, _ = torch.topk(attn_scribbles_inside_flat, k=top_k, dim=2, largest=True, sorted=False)
        inside_scribble_loss = (1.0 - topk_values_inside.mean())  # Mean over top-K values
    else:
        attn_max_scribbles_inside = attn_scribbles_inside.max(dim=3)[0].max(dim=2)[0]
        inside_scribble_loss = (1.0 - attn_max_scribbles_inside.mean())

    return inside_scribble_loss

def compute_focal_loss(attn_obj, scribbles_resized, alpha=.25, beta=2.0):
    cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(
        attn_obj, scribbles_resized.float()
    )

    pt = torch.exp(-cross_entropy_loss)
    at = alpha * scribbles_resized.float() + (1. - alpha) * (1. - scribbles_resized.float())
    focal_loss = (1 - pt) ** beta * at * cross_entropy_loss

    focal_loss = focal_loss.mean()
    
    return focal_loss


def compute_cross_attn_loss(
        all_cross_attn,
        masks,
        scribbles,
        token_indices,
        cross_attn_res=[16],
        focal_loss_alpha=.25,
        focal_loss_beta=2.0,
        focal_loss_weight=3.0,
        smooth_attn=True,
        sigma=0.5,
        kernel_size=3,
        top_k=5,
        min_top_k_obj_num=False,
        use_outside_mask_loss=True,
        use_outside_scribble_loss=True,
        use_outside_scribble_loss_with_mask=True,
        outside_not_largest=True,
        use_inside_scribble_loss=True,
        use_focal_loss=True,
    ):

    if type(cross_attn_res) == int:
        cross_attn_res = [cross_attn_res]
    
    total_loss = 0

    b, token_num, _, _ = masks.shape

    # Generate GaussianSmoothing instance outside the loop
    if smooth_attn:
        smoothing = GaussianSmoothing(channels=token_num, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()

    if top_k > 0:
        if min_top_k_obj_num:
            top_k = min(top_k, token_num)
        else:
            top_k = max(top_k, token_num)
            
    token_true_indices = token_indices - 1
    
    for r in cross_attn_res:
        attn_maps = all_cross_attn[r] # Get the current resolution from attn shape
        H = W = r

        attn_map = torch.stack(attn_maps, dim=0).mean(0)
        n, h, w, c = attn_map.shape

        attn_map = attn_map.reshape(b, h, w, c)

        attn_text = attn_map[:, :, :, 1:-1]
        attn_text *= 100
        attn_text = torch.nn.functional.softmax(attn_text, dim=-1)

        token_true_indices_reshaped = token_true_indices.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)
        attn_token = torch.gather(attn_text, dim=-1, index=token_true_indices_reshaped)
        attn_token = attn_token.permute(0, 3, 1, 2).reshape(-1, token_num, H, W)

        if smooth_attn:
            input_ = F.pad(attn_token, (1, 1, 1, 1), mode='reflect')
            attn_token = smoothing(input_)

        masks_resized = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False).bool()
        scribbles_resized = F.interpolate(scribbles, size=(H, W), mode='bilinear', align_corners=False).bool()

        if use_outside_mask_loss:
            outside_mask_loss = compute_cross_attn_outside_mask_loss(attn_token, masks_resized, top_k=top_k)
            total_loss += outside_mask_loss

        if use_outside_scribble_loss:
            if use_outside_scribble_loss_with_mask:
                outside_scribble_loss = compute_cross_attn_outside_scribble_loss(
                    attn_token, scribbles_resized, masks_resized=masks_resized, top_k=top_k, not_largest=outside_not_largest
                )
            else:
                outside_scribble_loss = compute_cross_attn_outside_scribble_loss(
                    attn_token, scribbles_resized, masks_resized=None, top_k=top_k, not_largest=outside_not_largest
                )
            total_loss += outside_scribble_loss

        if use_inside_scribble_loss:
            inside_scribble_loss = compute_cross_attn_inside_scribble_loss(attn_token, scribbles_resized, top_k=top_k)
            total_loss += inside_scribble_loss

        if use_focal_loss:
            focal_loss = compute_focal_loss(attn_token, scribbles_resized, alpha=focal_loss_alpha, beta=focal_loss_beta)
            total_loss += focal_loss_weight * focal_loss

    return total_loss / len(cross_attn_res)


def get_moment(attn_obj, device='cuda'):
    batch_size, obj_num, h, w = attn_obj.shape
    x, y = torch.meshgrid(torch.arange(0, w, device=device), torch.arange(0, h, device=device), indexing='ij')

    x = x.expand(batch_size, obj_num, h, w)
    y = y.expand(batch_size, obj_num, h, w)
    
    m00 = torch.sum(attn_obj, dim=(-1, -2))
    m10 = torch.sum(x * attn_obj, dim=(-1, -2))
    m01 = torch.sum(y * attn_obj, dim=(-1, -2))

    eps = 1e-5
    x_avg = m10 / (m00 + eps)
    y_avg = m01 / (m00 + eps)
    
    mu20 = torch.sum((x - x_avg.unsqueeze(-1).unsqueeze(-1)) ** 2 * attn_obj, dim=(-1, -2))
    mu02 = torch.sum((y - y_avg.unsqueeze(-1).unsqueeze(-1)) ** 2 * attn_obj, dim=(-1, -2))
    mu11 = torch.sum((x - x_avg.unsqueeze(-1).unsqueeze(-1)) * (y - y_avg.unsqueeze(-1).unsqueeze(-1)) * attn_obj, dim=(-1, -2))

    angle = 0.5 * torch.atan2(2 * mu11, mu20 - mu02)

    angle[angle != angle] = 0.

    angle[angle < 0] += math.pi

    return angle, x_avg, y_avg


def compute_moment_loss(
        all_cross_attn,
        individual_scribbles,
        individual_masks,
        token_indices,
        individual_to_phrase,
        phrase_to_obj,
        cross_attn_res=[16],
        alpha=1,
        smooth_attn=True,
        sigma=0.5,
        kernel_size=3
    ):
    
    if type(cross_attn_res) == int:
        cross_attn_res = [cross_attn_res]
    
    b, token_num = token_indices.shape
    _, individual_num, _, _ = individual_scribbles.shape
    total_loss = 0
    epsilon = 1e-5

    # Generate GaussianSmoothing instance outside the loop
    if smooth_attn:
        smoothing = GaussianSmoothing(channels=token_num, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()

    for r in cross_attn_res:
        attn_maps = all_cross_attn[r]
        attn_map = torch.stack(attn_maps, dim=0).mean(0)

        attn_text = attn_map[:, :, :, 1:-1]
        attn_text *= 100
        attn_text = torch.nn.functional.softmax(attn_text, dim=-1)

        token_true_indices = token_indices - 1
        token_true_indices = token_true_indices.unsqueeze(1).unsqueeze(1).repeat(1, r, r, 1)

        attn_token = torch.gather(attn_text, dim=-1, index=token_true_indices)
        attn_token = attn_token.permute(0, 3, 1, 2).reshape(-1, token_num, r, r)
        
        if smooth_attn:
            input = F.pad(attn_token, (1, 1, 1, 1), mode='reflect')
            attn_token = smoothing(input)

        individual_masks_resized = F.interpolate(individual_masks, size=(r, r), mode='bilinear', align_corners=False).bool()
        individual_scribbles_resized = F.interpolate(individual_scribbles, size=(r, r), mode='bilinear', align_corners=False).bool().float()

        individual_loss = 0
        
        for i in range(individual_num):
            individual_token_indices = phrase_to_obj[0][individual_to_phrase[0][i]]
            individual_token_indices = torch.tensor(individual_token_indices, device=attn_token.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat(b, 1, r, r)
            
            individual_mask = individual_masks_resized[:, i, :, :].unsqueeze(1)
            individual_scribble = individual_scribbles_resized[:, i, :, :].unsqueeze(1)
            
            individual_attn_token = torch.gather(attn_token, dim=1, index=individual_token_indices)
            individual_attn_obj_masked = individual_attn_token * individual_mask

            individual_attn_obj_masked_min = individual_attn_obj_masked.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            individual_attn_obj_masked_max = individual_attn_obj_masked.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

            norm_attn_obj_masked = (individual_attn_obj_masked - individual_attn_obj_masked_min) / (individual_attn_obj_masked_max - individual_attn_obj_masked_min + epsilon)

            attn_angle, attn_centroid_x, attn_centroid_y = get_moment(norm_attn_obj_masked)

            attn_centroid_x /= r
            attn_centroid_y /= r

            scribbles_angle, scribbles_centroid_x, scribbles_centroid_y = get_moment(individual_scribble)

            scribbles_centroid_x /= r
            scribbles_centroid_y /= r

            centroid_loss = torch.sqrt((attn_centroid_x - scribbles_centroid_x)**2 + (attn_centroid_y - scribbles_centroid_y)**2)
            angle_loss = torch.abs(attn_angle - scribbles_angle)
            angle_loss = torch.min(angle_loss, 2 * math.pi - angle_loss) / (math.pi) 
            # square the angle loss (angle_loss has a shape of (batch, obj_num))
            angle_loss_squared = angle_loss ** 2

            individual_loss += (centroid_loss + angle_loss_squared * alpha).mean()
            # total_loss += (centroid_loss).mean()
            # total_loss += (angle_loss_squared).mean()

        total_loss += individual_loss / individual_num

    return total_loss / len(cross_attn_res)
    

def compute_feature_loss(
        features,
        all_cross_attn,
        individual_scribbles,
        individual_masks,
        token_indices,
        individual_to_phrase,
        phrase_to_obj,
        feat_indices=[4,5,6,7,8,9],
        cross_attn_res=[16],
        cross_attn_mask_res=16,
        cross_attn_mask_thres=0.2,
        smooth_attn=False,
        sigma=0.5,
        kernel_size=3,
        alpha=1.,
        beta=1.,
    ):
    
    def cosine_similarity(x, y):
        norm_x = x / x.norm(dim=-1, keepdim=True)
        norm_y = y / y.norm(dim=-1, keepdim=True)
        
        return norm_x @ norm_y.transpose(-1, 0)
    
    if type(feat_indices) == int:
        feat_indices = [feat_indices]
        
    if type(cross_attn_res) == int:
        cross_attn_res = [cross_attn_res]
     
    b, token_num = token_indices.shape   
    _, individual_num, _, _ = individual_masks.shape
    feat_num = len(feat_indices)
    
    total_loss = 0
    
    if smooth_attn:
        smoothing = GaussianSmoothing(channels=token_num, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
    
    cross_attn_mean = torch.zeros((b, token_num, cross_attn_mask_res, cross_attn_mask_res), device=individual_scribbles.device)
    token_true_indices = token_indices - 1
    
    res_sum = 0
    
    for res in cross_attn_res:
        attn_maps = all_cross_attn[res]
        attn_map = torch.stack(attn_maps, dim=0).mean(0)
        n, h, w, c = attn_map.shape
        
        attn_map = attn_map.reshape(b, h, w, c)
        
        attn_text = attn_map[:, :, :, 1:-1]
        # attn_text *= 100
        # attn_text = torch.nn.functional.softmax(attn_text, dim=-1)
        
        token_true_indices_reshaped = token_true_indices.unsqueeze(1).unsqueeze(1).repeat(1, res, res, 1)
        attn_token = torch.gather(attn_text, dim=-1, index=token_true_indices_reshaped)
        attn_token = attn_token.permute(0, 3, 1, 2).reshape(-1, token_num, res, res)
        
        if smooth_attn:
            input_ = F.pad(attn_token, (1, 1, 1, 1), mode='reflect')
            attn_token = smoothing(input_)
    
        attn_token_resized = F.interpolate(attn_token, size=(cross_attn_mask_res, cross_attn_mask_res), mode='bilinear', align_corners=False)

        cross_attn_mean += attn_token_resized * 1 / (res ** 2)
        res_sum += 1 / (res ** 2)
        
    cross_attn_mean /= res_sum

    # cross_attn_norm = (cross_attn_mean - cross_attn_mean_min) / (cross_attn_mean_max - cross_attn_mean_min + 1e-5)
    # cross_attn_mask = torch.where(cross_attn_norm > cross_attn_mask_thres, torch.ones_like(cross_attn_norm), torch.zeros_like(cross_attn_norm))
    
    cross_attn_mask = cross_attn_mean
    
    for feat_index in feat_indices:
        feat = features[feat_index]
        b, c, h, w = feat.shape
        
        cross_attn_mask_resized = F.interpolate(cross_attn_mask, size=(h, w), mode='bilinear', align_corners=False)
        individual_masks_resized = F.interpolate(individual_masks, size=(h, w), mode='bilinear', align_corners=False).bool()
        individual_scribbles_resized = F.interpolate(individual_scribbles, size=(h, w), mode='bilinear', align_corners=False).bool().float()
        
        for i in range(individual_num):
            individual_token_indices = phrase_to_obj[0][individual_to_phrase[0][i]]
            individual_token_indices = torch.tensor(individual_token_indices, device=feat.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat(b, 1, h, w)
            
            individual_mask = individual_masks_resized[:, i, :, :].unsqueeze(1)
            individual_scribble = individual_scribbles_resized[:, i, :, :].unsqueeze(1)
            
            token_cross_attn_mask = torch.gather(cross_attn_mask_resized, dim=1, index=individual_token_indices)
            token_cross_attn_mask = torch.mean(token_cross_attn_mask, dim=1, keepdim=True)
            
            token_cross_attn_mask_min = token_cross_attn_mask.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            token_cross_attn_mask_max = token_cross_attn_mask.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            
            token_cross_attn_mask = (token_cross_attn_mask - token_cross_attn_mask_min) / (token_cross_attn_mask_max - token_cross_attn_mask_min + 1e-5)
            
            token_cross_attn_mask = torch.where(token_cross_attn_mask > cross_attn_mask_thres, torch.ones_like(token_cross_attn_mask), torch.zeros_like(token_cross_attn_mask))
            individual_cross_attn_mask = token_cross_attn_mask * individual_mask
            token_cross_attn_mask_outside = ~token_cross_attn_mask.bool()
            
            # # temporary codes for visualization
            # vis_cross_attn_mask = (individual_cross_attn_mask * (1. - individual_scribble))[0, 0, :, :].detach().cpu().numpy()
            # vis_cross_attn_mask = (vis_cross_attn_mask * 255).astype(np.uint8)
            # vis_cross_attn_mask_img = Image.fromarray(vis_cross_attn_mask)
            
            # vis_temp_path = 'temp_vis'
            # vis_cross_attn_mask_img.save(vis_temp_path + '/cross_attn_mask_{}.png'.format(i))
            
            # vis_scribble = (individual_scribble * (1. - individual_cross_attn_mask))[0, 0, :, :].detach().cpu().numpy()
            # vis_scribble = (vis_scribble * 255).astype(np.uint8)
            # vis_scribble_img = Image.fromarray(vis_scribble)
            
            # vis_scribble_img.save(vis_temp_path + '/scribble_{}.png'.format(i))
            
            feat_src_mask = (individual_cross_attn_mask.bool() == 1) & (individual_scribble.bool() == 0)
            # feat_tgt_mask = (individual_cross_attn_mask.bool() == 0) & (individual_scribble.bool() == 1)
            feat_tgt_mask = (individual_scribble.bool() == 1)
            feat_mask_outside = (token_cross_attn_mask_outside.bool() == 1) & (individual_mask.bool() == 0)
            
            if feat_src_mask.sum() == 0 or feat_tgt_mask.sum() == 0 or feat_mask_outside.sum() == 0:
                continue
            
            masked_feat_src = feat[feat_src_mask.repeat(1, c, 1, 1).detach()]
            masked_feat_tgt = feat[feat_tgt_mask.repeat(1, c, 1, 1).detach()]
            masked_feat_outside = feat[feat_mask_outside.repeat(1, c, 1, 1).detach()]
            
            individual_feat_src = masked_feat_src.view(feat.shape[0], feat.shape[1], -1).mean(dim=-1)
            individual_feat_tgt = masked_feat_tgt.view(feat.shape[0], feat.shape[1], -1).mean(dim=-1)
            individual_feat_outside = masked_feat_outside.view(feat.shape[0], feat.shape[1], -1).mean(dim=-1)
            
            local_sim = 0.5 * (cosine_similarity(individual_feat_src.detach(), individual_feat_tgt) + 1)
            content_sim = 0.5 * (cosine_similarity(individual_feat_src, individual_feat_outside.detach()) + 1)
            
            total_loss += (1. - local_sim.mean() + 1. - content_sim.mean()) / 2.
            
    return total_loss / feat_num / individual_num / len(cross_attn_res)
            
            