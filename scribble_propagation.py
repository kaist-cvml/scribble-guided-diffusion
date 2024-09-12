import os
import cv2
import torch
import numpy as np

from PIL import Image
from torch.nn import functional as F


class ScribblePropagator(object):
    
    def __init__(
            self, 
            batch_size=1, 
            individual_num=1, 
            anchor_res=32,
            top_k=30,
            threshold=0.0001,
            do_decay=False,
            masks_limit=False,
            decay_factor=1,
            prop_config_file=None,
            device=torch.device("cpu")
        ):
        super().__init__()
        
        self.batch_size = batch_size
        self.individual_num = individual_num
        self.anchor_res = anchor_res
        self.top_k = top_k
        self.threshold = threshold
        self.do_decay = do_decay
        self.masks_limit = masks_limit
        self.decay_factor = decay_factor
        self.device = device
        
        if prop_config_file is not None:
            for key, value in prop_config_file.items():
                assert hasattr(self, key), f"Invalid key {key} in loss_config_file."
                setattr(self, key, value)
        
        # visited array for each batch
        # self.visited = torch.zeros(batch_size, anchor_res, anchor_res, device=device)
        
        # self.scribbles_edge = torch.zeros(batch_size, obj_num, anchor_res, anchor_res, device=device)

        return
    
    def compute_distance(self, self_attn1, self_attn2):
        softmax_self_attn1 = F.softmax(self_attn1.reshape(-1), dim=-1)
        softmax_self_attn2 = F.softmax(self_attn2.reshape(-1), dim=-1)
        
        kl_div1 = softmax_self_attn1 * torch.log(softmax_self_attn1 / softmax_self_attn2)
        kl_div2 = softmax_self_attn2 * torch.log(softmax_self_attn2 / softmax_self_attn1)
        
        kl_div1 = kl_div1.sum()
        kl_div2 = kl_div2.sum()
        
        return (kl_div1 + kl_div2) / 2.
        

    
    def get_neighbors(self, idx, h, w):
        neighbors = []
        
        i, j = idx
        
        if i > 0:
            neighbors.append((i-1, j))
        if i < h - 1:
            neighbors.append((i+1, j))
        if j > 0:
            neighbors.append((i, j-1))
        if j < w - 1:
            neighbors.append((i, j+1))

        return neighbors
    
    
    def initialize_propagation(
            self, 
            scribbles
        ):
        b, o, h, w = scribbles.shape
        
        scribbles_resized = scribbles
        if h != self.anchor_res:
            scribbles_resized = F.interpolate(scribbles, size=(self.anchor_res, self.anchor_res), mode='bilinear', align_corners=False)
            scribbles_resized = scribbles_resized.bool().float()
        
        scribbles_indices = torch.nonzero(scribbles_resized, as_tuple=False)
        
        self.visited = torch.zeros(self.batch_size, self.anchor_res, self.anchor_res, device=self.device)
        self.visited[scribbles_indices[:, 0], scribbles_indices[:, 2], scribbles_indices[:, 3]] = scribbles_indices[:, 1].float() + 1.
        
        self.scribbles_edge = torch.zeros(self.batch_size, o, self.anchor_res, self.anchor_res, device=self.device)

        for (b, o, i, j) in scribbles_indices:
            neighbors = self.get_neighbors((i, j), self.anchor_res, self.anchor_res)
            for (ni, nj) in neighbors:
                if self.visited[b, ni, nj] != o + 1:
                    self.scribbles_edge[b, o, i, j] = 1
                    break
        return
    
    
    def propagate_scribble(
            self, 
            agg_self_attn, 
            scribbles, 
            timestep,
            masks=None
        ):
        # agg_self_attn shape: (batch_size, H, W, H, W)
        _, agg_self_h, agg_self_w, H, W = agg_self_attn.shape
        _, individual_num, scribble_h, scribble_w = scribbles.shape
        assert agg_self_h % self.anchor_res == 0 and agg_self_h == agg_self_w and H == W, "h must be divisible by anchor_res and h must be equal to w and H must be equal to W"
        
        top_k = self.top_k
        
        if self.do_decay:
            top_k = max(int(top_k * (timestep[0].item() / 1000.) ** self.decay_factor), self.individual_num)
        
        if masks is not None and self.masks_limit:
            masks_resized = F.interpolate(masks, size=(self.anchor_res, self.anchor_res), mode='bilinear', align_corners=False).bool().float()
        
        delta = agg_self_h // self.anchor_res
        
        scribbles_resized = scribbles
        if scribbles.shape[2] != self.anchor_res:
            scribbles_resized = F.interpolate(scribbles, size=(self.anchor_res, self.anchor_res), mode='bilinear', align_corners=False)
        
        anchor_self_attn = agg_self_attn.reshape(self.batch_size, self.anchor_res, delta, self.anchor_res, delta, H, W)
        anchor_self_attn = anchor_self_attn.mean(dim=(2, 4))
        
        edge_indices = torch.nonzero(self.scribbles_edge, as_tuple=False)
        neighbor_distance = torch.fill_(torch.zeros(self.batch_size, self.anchor_res, self.anchor_res), float('inf'))
        neighbor_obj = torch.zeros(self.batch_size, self.anchor_res, self.anchor_res, device=self.device)
        
        scribble_self_attn = torch.zeros(self.batch_size, individual_num, H, W, device=self.device)
        
        for b in range(self.batch_size):
            for o in range(individual_num):
                scribble_indices = torch.nonzero(scribbles_resized[b, o], as_tuple=False)
                if scribble_indices.shape[0] == 0:
                    continue
                scribble_self_attn[b, o] = anchor_self_attn[b, scribble_indices[:, 0], scribble_indices[:, 1]].mean(dim=0)
        
        
        for (b, o, i, j) in edge_indices:
            neighbors = self.get_neighbors((i, j), self.anchor_res, self.anchor_res)
            
            min_dist = float('inf')
            min_anchor = None
            
            for (ni, nj) in neighbors:
                if self.masks_limit:
                    if masks_resized[b, o, ni, nj] == 0:
                        continue
                if self.visited[b, ni, nj] == 0:
                    # dist = self.compute_distance(anchor_self_attn[b, i, j], anchor_self_attn[b, ni, nj])
                    dist = self.compute_distance(scribble_self_attn[b, o], anchor_self_attn[b, ni, nj])
                    if (dist < min_dist) and (dist < self.threshold):
                        min_dist = dist
                        min_anchor = (ni, nj)
                        
            if min_anchor != None:
                min_dist_tensor = torch.as_tensor(min_dist, dtype=neighbor_distance.dtype, device=self.device)
                if min_dist_tensor < neighbor_distance[b, min_anchor[0], min_anchor[1]]:
                    neighbor_distance[b, min_anchor[0], min_anchor[1]] = min_dist_tensor
                    neighbor_obj[b, min_anchor[0], min_anchor[1]] = o + 1
        
        # use top_k to get the closest anchor
        neighbor_distance_reshaped = neighbor_distance.reshape(self.batch_size, -1)
        _, neighbor_topk_indices = torch.topk(neighbor_distance_reshaped, top_k, dim=-1, largest=False, sorted=False)
        
        neighbor_topk_indices = neighbor_topk_indices.to(self.device)
        
        neighbor_obj_reshaped = neighbor_obj.reshape(self.batch_size, -1)
        neighbor_obj_reshaped = neighbor_obj_reshaped.type(torch.int64)
        
        # update edges with the closest anchor
        for b in range(self.batch_size):
            for k in range(top_k):
                neighbor_idx = neighbor_topk_indices[b, k]
                if neighbor_obj_reshaped[b, neighbor_idx] != 0 and neighbor_distance_reshaped[b, neighbor_idx] < self.threshold:
                    neighbor_i = neighbor_idx // self.anchor_res
                    neighbor_j = neighbor_idx % self.anchor_res
                    self.scribbles_edge[b, neighbor_obj_reshaped[b, neighbor_idx] - 1, neighbor_i, neighbor_j] = 1
                    self.visited[b, neighbor_i, neighbor_j] = neighbor_obj_reshaped[b, neighbor_idx]
                        
        for (b, o, i, j) in edge_indices:
            neighbors = self.get_neighbors((i, j), self.anchor_res, self.anchor_res)
            surrounded = True
            for (ni, nj) in neighbors:
                if self.visited[b, ni, nj] != 0:
                    surrounded = False
                    break
                
            if surrounded:
                self.scribbles_edge[b, o, i, j] = 0
        
        scribbles_edge_updated = F.interpolate(self.scribbles_edge, size=(scribble_h, scribble_w), mode='bilinear', align_corners=False).bool().float()
        
        # OR with original scribbles
        updated_scribbles = torch.logical_or(scribbles, scribbles_edge_updated).float()
        
        return updated_scribbles
    
    
    def save_scribble(self, scribble, scribble_save_dir, timestep):
        scribble_npy = scribble.detach().cpu().numpy()

        # PIL image save
        scribble_npy = scribble_npy.astype(np.uint8)
        scribble_npy = scribble_npy[:, :, np.newaxis]
        scribble_npy = np.repeat(scribble_npy, 3, axis=2)
        scribble_npy = scribble_npy * 255
        
        os.makedirs(scribble_save_dir, exist_ok=True)
        Image.fromarray(scribble_npy).save(f'{scribble_save_dir}/{timestep}.jpg')
                
        return
    
    
class SelfAttnAggregator(object):
    def __init__(
            self, 
            batch_size=1, 
            src_res=[8, 16, 32, 64], 
            tgt_res=64, 
            device='cuda'
        ):
        self.batch_size = batch_size
        
        if type(src_res) == int:
            src_res = [src_res]
            
        self.src_res = src_res
        self.tgt_res = tgt_res
        self.tgt_H = self.tgt_W = tgt_res
        self.device = device

    def aggregate_self_attn(self, all_self_attn):
        agg_self_attn = torch.zeros(self.batch_size, self.tgt_H, self.tgt_W, self.tgt_H, self.tgt_W, device=self.device)
        res_sum = 0

        for res in self.src_res:
            delta = self.tgt_H // res
            
            self_attn = torch.stack(all_self_attn[res * res], dim=0).mean(0)

            self_attn_reshaped = self_attn.reshape(self.batch_size, res * res, res, res)
            
            self_attn_resized = F.interpolate(self_attn_reshaped, size=(self.tgt_H, self.tgt_W), mode='bilinear', align_corners=False)

            self_attn_map = self_attn_resized.reshape(self.batch_size, res, res, self.tgt_H, self.tgt_W)

            for cx in range(self.tgt_H):
                for cy in range(self.tgt_W):
                    agg_self_attn[:, cx, cy, :, :] += self_attn_map[:, cx // delta, cy // delta, :, :] * (res * res)

            res_sum += res * res

        agg_self_attn.div_(res_sum)
        return agg_self_attn
