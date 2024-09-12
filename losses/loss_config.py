
class ScribbleLossConfig(object):
    def __init__(self, loss_config_file=None):
        self.loss_type = ['all']
        
        # self attention loss related
        self.self_attn_loss_res = [16]
        self.use_outside_mask_self_loss = True # if not, use scribbles
        
        # cross attention loss related
        self.cross_attn_loss_res = [16]
        self.cross_focal_loss_alpha = 0.25
        self.cross_focal_loss_beta = 2.0
        self.cross_focal_loss_weight = 3.0
        self.use_outside_mask_loss = True
        self.outside_not_largest = True  # suppress-to-impress
        self.use_outside_scribble_loss = True
        self.use_outside_scribble_loss_with_mask = True
        self.use_inside_scribble_loss = True
        self.use_focal_loss = True
        self.cross_top_k = 5
        
        # moment loss related
        self.moment_loss_alpha = 2.
        
        self.smooth_attn = True
        self.sigma = 0.5
        self.kernel_size=3
        
        self.top_k =5
        self.min_top_k_obj_num = True
        
        self.loss_type = ['all']
        
        if loss_config_file is not None:
            for key, value in loss_config_file.items():
                assert hasattr(self, key), f"Invalid key {key} in loss_config_file."
                setattr(self, key, value)