class ScribbleLossScheduler(object):
    def __init__(self, loss_scheduler_file=None):
        self.loss_threshold = 1.0
        self.decay_factor = 0.95
        
        # modulation
        self.mod_self_reg = 0.3
        self.mod_cross_reg = 1.0
        self.mod_res = [16]
        self.mod_pos = ['out']
        self.mod_with_masks = False
        
        # schedule step interval
        self.loss_end = 35

        self.moment_loss_start = 0
        self.moment_loss_end = 15

        self.focal_loss_start = 0
        self.focal_loss_end = 20

        self.cross_loss_start = 15
        self.cross_loss_end = 35

        self.self_loss_start = 0
        self.self_loss_end = 35

        self.propagation_start = 5
        self.propagation_end = 15

        self.modulation_start = 0
        self.modulation_end = 35
        
        self.feature_loss_start = 15
        self.feature_loss_end = 30

        self.intervals = [[0, 9], [10, 14], [15, 20]]
  
        self.interval_params = [
            {'step_size': 3, 'loss_self_scale': 5, 'loss_cross_scale': 5, 'loss_moment_scale': 3, 'loss_feature_scale': 1, 'max_iter': 3},
            {'step_size': 2, 'loss_self_scale': 3, 'loss_cross_scale': 3, 'loss_moment_scale': 2, 'loss_feature_scale': 1, 'max_iter': 2},
            {'step_size': 1, 'loss_self_scale': 1, 'loss_cross_scale': 1, 'loss_moment_scale': 1, 'loss_feature_scale': 1, 'max_iter': 1}
        ]

        if loss_scheduler_file is not None:
            for key, value in loss_scheduler_file.items():
                assert hasattr(self, key), f"Invalid key {key} in loss_scheduler_file."
                setattr(self, key, value)

    def schedule(self, step_num):
        result = {}

        for i, interval in enumerate(self.intervals):
            start = interval[0]
            end = interval[1]
            
            if start <= step_num <= end:
                params = self.interval_params[i]
                break
        else:
            params = self.interval_params[-1]

        decay_multiplier = self.decay_factor ** (step_num - start)
        for key in params:
            result[key] = max(0, params[key] * decay_multiplier)

        return result

