import torch.nn.utils.prune as prune

class myFavoritePruningMethod(prune.BasePruningMethod):
    
    PRUNING_TYPE = 'unstructured'

    def __init__(self, Mask):
        self.mask = Mask
        

    def compute_mask(self, t, default_mask):
        return self.mask