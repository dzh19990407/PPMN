import torch.nn as nn
import torch.nn.functional as F
from .kernel_update_head import KernelUpdateHead


class KernelIterHead(nn.Module):

    def __init__(self, 
        num_stages=3,
        num_points=100,
    ):
        super(KernelIterHead, self).__init__()
        self.num_stages = num_stages
        self.mask_head = nn.ModuleList()
        for i in range(num_stages):
            self.mask_head.append(KernelUpdateHead(num_points=num_points))


    def _mask_forward(self, stage, x, kernels, mask_preds):
        mask_head = self.mask_head[stage]
        mask_preds, kernels = mask_head(
            x, kernels, mask_preds)

        return mask_preds, kernels

    
    def forward_train(self, x, proposal_feats, mask_preds):
        # object_feats = proposal_feats
        kernels = proposal_feats
        all_stage_mask_results = []
        for stage in range(self.num_stages):
            mask_preds, kernels = self._mask_forward(stage, x, kernels,
                                              mask_preds)
            all_stage_mask_results.append(mask_preds)

        return all_stage_mask_results

    