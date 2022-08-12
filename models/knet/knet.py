import torch.nn as nn
import torch.nn.functional as F
from .kernel_head_new import ConvKernelHead
from .kernel_iter_head import KernelIterHead
import torch


class KNet(nn.Module):

    def __init__(
        self,
        num_stages=3,
        num_points=100
    ):
        super(KNet, self).__init__()

        self.rpn_head = ConvKernelHead()
        self.roi_head = KernelIterHead(
            num_stages=num_stages,
            num_points=num_points
        )
        

    def forward(self, x, lang_feat, gts=None, train=True):
        rpn_results = self.rpn_head(x, lang_feat)
        (proposal_feats, x_feats, mask_preds) = rpn_results
        # proposal_feats: [B, 230, 256]
        # x_feats: [B, 256, H//8, W//8]
        # mask_preds: [B, 230, H//4, W//4]
        masks = [mask_preds]
        masks_iter = self.roi_head.forward_train(x_feats, proposal_feats, mask_preds)
        masks = masks + masks_iter
        return masks
