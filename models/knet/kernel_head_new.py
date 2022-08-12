import torch
import torch.nn as nn
import torch.nn.functional as F


from .semantic_fpn_wrapper_new import SemanticFPNWrapper


class ConvKernelHead(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 lang_channels=768,
                 feat_downsample_stride=2,
                 hard_mask_thr=0.5,
                 conv_kernel_size=1
                 ):
        super(ConvKernelHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.localization_fpn = SemanticFPNWrapper()
        self.feat_downsample_stride = feat_downsample_stride
        self.hard_mask_thr = hard_mask_thr
        self.conv_kernel_size = conv_kernel_size

        self.init_kernels = nn.Linear(lang_channels, in_channels * (conv_kernel_size ** 2))

        self.loc_convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(32, 256, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def _decode_init_proposals(self, img, lang_feat):
        loc_feats = self.localization_fpn(img) # ./semantic_fpn_wrapper.py, row 190
        # loc_feats: [B, 256, H//8, W//8]
        loc_feats = self.loc_convs(loc_feats) #[2,256,100,152]
        # loc_feats: [B, 256, H//8, W//8]

        proposal_feats = self.init_kernels(lang_feat) # [B, 230, 256]
        B, N, C = proposal_feats.shape
        # proposal_feats = proposal_feats.reshape(B, N, C, self.conv_kernel_size, self.conv_kernel_size)
        # mask_preds = []
        # for i in range(B):
        #     mask_preds.append(
        #         F.conv2d(loc_feats[i:i+1], proposal_feats[i], padding=int(self.conv_kernel_size // 2))
        #     )
        # mask_preds = torch.cat(mask_preds, dim=0)
        mask_preds = torch.einsum('bchw,bnc->bnhw', loc_feats, proposal_feats)
        # mask_preds: [B, 100, H//8, W//8]

        # sigmoid_masks = mask_preds.sigmoid()
        # sigmoid_masks: [B, 100, H//8, W//8]
        # nonzero_inds = sigmoid_masks > self.hard_mask_thr
        # sigmoid_masks = nonzero_inds.float() * sigmoid_masks
        # obj_feats = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x_feats)
        # obj_feats = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, loc_feats)
        # obj_feats: [B, 100, 256]

        # proposal_feats = proposal_feats + obj_feats.reshape(B, N, C, self.conv_kernel_size, self.conv_kernel_size)
        # proposal_feats = proposal_feats + obj_feats.reshape(B, N, C)
        # proposal_feats: [B, 230, 256, 1, 1]
        
        return proposal_feats, loc_feats, mask_preds


    def forward(self, img, lang_feat):
        """Forward function in training stage."""
        #img:
        # p2: [B, 256, H//4, W//4]
        # p3: [B, 256, H//8, W//8]
        # p4: [B, 256, H//16, W//16]
        # p5: [B, 256, H//32, W//32]
        results = self._decode_init_proposals(img, lang_feat) # 200 row
        (proposal_feats, x_feats, mask_preds) = results
        # proposal_feats: [2, 230, 256]
        # x_feats: [2, 256, H//8, W//8] (semantic_feats + loc_feats)
        # mask_preds: [2, 100, H//8, W//8] (non binary)
        if self.feat_downsample_stride > 1:
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=self.feat_downsample_stride,
                mode='bilinear',
                align_corners=False)
            # scaled_mask_preds: [B, 100, H//4, W//4]
        else:
            scaled_mask_preds = mask_preds

        return proposal_feats, x_feats, scaled_mask_preds

