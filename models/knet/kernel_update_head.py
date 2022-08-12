import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiheadattention import (MultiheadAtten, Ffn)
from .kernel_updator import KernelUpdator


class KernelUpdateHead(nn.Module):

    def __init__(self,
                 num_classes=80,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_mask_fcs=3,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 mask_thr=0.5,
                 conv_kernel_size=1,
                 hard_mask_thr=0.5,
                 kernel_init=False,
                 with_ffn=True,
                 mask_out_stride=4,
                 relative_coors=False,
                 relative_coors_off=False,
                 feat_gather_stride=1,
                 mask_transform_stride=2,
                 num_points=100
                 ):
        super(KernelUpdateHead, self).__init__()
        self.num_classes = num_classes


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout

        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.mask_out_stride = mask_out_stride
        self.relative_coors = relative_coors
        self.relative_coors_off = relative_coors_off
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride
        self.num_points = num_points

        self.loc_convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(32, 256, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )

        self.topk_attn = MultiheadAtten(in_channels, num_heads, dropout)
        self.topk_norm = nn.LayerNorm(in_channels*(conv_kernel_size**2), \
                eps=1e-05, elementwise_affine=True)


        self.attention = MultiheadAtten(in_channels * (conv_kernel_size**2),
                                            num_heads, dropout)
        self.attention_norm = nn.LayerNorm(in_channels*(conv_kernel_size**2), eps=1e-05, elementwise_affine=True)
        self.kernel_update_conv = KernelUpdator(in_channels=256,
                                                feat_channels=256,
                                                out_channels=256,
                                                input_feat_shape=3,
                                                gate_sigmoid=True,
                                                gate_norm_act=False,
                                                activate_out=False,
                                                act_cfg=dict(type='ReLU', inplace=True),
                                                norm_cfg=dict(type='LN'))

  

        if self.with_ffn:
            self.ffn = Ffn()
            self.ffn_norm = nn.LayerNorm(in_channels, eps=1e-05, elementwise_affine=True)
            self.ffn_pre = Ffn()
            self.ffn_norm_pre = nn.LayerNorm(in_channels, eps=1e-05, elementwise_affine=True)
           

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True))
            self.cls_fcs.append(nn.ReLU(inplace=True))

        self.fc_cls = nn.Linear(in_channels, self.num_classes)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True))
            self.mask_fcs.append(nn.ReLU(inplace=True))

        self.fc_mask = nn.Linear(in_channels, out_channels)
        


    def forward(self, x, proposal_feat, mask_preds, mask_shape=None):
        K = self.num_points
        x = self.loc_convs(x)
        # proposal_feat: [B, 230, 256]
        B, N = proposal_feat.shape[:2]
        # x: [B, 256, H//8, W//8] <--> Features $F$
        C, H, W = x.shape[-3:]
        # mask_preds: [B, 230, H//4, W//4] <--> $M$
        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
            # gather_mask: [B, 230, H//8, W//8]
        else:
            gather_mask = mask_preds

        _, topk_inds = torch.topk(gather_mask.flatten(-2), K)
        # [B, N, K]
        v_feat = x.flatten(-2).transpose(1, 2)
        # [B, HW, C]
        topk_feats = []
        for i in range(B):
            topk_inds_tmp = topk_inds[i]
            # [N, K]
            v_feat_tmp = v_feat[i]
            # [HW, C]
            topk_feats.append(v_feat_tmp[topk_inds_tmp])
        topk_feats = torch.stack(topk_feats)
        # [B, N, K, C]
        obj_feat = proposal_feat.unsqueeze(2)
        # [B, N, 1, C]
        topk_feats = topk_feats.reshape(B*N, K, C)
        obj_feat = obj_feat.reshape(B*N, 1, C)
        topk_feats = topk_feats.transpose(0, 1)
        # [B*N, K, C]
        obj_feat = obj_feat.transpose(0, 1)
        # [B*N, 1, C]
            
        # [B, N, K]
        obj_feat = self.topk_attn(obj_feat, topk_feats)
        obj_feat = self.topk_norm(obj_feat)
        
        obj_feat = obj_feat.transpose(0, 1)
        obj_feat = obj_feat.reshape(B, N, 1, C).squeeze(2)
        obj_feat = self.ffn_norm_pre(self.ffn_pre(obj_feat))
        # [B, N, C]

        mask_feat = obj_feat

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        mask_feat = self.fc_mask(mask_feat)
        # [B, N, C, K*K] -> [B*N, C, K, K]

    
        mask_x = x
        # new_mask_preds: [B, C, H//8, W//8]
        new_mask_preds = torch.einsum('bchw,bnc->bnhw', mask_x, mask_feat)

        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)


        return new_mask_preds, obj_feat
    