import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvKernelHead(nn.Module):
    def __init__(self):
        super(ConvKernelHead, self).__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(32, 256, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )
        self.init_kernels = nn.Conv2d(
            self.out_channels,
            self.num_proposals,
            self.conv_kernel_size,
            padding=int(self.conv_kernel_size // 2),
            bias=False)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(32, 256, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(256, 133, kernel_size=(1, 1), stride=(1, 1))

    def _decode_init_proposals(self, localization_feats):
        # num_imgs = len(img_metas) # <==> batch_size

        if isinstance(localization_feats, list):
            loc_feats = localization_feats[0]
        else:
            loc_feats = localization_feats
        # loc_feats: [B, 256, H//8, W//8]
        loc_feats = self.loc_conv(loc_feats) #[2,256,100,152]
        mask_preds = self.init_kernels(loc_feats)
        # mask_preds: [B, 100, H//8, W//8]

        semantic_feats = localization_feats[1]
        for conv in self.seg_conv:
            semantic_feats = conv(semantic_feats)

        if semantic_feats is not None:
            seg_preds = self.conv_seg(semantic_feats) #[2,133,100,152]
        else:
            seg_preds = None
        # seg_preds: [B, 133, H//8, W//8]

        proposal_feats = self.init_kernels.weight.clone()
        # proposal_feats: [100, 256, 1, 1]
        proposal_feats = proposal_feats[None].expand(1,
                                                     *proposal_feats.size())
        # proposal_feats: [B, 100, 256, 1, 1]

        if semantic_feats is not None:
            x_feats = semantic_feats + loc_feats
        else:
            x_feats = loc_feats
        # x_feats: [B, 256, H//8, W//8]

        sigmoid_masks = mask_preds.sigmoid()
        # sigmoid_masks: [B, 100, H//8, W//8]
        nonzero_inds = sigmoid_masks > 0.5
        sigmoid_masks = nonzero_inds.float() #[2,100,100,152]
        obj_feats = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x_feats)
        # obj_feats: [B, 100, 256]

        cls_scores = None

        if self.proposal_feats_with_obj:
            proposal_feats = proposal_feats + obj_feats.view(
                1, 100, 256, 1, 1)
            # proposal_feats: [B, 100, 256, 1, 1]

        # if self.cat_stuff_mask and not self.training:
        #     mask_preds = torch.cat(
        #         [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
        #     stuff_kernels = self.conv_seg.weight[self.
        #                                          num_thing_classes:].clone()
        #     stuff_kernels = stuff_kernels[None].expand(num_imgs,
        #                                                *stuff_kernels.size())
        #     proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)

        return proposal_feats, x_feats, mask_preds, cls_scores, seg_preds
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_sem_seg=None,
                      gt_sem_cls=None):
        """Forward function in training stage."""
        # p2: [B, 256, H//4, W//4]
        # p3: [B, 256, H//8, W//8]
        # p4: [B, 256, H//16, W//16]
        # p5: [B, 256, H//32, W//32]
        num_imgs = 1 # <==> batch_size
        results = self._decode_init_proposals(img, img_metas) # 200 row
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = results
        # proposal_feats: [2, 100, 256, 1, 1] (?)
        # x_feats: [2, 256, H//8, W//8] (semantic_feats + loc_feats)
        # mask_preds: [2, 100, H//8, W//8] (non binary)
        # cls_scores: None
        # seg_preds: [2, 133, H//8, W//8] (conv(semantic_feats))
        if self.feat_downsample_stride > 1:
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=self.feat_downsample_stride,
                mode='bilinear',
                align_corners=False)
            # scaled_mask_preds: [B, 100, H//4, W//4]
            if seg_preds is not None:
                scaled_seg_preds = F.interpolate(
                    seg_preds,
                    scale_factor=self.feat_downsample_stride,
                    mode='bilinear',
                    align_corners=False)
            # scaled_seg_preds: [B, 133, H//4, W//4]
        else:
            scaled_mask_preds = mask_preds
            scaled_seg_preds = seg_preds

        if self.hard_target:
            gt_masks = [x.bool().float() for x in gt_masks]
        else:
            gt_masks = gt_masks #len(gt_mask) = B, gt_mask[0]=[num_instance,200,304]
        # gt_masks: float, not 0/1, is 0~2

        sampling_results = []
        if cls_scores is None:
            detached_cls_scores = [None] * 2
        else:
            detached_cls_scores = cls_scores.detach()
        # detached_cls_scores: [None, None]

        for i in range(num_imgs):
            assign_result = self.assigner.assign(scaled_mask_preds[i].detach(),
                                                 detached_cls_scores[i],
                                                 gt_masks[i], gt_labels[i],
                                                 img_metas[i])
            sampling_result = self.sampler.sample(assign_result,
                                                  scaled_mask_preds[i],
                                                  gt_masks[i])
            sampling_results.append(sampling_result)

        import pdb; pdb.set_trace()
        mask_targets = self.get_targets(
            sampling_results,
            gt_masks,
            self.train_cfg,
            True,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls)

        losses = self.loss(scaled_mask_preds, cls_scores, scaled_seg_preds,
                           proposal_feats, *mask_targets)
        import pdb; pdb.set_trace()
        
        if self.cat_stuff_mask and self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)

        return losses, proposal_feats, x_feats, mask_preds, cls_scores