from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Knet(nn.Module):

    def __init__(self):
        super(KNet, self).__init__(self)

    def forward_train(x):
        rpn_results = self.rpn_head.forward_train(x, img_metas, gt_masks, 
                                                  gt_labels, gt_sem_seg,
                                                  gt_sem_cls) # ./kernel_head.py, 277 row
        (rpn_losses, proposal_feats, x_feats, mask_preds,
         cls_scores) = rpn_results
        # rpn_losses.keys(): dict_keys(['loss_rpn_mask', 'loss_rpn_dice', 'loss_rpn_rank', 'loss_rpn_seg'])
        # proposal_feats: [2, 153, 256, 1, 1]
        # x_feats: [2, 256, 136, 100]
        # mask_preds: [2, 153, 136, 100]
        # cls_scores: [None]

        import pdb; pdb.set_trace()
        losses = self.roi_head.forward_train(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            gt_masks,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_bboxes=gt_bboxes,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            imgs_whwh=None)

        losses.update(rpn_losses)
        import pdb; pdb.set_trace()
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results
        segm_results = self.roi_head.simple_test(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            imgs_whwh=None,
            rescale=rescale)
        return segm_results

    def forward_dummy(self, img):
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        rpn_results = self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x_feats, proposal_feats,
                                               dummy_img_metas)
        return roi_outs
