# Standard lib imports
import time
import numpy as np
import os.path as osp
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from detectron2.structures import ImageList

# PyTorch imports
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

# Local imports
from utils.meters import average_accuracy
from utils import AverageMeter
from utils import compute_mask_IoU
from data import PanopticNarrativeGroundingDataset, PanopticNarrativeGroundingValDataset
from models.knet.knet import KNet
from models.knet.dice_loss import DiceLoss
from models.knet.cross_entropy_loss import CrossEntropyLoss
from models.encoder_bert import BertEncoder
from utils.logger import setup_logger
from utils.collate_fn import default_collate
from utils.distributed import (all_gather, all_reduce)
from models.extract_fpn_with_ckpt_load_from_detectron2 import fpn
from utils.contrastive import CKDLoss



def train_epoch(train_loader, bert_encoder, fpn_model, model, 
    optimizer, epoch, cfg, logger, writer):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): train loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        loss_functions (loss): the loss function to optimize.
        epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            PNG/config/defaults.py
    """
    if dist.get_rank()==0:
        logger.info('-' * 89)
        logger.info('Training epoch {:5d}'.format(epoch))
        logger.info('-' * 89)

    # Enable train mode.
    model.train()
    if cfg.bert_freeze:
        bert_encoder.eval()
    else:
        bert_encoder.train()
    if cfg.fpn_freeze:
        fpn_model.eval()
    else:
        fpn_model.train()

    epoch_loss = AverageMeter()
    time_stats = AverageMeter()

    # Use cuda if available
    dice_loss = DiceLoss()
    ce_loss = CrossEntropyLoss(use_sigmoid=True)
    # closs, c2loss = [], []
    # for i in range(cfg.num_stages):
    #     closs.append(CKDLoss())
        # c2loss.append(CKDLoss())

    for (batch_idx, (caption, grounding_instances, ann_categories, \
        ann_types, noun_vector_padding, ret_noun_vector, fpn_input_data)) in enumerate(train_loader):
        ret_noun_vector = ret_noun_vector.to(cfg.local_rank)
        ann_types = ann_types.to(cfg.local_rank)
        ann_categories = ann_categories.to(cfg.local_rank)
        
        start_time = time.time()
        
        with torch.no_grad():
            lang_feat, _ = bert_encoder(caption) #bert for caption
            lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
                cfg.max_seg_num, lang_feat.shape[-1]))

            for i in range(len(lang_feat)):
                cur_lang_feat = lang_feat[i][noun_vector_padding[i].nonzero().flatten()]
                lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat
            
        fpn_feature = fpn_model(fpn_input_data) #fpn for imgs

        # preprocessing for gt masks
        with torch.no_grad():
            gts = [F.interpolate(grounding_instances[i]["gt"].to(cfg.local_rank).unsqueeze(0), \
                                (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                mode='bilinear').squeeze() for i in range(len(grounding_instances))]
            gts = ImageList.from_tensors(gts, 32).tensor
            gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
            gts = (gts > 0).float()
        
        # gts: [B, max_seg_num, H//4, W//4]
        # lang_feat_valid: [B, max_seg_num, C]
        # predictions, kernels, gt_ins_feats = model(fpn_feature, lang_feat_valid, gts=gts) #Knet
        predictions = model(fpn_feature, lang_feat_valid, train=False) #Knet

        loss = 0
        contrastive_loss = 0
        grad_sample = ann_types != 0
        gt = gts[grad_sample]

        for i in range(len(predictions)):
            pred = predictions[i][grad_sample]
            loss = loss + ce_loss(pred, gt) + dice_loss(pred, gt)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()

        # Gather all the predictions across all the devices.
        if cfg.num_gpus > 1:
            loss = all_reduce([loss])[0]
        
        time_stats.update(time.time() - start_time, 1)
        epoch_loss.update(loss, 1)

        if dist.get_rank()==0:
            if (batch_idx % cfg.log_period == 0):
                elapsed_time = time_stats.avg
                logger.info(' [{:5d}] ({:5d}/{:5d}) | ms/batch {:.4f} |'
                    ' avg loss {:.6f} |'
                    ' lr {:.7f} |'.format(
                        epoch, batch_idx, len(train_loader),
                        elapsed_time * 1000, 
                        epoch_loss.avg,
                        optimizer.param_groups[0]["lr"]))
                writer.add_scalar('train/loss', epoch_loss.avg, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch * len(train_loader) + batch_idx)
                writer.flush()


    return epoch_loss.avg

def upsample_eval(tensors, pad_value=0, t_size=[400, 400]):
    batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(t_size)
    batched_imgs = tensors[0].new_full(batch_shape, pad_value)
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


@torch.no_grad()
def evaluate(val_loader, bert_encoder, fpn_model, model, epoch, cfg, logger, writer):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        cfg (CfgNode): configs. Details can be found in
            PNG/config/defaults.py
    """
    if dist.get_rank()==0:
        logger.info('-' * 89)
        logger.info('Evaluation on val set epoch {:5d}'.format(epoch))
        logger.info('-' * 89)
    
    # Enable eval mode.
    model.eval()
    bert_encoder.eval()
    fpn_model.eval()
    
    instances_iou = []
    singulars_iou = []
    plurals_iou = []
    things_iou = []
    stuff_iou = []
    # pbar = tqdm(total=len(val_loader))
    for (batch_idx, (caption, grounding_instances, ann_categories, \
        ann_types, noun_vector_padding, ret_noun_vector, fpn_input_data)) in enumerate(val_loader):
        ann_categories = ann_categories.to(cfg.local_rank)
        ann_types = ann_types.to(cfg.local_rank)
        # ret_noun_vector = ret_noun_vector.to(cfg.local_rank)
        
        # Perform the forward pass
        with torch.no_grad():
            lang_feat, _ = bert_encoder(caption) #bert for caption
            lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
                cfg.max_seg_num, lang_feat.shape[-1]))

            for i in range(len(lang_feat)):
                cur_lang_feat = lang_feat[i][noun_vector_padding[i].nonzero().flatten()]
                lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat


        fpn_feature = fpn_model(fpn_input_data)
        predictions = model(fpn_feature, lang_feat_valid, train=False)
        predictions = predictions[-1]
        predictions = predictions.sigmoid() #[2,230,272,304]

        predictions_valid = predictions.new_zeros((predictions.shape[0], cfg.max_phrase_num, \
            predictions.shape[-2], predictions.shape[-1]))
        for i in range(len(predictions)):
            cur_phrase_interval = ret_noun_vector[i]['inter']
            for j in range(len(cur_phrase_interval)-1):
                for k in range(cur_phrase_interval[j], cur_phrase_interval[j+1]):
                    predictions_valid[i, j, :] = predictions_valid[i, j, :] + predictions[i][k]
                predictions_valid[i, j, :] = predictions_valid[i, j, :] / (cur_phrase_interval[j+1]-cur_phrase_interval[j])
                
        predictions = (predictions_valid > 0.5).float()

        predictions = upsample_eval(predictions)

        # preprocessing for gt masks
        with torch.no_grad():
            gts = [F.interpolate(grounding_instances[i]["gt"].to(cfg.local_rank).unsqueeze(0), \
                                (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                mode='bilinear').squeeze() for i in range(len(grounding_instances))]
            gts = ImageList.from_tensors(gts, 32).tensor
            gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
            gts = (gts > 0).float()
            gts = upsample_eval(gts)
        
        # Gather all the predictions across all the devices.
        if cfg.num_gpus > 1:
            predictions, gts, ann_categories, ann_types = all_gather(
                [predictions, gts, ann_categories, ann_types]
            )

        # Evaluation
        


        for p, t, th, s in zip(predictions, gts, ann_categories, ann_types):
            for i in range(cfg.max_phrase_num):
                if s[i] == 0:
                    continue
                else:
                    pd = p[i]
                    _, _, instance_iou = compute_mask_IoU(pd, t[i])
                    instances_iou.append(instance_iou.cpu().item())
                    if s[i] == 1:
                        singulars_iou.append(instance_iou.cpu().item())
                    else:
                        plurals_iou.append(instance_iou.cpu().item())
                    if th[i] == 1:
                        things_iou.append(instance_iou.cpu().item())
                    else:
                        stuff_iou.append(instance_iou.cpu().item())
        
        if batch_idx % 100 == 0:
            print(f'{batch_idx}/{len(val_loader)}')
        # if dist.get_rank()==0:
        #     pbar.update(1)
            # if batch_idx % cfg.log_period == 0:
            # tqdm.write('acc@0.5: {:.5f} | AA: {:.5f}'.format(accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5), average_accuracy(instances_iou))) 
    
    # pbar.close()
    # Final evaluation metrics
    AA = average_accuracy(instances_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='overall')
    AA_singulars = average_accuracy(singulars_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='singulars')
    AA_plurals = average_accuracy(plurals_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='plurals')
    AA_things = average_accuracy(things_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='things')
    AA_stuff = average_accuracy(stuff_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='stuff')
    accuracy = accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5)
    if dist.get_rank()==0:
        logger.info('| final acc@0.5: {:.5f} | final AA: {:.5f} |  AA singulars: {:.5f} | AA plurals: {:.5f} | AA things: {:.5f} | AA stuff: {:.5f} |'.format(
                                               accuracy,
                                               AA,
                                               AA_singulars,
                                               AA_plurals,
                                               AA_things,
                                               AA_stuff))
        writer.add_scalar('aa/acc@0.5', accuracy, epoch)
        writer.add_scalar('aa/final', AA, epoch)
        writer.add_scalar('aa/singulars', AA_singulars, epoch)
        writer.add_scalar('aa/plurals', AA_plurals, epoch)
        writer.add_scalar('aa/things', AA_things, epoch)
        writer.add_scalar('aa/stuffs', AA_stuff, epoch)
        
    return AA



def train(cfg):
    local_rank = cfg.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=cfg.backend)

    if dist.get_rank() == 0:
        logger = setup_logger(cfg.output_dir, dist.get_rank())
        writer = SummaryWriter(osp.join(cfg.output_dir, 'tensorboard'))
    else:
        logger, writer = None, None

    # Set random seed from configs.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if dist.get_rank() == 0:
        logger.info(cfg)

    bert_encoder = BertEncoder(cfg).to(local_rank)
    bert_encoder = DDP(bert_encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) 
    fpn_model = fpn(cfg.detectron2_ckpt, cfg.detectron2_cfg)
    fpn_model = fpn_model.to(local_rank)
    fpn_model = DDP(fpn_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model = KNet(
        num_stages=cfg.num_stages,
        num_points=cfg.num_points,
    ).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if cfg.bert_freeze:
        cnt = 0
        for n, c in bert_encoder.named_parameters():
            c.requires_grad = False
            cnt += 1
        if dist.get_rank() == 0:
            logger.info(f'Freezing {cnt} parameters of BERT.')

    if cfg.fpn_freeze:
        cnt = 0
        for n, c in fpn_model.named_parameters():
            c.requires_grad = False
            cnt += 1
        if dist.get_rank() == 0:
            logger.info(f'Freezing {cnt} parameters of FPN.')

    if not cfg.test_only:
        train_dataset = PanopticNarrativeGroundingDataset(cfg, 'train2017')
        distributed_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler = distributed_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=default_collate,
        )

    val_dataset = PanopticNarrativeGroundingValDataset(cfg, 'val2017', False)
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler = distributed_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=default_collate,
    )

    if cfg.bert_freeze and cfg.fpn_freeze:
        # train_params += list(filter(lambda p: p.requires_grad, bert_encoder.parameters()))
        # train_params += list(filter(lambda p: p.requires_grad, fpn_model.parameters()))
        train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if dist.get_rank() == 0:
            logger.info(f'{len(train_params)} training params.')
        optimizer = optim.Adam(train_params,
                            lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.fpn_freeze:
        bert_encoder_params = list(filter(lambda p: p.requires_grad, bert_encoder.parameters()))
        model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.Adam([{'params': model_params, 'lr':cfg.base_lr},
                               {'params': bert_encoder_params, 'lr':cfg.base_lr/10}])
    else:
        raise RuntimeError('Not Implement!!!!')
    
    if cfg.scheduler == 'step':
        if cfg.fpn_freeze and not cfg.bert_freeze:
            milestones = [10, 12, 14]
            lambda1 = lambda epoch: 1 if epoch < milestones[0] else 0.5 if epoch < milestones[1] else 0.25 if epoch < milestones[2] else 0.125
            lambda2 = lambda epoch: 1
            lambda_list = [lambda1, lambda2]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_list)
        else:
            milestones = [10, 12, 14]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, \
                                                      gamma=0.5)
    elif cfg.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, \
                                                               mode='max', min_lr=1e-6, \
                                                               patience=2)
    else:
        raise ValueError(f'{cfg.scheduler} NOT IMPLEMENT!!!')


    start_epoch, best_val_score = 0, None
    if osp.exists(cfg.ckpt_path):
        if dist.get_rank()==0:
            print('Loading model from: {0}'.format(cfg.ckpt_path))
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state'])
        fpn_model.load_state_dict(checkpoint['fpn_model_state'])
        bert_encoder.load_state_dict(checkpoint['bert_model_state'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_val_score = checkpoint['best_val_score']

        if cfg.test_only:
            epoch = 0
            evaluate(val_loader, bert_encoder, \
            fpn_model, model, epoch, cfg, logger, writer)
            return

    if dist.get_rank()==0:
        logger.info('Train begins...')

    # Perform the training loop
    for epoch in range(start_epoch, cfg.epoch):
        epoch_start_time = time.time()
        # Shuffle the dataset
        train_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        train_loss = train_epoch(train_loader, bert_encoder, \
            fpn_model, model, optimizer, epoch, cfg, logger, writer)
        accuracy = evaluate(val_loader, bert_encoder, \
            fpn_model, model, epoch, cfg, logger, writer)
        if dist.get_rank() == 0:
            writer.flush()

        if cfg.scheduler == 'step':
            scheduler.step()
        elif cfg.scheduler == 'reduce':
            scheduler.step(accuracy)
        else:
            raise ValueError(f'{cfg.scheduler} NOT IMPLEMENT!!!')

        if dist.get_rank()==0:
            # Save best model in the validation set
            if best_val_score is None or accuracy > best_val_score:
                best_val_score = accuracy
                model_final_path = osp.join(cfg.output_dir, 'model_best.pth')
                model_final = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "fpn_model_state": fpn_model.state_dict(),
                    "bert_model_state": bert_encoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_score": accuracy
                }
                torch.save(model_final, model_final_path)
            if epoch > cfg.save_ckpt:
                model_final_path = osp.join(cfg.output_dir, f'checkpoint_{epoch}.pth')
            else:
                model_final_path = osp.join(cfg.output_dir, f'checkpoint.pth')

            model_final = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "fpn_model_state": fpn_model.state_dict(),
                "bert_model_state": bert_encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_score": accuracy
            }
            torch.save(model_final, model_final_path)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s '
                    '| epoch loss {:.6f} |'.format(
                        epoch, time.time() - epoch_start_time, train_loss))
            logger.info('-' * 89)
    if dist.get_rank() == 0:
        writer.close()
