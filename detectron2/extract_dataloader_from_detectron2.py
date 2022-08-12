from detectron2.data.build import (_train_loader_from_config, _test_loader_from_config)
from detectron2.config import get_cfg
from detectron2.data.common import DatasetFromList, MapDataset
from torch.utils.data import DataLoader
from detectron2.modeling import build_model
from detectron2.data.samplers import InferenceSampler
import torch

import os

# import sys
# sys.path.append('/mnt/data1/dzh/PNG-main/detectron2')

def setup(cfg_path):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def fpn_data():
    # cfg_path = './configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_val.yaml'
    # cfg = setup(cfg_path)
    cfg_path = '../detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_train.yaml'
    cfg = setup(cfg_path)
    # import pdb; pdb.set_trace()
    print(os.getcwd())
    os.chdir('/mnt/data1/dzh/PNG-main/detectron2')
    data_dict = _test_loader_from_config(cfg, cfg.DATASETS.TRAIN[0])
    # return data_dict
    dataset = data_dict['dataset']
    mapper = data_dict['mapper']
    # import pdb; pdb.set_trace()
    return dataset, mapper
    import pdb; pdb.set_trace()
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    import pdb; pdb.set_trace()
    # return dataset
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    return dataset, batch_sampler
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    model = build_model(cfg)
    model.eval()
    for idx, inputs in enumerate(data_loader):
        outputs = model.inference(inputs)
        import pdb; pdb.set_trace()