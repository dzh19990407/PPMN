from detectron2.data.build import _train_loader_from_config
from detectron2.config import get_cfg
from detectron2.data.common import DatasetFromList, MapDataset
from torch.utils.data import DataLoader
from detectron2.modeling import build_model
from detectron2.data.samplers import InferenceSampler
import torch

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

if __name__ == '__main__':
    cfg_path = '../detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_train.yaml'
    cfg = setup(cfg_path)
    data_dict = _train_loader_from_config(cfg, cfg.DATASETS.TRAIN[0])
    dataset = data_dict['dataset']
    mapper = data_dict['mapper']
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    import pdb; pdb.set_trace()
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
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