import sys
import json
import torch
import argparse
import os.path as osp
from tqdm import tqdm
from skimage import io

sys.path.append("..")
from baseline.models.tokenization import BertTokenizer

parser = argparse.ArgumentParser()

# Data related settings
parser.add_argument(
    "--data_dir",
    default="/mnt/data1/dzh/PNG-main/panoptic_narrative_grounding",
    help="Path to data directory",
)

args = parser.parse_args()
args_dict = vars(args)
print("Argument list to program")
print("\n".join(["--{0} {1}".format(arg, args_dict[arg]) for arg in args_dict]))
print("\n\n")

splits = ["train2017", "val2017"]
PATH_TO_DATA_DIR = args.data_dir
PATH_TO_FEATURES_DIR = osp.join(PATH_TO_DATA_DIR, "features")
ann_dir = osp.join(PATH_TO_DATA_DIR, "annotations")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


def compute_mask_IoU(masks, target):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union, intersection / union


cont = 0
max_len = 0
for split in tqdm(splits):
    tqdm.write("LOADING {} ANNOTATIONS".format(split.upper()))
    panoptic = load_json(osp.join(ann_dir, "panoptic_{:s}.json".format(split)))
    # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    images = panoptic["images"]
    # 118287 for training
    images = {i["id"]: i for i in images}
    # image id (int) -> image item
    panoptic_anns = panoptic["annotations"]
    panoptic_anns = {int(a["image_id"]): a for a in panoptic_anns}
    # image id (int) -> ann item

    panoptic_pred_path = osp.join(
        PATH_TO_FEATURES_DIR, split, "inference", "panoptic_seg_predictions"
    )

    tqdm.write("LOADING {} DATA".format(split.upper()))
    panoptic_narratives = load_json(
        osp.join(PATH_TO_DATA_DIR, "annotations", "png_coco_{:s}.json".format(split))
    )

    length = len(panoptic_narratives)
    # 134272 for training
    iterable = tqdm(range(0, length))
    all_dict = []
    tqdm.write("FOMATING {} DATA".format(split.upper()))
    for idx in iterable:

        narr = panoptic_narratives[idx]
        # {'image_id': '226461', 'annotator_id': 36, 'caption': 'In this picture we can able to see a black bear on this tree.', 
        # 'segments': [{'utterance': 'in this', 'segment_ids': [], 'plural': False, 'noun': False}, 
        # {'utterance': 'picture', 'segment_ids': [], 'plural': False, 'noun': True}, 
        # {'utterance': 'we can able to see a', 'segment_ids': [], 'plural': False, 'noun': False}, 
        # {'utterance': 'black bear', 'segment_ids': ['1648163'], 'plural': False, 'noun': True}, 
        # {'utterance': 'on this', 'segment_ids': [], 'plural': False, 'noun': False}, 
        # {'utterance': 'tree', 'segment_ids': ['3165766'], 'plural': False, 'noun': True}, 
        # {'utterance': '.', 'segment_ids': [], 'plural': False, 'noun': False}]}
        # words = tokenizer.basic_tokenizer.tokenize(narr["caption"].strip())
        # ['in', 'this', 'picture', 'we', 'can', 'able', 'to', 'see', 'a', 'black', 'bear', 'on', 'this', 'tree', '.']
        words = []
        for token in tokenizer.basic_tokenizer.tokenize(narr["caption"].strip()):
                for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                    words.append(sub_token)

        segments = narr["segments"]
        narr["boxes"] = []
        narr["noun_vector"] = []

        image_id = int(narr["image_id"])
        panoptic_ann = panoptic_anns[image_id]
        segment_infos = {}
        for s in panoptic_ann["segments_info"]:
            idi = s["id"]
            segment_infos[idi] = s
        # box ann in panoptic segmentation
        nom_count = 0
        for seg in segments:
            utter = seg["utterance"].strip()
            # "in this"
            if "n't" in utter.lower():
                ind = utter.lower().index("n't")
                all_words1 = []
                for w in tokenizer.basic_tokenizer.tokenize(utter[:ind]):
                    for w_s in tokenizer.wordpiece_tokenizer.tokenize(w):
                        all_words1.append(w_s)
                all_words2 = []
                for w in tokenizer.basic_tokenizer.tokenize(utter[ind + 3 :]):
                    for w_s in tokenizer.wordpiece_tokenizer.tokenize(w):
                        all_words2.append(w_s)

                all_words = all_words1 + ["'", "t"] + all_words2
            else:
                all_words = []
                for w in tokenizer.basic_tokenizer.tokenize(utter):
                    for w_s in tokenizer.wordpiece_tokenizer.tokenize(w):
                        all_words.append(w_s)

                
            # ['in', 'this']

            nom_count = nom_count + 1 if len(seg["segment_ids"]) > 0 else nom_count

            for word in all_words:
                word_pi = word

                if not seg["noun"]:
                    narr["boxes"].append([[0] * 4])
                    narr["noun_vector"].append(0)
                elif len(seg["segment_ids"]) == 0:
                    narr["boxes"].append([[0] * 4])
                    narr["noun_vector"].append(0)
                elif len(seg["segment_ids"]) > 0:
                    ids_list = seg["segment_ids"]
                    nose = []
                    for lab in ids_list:
                        box = segment_infos[int(lab)]["bbox"]
                        nose.append(box)
                    narr["boxes"].append(nose)
                    narr["noun_vector"].append(nom_count)
                else:
                    raise ValueError("Error in data")

        if len(words) == len(narr["boxes"]):
            labels = [[-1 for i in sublist] for sublist in narr["boxes"]]
            ann_mask = [
                [True if ann == [0] * 4 else False for ann in sublist]
                for sublist in narr["boxes"]
            ]
            labels = [
                [-2 if m else l for (m, l) in zip(submask, sublabels)]
                for (submask, sublabels) in zip(ann_mask, labels)
            ]

            narr["labels"] = labels
            if len(labels) > max_len:
                max_len = len(labels)

            del narr["segments"]
            all_dict.append(narr)

        else:
            cont += 1

    tqdm.write("{} DATA FORMATED".format(split.upper()), end="\r")

    save_json(
        osp.join(
            "/mnt/data1/dzh/PNG-main/panoptic_narrative_grounding/annotations/png_coco_{}_dataloader.json".format(split),
        ),
        all_dict,
    )

tqdm.write("{} Narratives Excluded".format(cont))
tqdm.write(f'{max_len}')
