# config setting
EXP_NAME=
# EXP_NAME=debug
DETECTRON2_CKPT=${PROJECT_ROOT}/mm22/pretrained_models/fpn/model_final_cafdb1.pkl
OUTPUT_DIR=${PROJECT_ROOT}/${EXP_NAME}
BERT=${PROJECT_ROOT}/pretrained_models/bert/bert-base-uncased
BERT_TOKENIZE=${PROJECT_ROOT}/pretrained_models/bert/bert-base-uncased.txt
DATA_PATH=${PROJECT_ROOT}/panoptic_narrative_grounding
NODES=1
NUM_GPUS=4
BATCH_SIZE=12

if [ ! \( -d "datasets/coco" \) ]
then
    echo "Creating the Link!"
    mkdir -p datasets/coco
    ln -s ${PROJECT_ROOT}/panoptic_narrative_grounding/annotations datasets/coco/
    ln -s ${PROJECT_ROOT}/panoptic_narrative_grounding/images/train2017 datasets/coco/train2017
    ln -s ${PROJECT_ROOT}/panoptic_narrative_grounding/images/val2017 datasets/coco/val2017
    ln -s ${PROJECT_ROOT}/panoptic_narrative_grounding/annotations/panoptic_segmentation/train2017 datasets/coco/panoptic_train2017
    ln -s ${PROJECT_ROOT}/panoptic_narrative_grounding/annotations/panoptic_segmentation/val2017 datasets/coco/panoptic_val2017
    ln -s ${PROJECT_ROOT}/panoptic_narrative_grounding/annotations/panoptic_stuff_train2017 datasets/coco/panoptic_stuff_train2017
    ln -s ${PROJECT_ROOT}/panoptic_narrative_grounding/annotations/panoptic_stuff_val2017 datasets/coco/panoptic_stuff_val2017
fi

# trainging
 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=${NODES}  \
     main.py --training \
     --num_gpus ${NUM_GPUS} \
     --batch_size ${BATCH_SIZE} \
     --fpn_freeze \
     --output_dir ${OUTPUT_DIR} \
     --detectron2_ckpt ${DETECTRON2_CKPT} \
     --pretrained_bert ${BERT} \
     --bert_tokenize ${BERT_TOKENIZE} \
     --data_path ${DATA_PATH} \
     --num_points 200 \
     --num_stages 3 \
     $@

if [ \( -d "datasets/coco" \) ]
then
    echo "Removing the Link!"
    rm datasets/coco/annotations
    rm datasets/coco/panoptic_stuff_train2017
    rm datasets/coco/panoptic_stuff_val2017
    rm datasets/coco/panoptic_train2017
    rm datasets/coco/panoptic_val2017
    rm datasets/coco/train2017
    rm datasets/coco/val2017
    rm -rf datasets
fi