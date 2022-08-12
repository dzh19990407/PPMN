# NUM_GPUS=1
# EXP_DIR="/mnt/data1/dzh/PNG-main/detectron2"
# CONFIG_PATH="configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
# NUM_WORKERS=0

mkdir -p datasets/coco
ln -s /mnt/data3/dzh/MM2022/PNG-main-old/panoptic_narrative_grounding/annotations/instances /mnt/data3/dzh/MM2022/PNG-main/baseline/datasets/coco/annotations
ln -s /mnt/data3/dzh/MM2022/PNG-main-old/panoptic_narrative_grounding/images/train2017/train2017 /mnt/data3/dzh/MM2022/PNG-main/baseline/datasets/coco/train2017
ln -s /mnt/data3/dzh/MM2022/PNG-main-old/panoptic_narrative_grounding/images/val2017/val2017 /mnt/data3/dzh/MM2022/PNG-main/baseline/datasets/coco/val2017
ln -s /mnt/data3/dzh/MM2022/PNG-main-old/panoptic_narrative_grounding/annotations/panoptic_segmentation/train2017 /mnt/data3/dzh/MM2022/PNG-main/baseline/datasets/coco/panoptic_train2017
ln -s /mnt/data3/dzh/MM2022/PNG-main-old/panoptic_narrative_grounding/annotations/panoptic_segmentation/val2017 /mnt/data3/dzh/MM2022/PNG-main/baseline/datasets/coco/panoptic_val2017
ln -s /mnt/data3/dzh/MM2022/PNG-main-old/panoptic_narrative_grounding/annotations/panoptic_stuff_train2017 /mnt/data3/dzh/MM2022/PNG-main/baseline/datasets/coco/panoptic_stuff_train2017
ln -s /mnt/data3/dzh/MM2022/PNG-main-old/panoptic_narrative_grounding/annotations/panoptic_stuff_val2017 /mnt/data3/dzh/MM2022/PNG-main/baseline/datasets/coco/panoptic_stuff_val2017

# # for epoch in {0000337..0014195..0000338}
# # do
# WEIGHTS=$EXP_DIR"/model_final_cafdb1.pkl"
# CUDA_VISIBLE_DEVICES=1,2,3 python ./tools/train_net.py --num-gpus $NUM_GPUS \
#     --eval-only \
#     --config-file $CONFIG_PATH \
#     --dist-url tcp://0.0.0.0:12340 OUTPUT_DIR "../data/panoptic_narrative_grounding/features/val2017" \
#     MODEL.WEIGHTS $WEIGHTS \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS
# # done