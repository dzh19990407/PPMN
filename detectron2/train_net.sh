python tools/train_net.py --num-gpus 4 \
    --eval-only \
    --config-file "configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_train.yaml" \
    --dist-url tcp://0.0.0.0:12340 OUTPUT_DIR "../data/panoptic_narrative_grounding/features/train2017" \
    MODEL.WEIGHTS "/mnt/data1/dzh/PNG-main/detectron2/model_final_cafdb1.pkl" \