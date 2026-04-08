#!/bin/bash

# Configuration
PROJECT_DIR="/mnt/backup/dassault/ysf/airbus_hackathon"
OPENPCDET_DIR="$PROJECT_DIR/OpenPCDet"
DATASET_CFG="cfgs/dataset_configs/airbus_dataset.yaml"
MODEL_CFG="cfgs/airbus_models/airbus_centerpoint.yaml"
TAG="centerpoint_height_optimized"
LOG_FILE="$PROJECT_DIR/centerpoint_optim_height.log"

# 1. Regenerate dataset infos (just to be safe)
echo "Regenerating dataset infos..."
cd $OPENPCDET_DIR
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos $DATASET_CFG

# 2. Launch training
echo "Launching CenterPoint training..."
cd tools
python3 train.py \
    --cfg_file $MODEL_CFG \
    --batch_size 4 \
    --epochs 120 \
    --workers 4 \
    --extra_tag $TAG \
    2>&1 | tee $LOG_FILE
