#!/usr/bin/env python3
import os, sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup paths
TOOLS_DIR = '/mnt/backup/dassault/ysf/airbus_hackathon/OpenPCDet/tools'
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(TOOLS_DIR, '..'))
os.chdir(TOOLS_DIR)

import _init_path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu

# Config and Checkpoint
CFG_FILE = os.path.join(TOOLS_DIR, 'cfgs', 'airbus_models', 'airbus_centerpoint.yaml')
CKPT_PATH = '/mnt/backup/dassault/ysf/airbus_hackathon/OpenPCDet/output/airbus_models/airbus_centerpoint/centerpoint_height_optimized/ckpt/checkpoint_epoch_120.pth'
CLASS_NAMES = ['Antenna', 'Cable', 'Electric_Pole', 'Wind_Turbine']

def main():
    cfg_from_yaml_file(CFG_FILE, cfg)
    logger = common_utils.create_logger()

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=4, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=CKPT_PATH, logger=logger, to_cpu=False)
    model.cuda().eval()

    results = []
    
    print(f"Running inference on {len(test_set)} samples...")
    with torch.no_grad():
        for i in tqdm(range(len(test_set))):
            try:
                data_dict = test_set[i]
                info = test_set.kitti_infos[i]
                
                # Map batch
                batch_dict = test_set.collate_batch([data_dict])
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = model(batch_dict)
            except Exception as e:
                print(f"Skipping frame {i} due to error: {e}")
                continue

            # Get predictions for this sample
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()

            # Filter by confidence threshold (0.3 for submission)
            mask = pred_scores > 0.3
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]

            # Collect results
            for box, label in zip(pred_boxes, pred_labels):
                results.append({
                    'ego_x': info.get('ego_x', 0),
                    'ego_y': info.get('ego_y', 0),
                    'ego_z': info.get('ego_z', 0),
                    'ego_yaw': info.get('ego_yaw', 0),
                    'bbox_center_x': box[0],
                    'bbox_center_y': box[1],
                    'bbox_center_z': box[2],
                    'bbox_length': box[3],
                    'bbox_width': box[4],
                    'bbox_height': box[5],
                    'bbox_yaw': box[6],
                    'class_ID': int(label - 1),
                    'class_label': CLASS_NAMES[label - 1]
                })

    # Save to CSV
    df = pd.DataFrame(results)
    output_path = '/mnt/backup/dassault/ysf/airbus_hackathon/predictions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Saved {len(results)} detections to {output_path}")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == '__main__':
    main()
