#!/usr/bin/env python3
"""
Robustness Test: Subsample validation point clouds to 25% and measure recall.
Ensures model handles the 'Robustness' criteria of the hackathon.
"""
import os, sys
import torch
import numpy as np

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

CFG_FILE = os.path.join(TOOLS_DIR, 'cfgs', 'airbus_models', 'airbus_centerpoint.yaml')
CKPT_PATH = '/mnt/backup/dassault/ysf/airbus_hackathon/OpenPCDet/output/airbus_models/airbus_centerpoint/centerpoint_height_optimized/ckpt/checkpoint_epoch_120.pth'

def main():
    cfg_from_yaml_file(CFG_FILE, cfg)
    logger = common_utils.create_logger()

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=1, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=CKPT_PATH, logger=logger, to_cpu=False)
    model.cuda().eval()

    total_recalled_01 = 0
    total_recalled_03 = 0
    total_gt = 0
    
    # Test on a subset of 50 frames for speed
    n_frames = 50
    print(f"Running robustness test on {n_frames} frames with 25% density...")
    
    with torch.no_grad():
        for i in range(n_frames):
            data_dict = test_set[i]
            
            # Manually subsample points to 25%
            points = data_dict['points']
            n_points = len(points)
            sub_indices = np.random.choice(n_points, n_points // 4, replace=False)
            data_dict['points'] = points[sub_indices]
            
            batch_dict = test_set.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)
            
            pred_dicts, _ = model(batch_dict)
            
            gt_boxes = batch_dict['gt_boxes'][0]
            valid_gt = gt_boxes.sum(dim=-1) != 0
            gt_boxes = gt_boxes[valid_gt]
            
            pred_boxes = pred_dicts[0]['pred_boxes']
            
            if len(gt_boxes) > 0:
                total_gt += len(gt_boxes)
                if len(pred_boxes) > 0:
                    from pcdet.ops.iou3d_nms import iou3d_nms_utils
                    iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7], gt_boxes[:, :7])
                    max_ious = iou.max(dim=0)[0]
                    total_recalled_01 += (max_ious >= 0.1).sum().item()
                    total_recalled_03 += (max_ious >= 0.3).sum().item()

    print(f"\n--- ROBUSTNESS RESULTS (25% Density) ---")
    print(f"Recall@0.1: {total_recalled_01}/{total_gt} = {total_recalled_01/total_gt*100:.2f}%")
    print(f"Recall@0.3: {total_recalled_03}/{total_gt} = {total_recalled_03/total_gt*100:.2f}%")
    print(f"Stability Factor (R@0.1_25% / R@0.1_100%): {(total_recalled_01/total_gt) / 0.2423:.2f}")

if __name__ == '__main__':
    main()
