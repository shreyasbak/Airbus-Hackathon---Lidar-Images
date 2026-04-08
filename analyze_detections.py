#!/usr/bin/env python3
"""
Per-class analysis of LinkNet3D predictions on the Airbus validation set.
Runs inference, computes IoU between predictions and ground truth,
and prints a detailed per-class recall breakdown.
"""
import os, sys
import numpy as np
import torch

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
from pcdet.ops.iou3d_nms import iou3d_nms_utils

CFG_FILE = os.path.join(TOOLS_DIR, 'cfgs', 'airbus_models', 'airbus_centerpoint.yaml')
CKPT_PATH = '/mnt/backup/dassault/ysf/airbus_hackathon/OpenPCDet/output/airbus_models/airbus_centerpoint/centerpoint_height_optimized/ckpt/checkpoint_epoch_120.pth'
CLASS_NAMES = ['Antenna', 'Cable', 'Electric_Pole', 'Wind_Turbine']
THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7]


def main():
    np.random.seed(1024)
    cfg_from_yaml_file(CFG_FILE, cfg)
    cfg.TAG = 'linknet3d_airbus'
    cfg.EXP_GROUP_PATH = 'airbus_models'
    logger = common_utils.create_logger()

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=4, dist=False, workers=4, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=CKPT_PATH, logger=logger, to_cpu=False)
    model.cuda().eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")
    
    # Per-class tracking
    class_gt = {c: 0 for c in CLASS_NAMES}
    class_pred = {c: 0 for c in CLASS_NAMES}
    class_recalled = {t: {c: 0 for c in CLASS_NAMES} for t in THRESHOLDS}
    class_max_ious = {c: [] for c in CLASS_NAMES}  # Store max IoU per GT box
    class_pred_scores = {c: [] for c in CLASS_NAMES}  # Store prediction scores
    
    # Box size stats
    class_gt_sizes = {c: [] for c in CLASS_NAMES}   # (l, w, h) per GT box
    class_pred_sizes = {c: [] for c in CLASS_NAMES}  # (l, w, h) per pred box
    
    total_gt = 0
    total_pred = 0
    frames_with_gt = 0
    frames_with_pred = 0
    
    with torch.no_grad():
        for batch_dict in test_loader:
            load_data_to_gpu(batch_dict)
            
            pred_dicts, _ = model(batch_dict)
            batch_size = batch_dict['batch_size']
            
            for b in range(batch_size):
                gt_boxes = batch_dict['gt_boxes'][b]
                valid = gt_boxes.sum(dim=-1) != 0
                gt_boxes = gt_boxes[valid]
                
                pred_boxes = pred_dicts[b]['pred_boxes']
                pred_labels = pred_dicts[b]['pred_labels']
                pred_scores = pred_dicts[b]['pred_scores']
                
                if gt_boxes.shape[0] > 0:
                    frames_with_gt += 1
                    total_gt += gt_boxes.shape[0]
                    
                    gt_labels = gt_boxes[:, -1].long()
                    gt_dims = gt_boxes[:, 3:6].cpu().numpy()  # dx, dy, dz
                    
                    for i, (label_id, dims) in enumerate(zip(gt_labels, gt_dims)):
                        cls = CLASS_NAMES[label_id.item() - 1]
                        class_gt[cls] += 1
                        class_gt_sizes[cls].append(dims)
                
                if pred_boxes.shape[0] > 0:
                    frames_with_pred += 1
                    total_pred += pred_boxes.shape[0]
                    
                    pred_dims = pred_boxes[:, 3:6].cpu().numpy()
                    for i, (label_id, score, dims) in enumerate(zip(pred_labels, pred_scores, pred_dims)):
                        cls = CLASS_NAMES[label_id.item() - 1]
                        class_pred[cls] += 1
                        class_pred_scores[cls].append(score.item())
                        class_pred_sizes[cls].append(dims)
                
                # Compute IoU between pred and GT
                if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
                    iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7], gt_boxes[:, :7])
                    
                    for gt_idx in range(gt_boxes.shape[0]):
                        gt_cls = CLASS_NAMES[gt_boxes[gt_idx, -1].long().item() - 1]
                        max_iou = iou[:, gt_idx].max().item()
                        class_max_ious[gt_cls].append(max_iou)
                        
                        for thresh in THRESHOLDS:
                            if max_iou >= thresh:
                                class_recalled[thresh][gt_cls] += 1
                elif gt_boxes.shape[0] > 0:
                    # No predictions for this frame — all GT missed
                    for gt_idx in range(gt_boxes.shape[0]):
                        gt_cls = CLASS_NAMES[gt_boxes[gt_idx, -1].long().item() - 1]
                        class_max_ious[gt_cls].append(0.0)

    # ============================================================
    # PRINT RESULTS
    # ============================================================
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS — Epoch 80 Checkpoint")
    print(f"{'='*80}")
    print(f"Validation samples: 199")
    print(f"Frames with GT boxes: {frames_with_gt}")
    print(f"Frames with predictions: {frames_with_pred}")
    print(f"Total GT boxes: {total_gt}")
    print(f"Total predicted boxes: {total_pred}")

    # Overall recall table
    print(f"\n--- OVERALL RECALL ---")
    for thresh in THRESHOLDS:
        recalled = sum(class_recalled[thresh].values())
        recall = recalled / max(total_gt, 1) * 100
        print(f"  Recall@{thresh}: {recalled}/{total_gt} = {recall:.2f}%")

    # Per-class recall table
    print(f"\n--- PER-CLASS RECALL ---")
    header = f"{'Class':>15} | {'GT':>6} | {'Pred':>6} | "
    header += " | ".join([f"r@{t}" for t in THRESHOLDS])
    print(header)
    print("-" * len(header))
    for cls in CLASS_NAMES:
        gt = class_gt[cls]
        pred = class_pred[cls]
        recalls = []
        for thresh in THRESHOLDS:
            r = class_recalled[thresh][cls] / max(gt, 1) * 100
            recalls.append(f"{r:6.1f}%")
        print(f"{cls:>15} | {gt:>6} | {pred:>6} | {' | '.join(recalls)}")

    # IoU distribution per class
    print(f"\n--- IoU DISTRIBUTION PER CLASS (best matching pred for each GT box) ---")
    for cls in CLASS_NAMES:
        ious = class_max_ious[cls]
        if len(ious) > 0:
            ious_arr = np.array(ious)
            zero_pct = (ious_arr == 0).sum() / len(ious_arr) * 100
            print(f"  {cls}:")
            print(f"    N={len(ious)}, mean={ious_arr.mean():.4f}, median={np.median(ious_arr):.4f}, "
                  f"max={ious_arr.max():.4f}, zero_iou={zero_pct:.1f}%")
            print(f"    Percentiles: p25={np.percentile(ious_arr, 25):.4f}, "
                  f"p75={np.percentile(ious_arr, 75):.4f}, p90={np.percentile(ious_arr, 90):.4f}")

    # Box size comparison GT vs Pred
    print(f"\n--- BOX SIZE COMPARISON (GT vs Pred) ---")
    for cls in CLASS_NAMES:
        gt_sizes = class_gt_sizes[cls]
        pred_sizes = class_pred_sizes[cls]
        if len(gt_sizes) > 0:
            gt_arr = np.array(gt_sizes)
            print(f"  {cls} GT (N={len(gt_sizes)}):")
            print(f"    dx: mean={gt_arr[:,0].mean():.1f}, min={gt_arr[:,0].min():.1f}, max={gt_arr[:,0].max():.1f}")
            print(f"    dy: mean={gt_arr[:,1].mean():.1f}, min={gt_arr[:,1].min():.1f}, max={gt_arr[:,1].max():.1f}")
            print(f"    dz: mean={gt_arr[:,2].mean():.1f}, min={gt_arr[:,2].min():.1f}, max={gt_arr[:,2].max():.1f}")
        if len(pred_sizes) > 0:
            pred_arr = np.array(pred_sizes)
            print(f"  {cls} Pred (N={len(pred_sizes)}):")
            print(f"    dx: mean={pred_arr[:,0].mean():.1f}, min={pred_arr[:,0].min():.1f}, max={pred_arr[:,0].max():.1f}")
            print(f"    dy: mean={pred_arr[:,1].mean():.1f}, min={pred_arr[:,1].min():.1f}, max={pred_arr[:,1].max():.1f}")
            print(f"    dz: mean={pred_arr[:,2].mean():.1f}, min={pred_arr[:,2].min():.1f}, max={pred_arr[:,2].max():.1f}")

    # Prediction confidence stats
    print(f"\n--- PREDICTION CONFIDENCE DISTRIBUTION ---")
    for cls in CLASS_NAMES:
        scores = class_pred_scores[cls]
        if len(scores) > 0:
            scores_arr = np.array(scores)
            print(f"  {cls} (N={len(scores)}): mean={scores_arr.mean():.3f}, "
                  f"median={np.median(scores_arr):.3f}, "
                  f"min={scores_arr.min():.3f}, max={scores_arr.max():.3f}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
