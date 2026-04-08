#!/usr/bin/env python3
"""
Compute detailed metrics for CenterPoint:
1. Confusion Matrix (GT Class vs Pred Class at IoU 0.3)
2. Average Precision (AP) per class and mAP
"""
import os, sys
import numpy as np
import torch
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
from pcdet.ops.iou3d_nms import iou3d_nms_utils

CFG_FILE = os.path.join(TOOLS_DIR, 'cfgs', 'airbus_models', 'airbus_centerpoint.yaml')
CKPT_PATH = '/mnt/backup/dassault/ysf/airbus_hackathon/OpenPCDet/output/airbus_models/airbus_centerpoint/centerpoint_height_optimized/ckpt/checkpoint_epoch_120.pth'
CLASS_NAMES = ['Antenna', 'Cable', 'Electric_Pole', 'Wind_Turbine']
IOU_THRESH = 0.3

def voc_ap(rec, prec):
    """
    Compute AP using the VOC 11-point interpolation or all-point interpolation.
    """
    # Append points for zero and one
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To compute area under PR curve, look for points where X axis (recall) changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum of (delta_recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def main():
    cfg_from_yaml_file(CFG_FILE, cfg)
    logger = common_utils.create_logger()

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=4, dist=False, workers=4, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=CKPT_PATH, logger=logger, to_cpu=False)
    model.cuda().eval()
    
    # Store all detections for AP calculation: (score, class_id, is_tp, frame_id)
    all_detections = {c: [] for c in CLASS_NAMES}
    class_gt_counts = {c: 0 for c in CLASS_NAMES}
    
    # Confusion Matrix: GT Class (row) vs Pred Class (col)
    # Rows: 0-3 (Classes), 4 (Missed/Background)
    # Cols: 0-3 (Classes), 4 (False Positive/Background)
    confusion_matrix = np.zeros((5, 5))
    
    print("Gathering predictions...")
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(test_loader)):
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)
            
            for b in range(batch_dict['batch_size']):
                frame_id = batch_idx * 4 + b
                
                gt_boxes = batch_dict['gt_boxes'][b]
                valid = gt_boxes.sum(dim=-1) != 0
                gt_boxes = gt_boxes[valid]
                
                pred_boxes = pred_dicts[b]['pred_boxes']
                pred_labels = pred_dicts[b]['pred_labels']
                pred_scores = pred_dicts[b]['pred_scores']
                
                # Update GT counts
                for i in range(len(gt_boxes)):
                    gt_cls_id = int(gt_boxes[i, -1].item()) - 1
                    class_gt_counts[CLASS_NAMES[gt_cls_id]] += 1
                
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7], gt_boxes[:, :7])
                    iou = iou.cpu().numpy()
                    
                    # Match each prediction to the best GT
                    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
                    
                    # For confusion matrix, we look at the interaction
                    # First, populate matches
                    for p_idx in range(len(pred_boxes)):
                        p_cls_id = int(pred_labels[p_idx]) - 1
                        p_score = pred_scores[p_idx].item()
                        
                        best_gt_idx = np.argmax(iou[p_idx, :])
                        best_iou = iou[p_idx, best_gt_idx]
                        
                        if best_iou >= IOU_THRESH:
                            gt_cls_id = int(gt_boxes[best_gt_idx, -1]) - 1
                            # Update confusion matrix
                            confusion_matrix[gt_cls_id, p_cls_id] += 1
                            
                            # For TP/FP calculation in AP:
                            # A detection is TP if it's correct class AND hasn't matched this GT yet
                            is_tp = 0
                            if p_cls_id == gt_cls_id and not matched_gt[best_gt_idx]:
                                is_tp = 1
                                matched_gt[best_gt_idx] = True
                            
                            all_detections[CLASS_NAMES[p_cls_id]].append((p_score, is_tp))
                        else:
                            # False Positive (Background row)
                            confusion_matrix[4, p_cls_id] += 1
                            all_detections[CLASS_NAMES[p_cls_id]].append((p_score, 0))
                    
                    # Any GT not matched is a False Negative
                    for g_idx in range(len(gt_boxes)):
                        if not matched_gt[g_idx]:
                            gt_cls_id = int(gt_boxes[g_idx, -1]) - 1
                            confusion_matrix[gt_cls_id, 4] += 1
                
                elif len(gt_boxes) > 0:
                    # All GT are Missed
                    for i in range(len(gt_boxes)):
                        gt_cls_id = int(gt_boxes[i, -1]) - 1
                        confusion_matrix[gt_cls_id, 4] += 1
                
                elif len(pred_boxes) > 0:
                    # All Pred are False Positives
                    for i in range(len(pred_boxes)):
                        p_cls_id = int(pred_labels[i]) - 1
                        confusion_matrix[4, p_cls_id] += 1
                        all_detections[CLASS_NAMES[p_cls_id]].append((pred_scores[i].item(), 0))

    # Calculate AP for each class
    print("\n--- AVERAGE PRECISION (mAP) ---")
    aps = []
    for cls in CLASS_NAMES:
        detections = all_detections[cls]
        gt_count = class_gt_counts[cls]
        
        if gt_count == 0:
            print(f"{cls:>15}: N/A (No GT)")
            continue
            
        if len(detections) == 0:
            print(f"{cls:>15}: 0.00% (No Pred)")
            aps.append(0.0)
            continue
            
        # Sort by score descending
        detections.sort(key=lambda x: x[0], reverse=True)
        
        tps = np.array([d[1] for d in detections])
        fps = 1 - tps
        
        tp_sum = np.cumsum(tps)
        fp_sum = np.cumsum(fps)
        
        recall = tp_sum / gt_count
        precision = tp_sum / np.maximum(tp_sum + fp_sum, np.finfo(np.float64).eps)
        
        ap = voc_ap(recall, precision)
        aps.append(ap)
        print(f"{cls:>15}: {ap*100:6.2f}%")
        
    mAP = np.mean(aps) if aps else 0
    print(f"{'-'*25}")
    print(f"{'mAP':>15}: {mAP*100:6.2f}%")
    
    # Print Confusion Matrix
    print("\n--- CONFUSION MATRIX (IoU >= 0.3) ---")
    print("Rows: Ground Truth | Columns: Predictions")
    names_5 = CLASS_NAMES + ['Missed/Bg']
    
    header = f"{'GT \\ Pred':>15} | " + " | ".join([f"{n:>12}" for n in names_5])
    print(header)
    print("-" * len(header))
    
    for i in range(5):
        row_vals = " | ".join([f"{int(confusion_matrix[i, j]):>12}" for j in range(5)])
        print(f"{names_5[i]:>15} | {row_vals}")

    # Accuracy Metrics from Confusion Matrix
    print("\n--- ANALYSIS ---")
    diag = np.diag(confusion_matrix[:4, :4])
    total_matches = confusion_matrix[:4, :4].sum()
    if total_matches > 0:
        cls_acc = diag.sum() / total_matches * 100
        print(f"Classification Accuracy (on matched boxes): {cls_acc:.2f}%")
        print("This shows how well the model identifies the class IF it finds the box.")
    else:
        print("No matches to compute classification accuracy.")

if __name__ == '__main__':
    main()
