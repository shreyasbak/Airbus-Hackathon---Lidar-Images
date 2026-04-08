#!/usr/bin/env python3
"""
Evaluate multiple LinkNet3D checkpoints on the Airbus validation set.
Collects recall metrics and per-class detection statistics.
"""
import os
import sys
import glob
import re
import json
import numpy as np
import torch

# Insert OpenPCDet paths
TOOLS_DIR = os.path.join(os.path.dirname(__file__), 'OpenPCDet', 'tools')
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(TOOLS_DIR, '..'))

import _init_path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

# ---- Config ----
CFG_FILE = os.path.join(TOOLS_DIR, 'cfgs', 'airbus_models', 'linknet3d_airbus.yaml')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'OpenPCDet', 'output',
                         'airbus_models', 'linknet3d_airbus', 'optimized_run1', 'ckpt')

# Epochs to evaluate (spread across training)
EPOCHS_TO_EVAL = [100, 105, 110, 115, 120]

CLASS_NAMES = ['Antenna', 'Cable', 'Electric_Pole', 'Wind_Turbine']
RECALL_THRESHOLDS = [0.3, 0.5, 0.7]


def evaluate_single_checkpoint(model, test_loader, ckpt_path, class_names):
    """Evaluate a single checkpoint — returns recall stats and per-class detection info."""
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=None, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.cuda()
    model.eval()

    # Track metrics
    recall_dict = {}
    per_class_gt_count = {c: 0 for c in class_names}
    per_class_pred_count = {c: 0 for c in class_names}
    per_class_recalled = {thresh: {c: 0 for c in class_names} for thresh in RECALL_THRESHOLDS}
    total_pred_boxes = 0
    total_gt_boxes = 0
    num_samples = 0

    with torch.no_grad():
        for batch_dict in test_loader:
            # Move to GPU
            for key, val in batch_dict.items():
                if isinstance(val, torch.Tensor):
                    batch_dict[key] = val.cuda()

            pred_dicts, ret_dict = model(batch_dict)
            num_samples += batch_dict['batch_size']

            # Accumulate recall
            for key, val in ret_dict.items():
                if key not in recall_dict:
                    recall_dict[key] = 0
                recall_dict[key] += val

            # Per-class analysis
            batch_size = batch_dict['batch_size']
            for b in range(batch_size):
                gt_boxes = batch_dict['gt_boxes'][b]
                # Remove padding zeros
                valid_mask = gt_boxes.sum(dim=-1) != 0
                gt_boxes = gt_boxes[valid_mask]

                if gt_boxes.shape[0] > 0:
                    gt_labels = gt_boxes[:, -1].long()
                    total_gt_boxes += gt_boxes.shape[0]
                    for label_id in gt_labels:
                        cls_name = class_names[label_id.item() - 1]
                        per_class_gt_count[cls_name] += 1

                pred_boxes = pred_dicts[b]['pred_boxes']
                pred_labels = pred_dicts[b]['pred_labels']
                pred_scores = pred_dicts[b]['pred_scores']
                total_pred_boxes += pred_boxes.shape[0]

                for label_id in pred_labels:
                    cls_name = class_names[label_id.item() - 1]
                    per_class_pred_count[cls_name] += 1

                # Per-class recall at each threshold
                if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
                    from pcdet.ops.iou3d_nms import iou3d_nms_utils
                    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(
                        pred_boxes[:, :7], gt_boxes[:, :7]
                    )
                    for thresh in RECALL_THRESHOLDS:
                        for gt_idx in range(gt_boxes.shape[0]):
                            max_iou = iou_matrix[:, gt_idx].max().item() if iou_matrix.shape[0] > 0 else 0
                            gt_cls = class_names[gt_boxes[gt_idx, -1].long().item() - 1]
                            if max_iou >= thresh:
                                per_class_recalled[thresh][gt_cls] += 1

    # Compute aggregate recall
    gt_total = recall_dict.get('gt', 1)
    results = {
        'num_samples': num_samples,
        'total_gt_boxes': total_gt_boxes,
        'total_pred_boxes': total_pred_boxes,
        'avg_pred_per_sample': total_pred_boxes / max(num_samples, 1),
        'overall_recall': {},
        'per_class_gt': per_class_gt_count,
        'per_class_pred': per_class_pred_count,
        'per_class_recall': {},
    }

    for thresh in RECALL_THRESHOLDS:
        key = f'rcnn_{thresh}'
        val = recall_dict.get(key, 0)
        results['overall_recall'][f'recall@{thresh}'] = val / max(gt_total, 1)

        per_cls_recall = {}
        for cls_name in class_names:
            gt_cnt = per_class_gt_count[cls_name]
            recalled = per_class_recalled[thresh][cls_name]
            per_cls_recall[cls_name] = recalled / max(gt_cnt, 1)
        results['per_class_recall'][f'recall@{thresh}'] = per_cls_recall

    return results


def main():
    # Load config
    cfg_from_yaml_file(CFG_FILE, cfg)
    cfg.TAG = 'linknet3d_airbus'
    cfg.EXP_GROUP_PATH = 'airbus_models'

    logger = common_utils.create_logger()

    # Build dataloader (validation)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=4,
        dist=False, workers=4, logger=logger, training=False
    )

    # Build model (structure only, weights loaded per checkpoint)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    all_results = {}
    for epoch in EPOCHS_TO_EVAL:
        ckpt_path = os.path.join(CKPT_DIR, f'checkpoint_epoch_{epoch}.pth')
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] Epoch {epoch}: checkpoint not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating Epoch {epoch}")
        print(f"{'='*60}")

        try:
            results = evaluate_single_checkpoint(model, test_loader, ckpt_path, CLASS_NAMES)
            all_results[epoch] = results

            # Print summary
            print(f"  GT boxes: {results['total_gt_boxes']}")
            print(f"  Pred boxes: {results['total_pred_boxes']} "
                  f"(avg {results['avg_pred_per_sample']:.1f}/sample)")
            print(f"  Overall recall:")
            for k, v in results['overall_recall'].items():
                print(f"    {k}: {v*100:.2f}%")
            print(f"  Per-class recall@0.3:")
            for cls, val in results['per_class_recall']['recall@0.3'].items():
                gt = results['per_class_gt'][cls]
                pred = results['per_class_pred'][cls]
                print(f"    {cls:15s}: recall={val*100:.1f}%  (GT={gt}, Pred={pred})")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    # Save full results to JSON
    output_path = os.path.join(os.path.dirname(__file__), 'evaluation_results.json')
    serializable = {}
    for epoch, res in all_results.items():
        serializable[str(epoch)] = res
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[SAVED] Full results to {output_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    header = f"{'Epoch':>6} | {'Pred/Samp':>10} | {'r@0.3':>7} | {'r@0.5':>7} | {'r@0.7':>7}"
    print(header)
    print("-" * len(header))
    for epoch in sorted(all_results.keys()):
        res = all_results[epoch]
        r03 = res['overall_recall'].get('recall@0.3', 0) * 100
        r05 = res['overall_recall'].get('recall@0.5', 0) * 100
        r07 = res['overall_recall'].get('recall@0.7', 0) * 100
        print(f"{epoch:>6} | {res['avg_pred_per_sample']:>10.1f} | {r03:>6.2f}% | {r05:>6.2f}% | {r07:>6.2f}%")

    # Per-class breakdown for best epoch
    if all_results:
        best_epoch = max(all_results.keys(),
                         key=lambda e: all_results[e]['overall_recall'].get('recall@0.3', 0))
        best = all_results[best_epoch]
        print(f"\n{'='*80}")
        print(f"PER-CLASS BREAKDOWN (Best epoch: {best_epoch})")
        print(f"{'='*80}")
        print(f"{'Class':>15} | {'GT':>5} | {'Pred':>5} | {'r@0.3':>7} | {'r@0.5':>7} | {'r@0.7':>7}")
        print("-" * 62)
        for cls in CLASS_NAMES:
            gt = best['per_class_gt'][cls]
            pred = best['per_class_pred'][cls]
            r03 = best['per_class_recall']['recall@0.3'][cls] * 100
            r05 = best['per_class_recall']['recall@0.5'][cls] * 100
            r07 = best['per_class_recall']['recall@0.7'][cls] * 100
            print(f"{cls:>15} | {gt:>5} | {pred:>5} | {r03:>6.2f}% | {r05:>6.2f}% | {r07:>6.2f}%")


if __name__ == '__main__':
    main()
