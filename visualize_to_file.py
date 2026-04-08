#!/usr/bin/env python3
"""
Generate up to 10 frames showing point clouds with predicted 3D bounding boxes
colored by class, as required by the Airbus Hackathon submission.
"""
import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

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
OUTPUT_DIR = '/mnt/backup/dassault/ysf/airbus_hackathon/eval_results/submission_frames'
CLASS_NAMES = ['Antenna', 'Cable', 'Electric_Pole', 'Wind_Turbine']
CLASS_COLORS = {
    'Antenna':       '#FF3333',   # Bright Red
    'Cable':         '#3399FF',   # Bright Blue
    'Electric_Pole': '#FFCC00',   # Yellow
    'Wind_Turbine':  '#33FF33',   # Bright Green
}

def get_box_corners(box):
    """Get 8 corners of a 3D bounding box."""
    x, y, z, dx, dy, dz, heading = box
    l, w, h = dx, dy, dz
    corners_local = np.array([
        [ l/2,  w/2,  h/2], [ l/2,  w/2, -h/2],
        [ l/2, -w/2,  h/2], [ l/2, -w/2, -h/2],
        [-l/2,  w/2,  h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [-l/2, -w/2, -h/2]
    ])
    rot = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading),  np.cos(heading), 0],
        [0, 0, 1]
    ])
    return (rot @ corners_local.T).T + np.array([x, y, z])

def draw_3d_box(ax, box, color, linewidth=1.5):
    """Draw a wireframe 3D bounding box."""
    c = get_box_corners(box)
    edges = [
        [0,1],[1,3],[3,2],[2,0],  # front face
        [4,5],[5,7],[7,6],[6,4],  # back face
        [0,4],[1,5],[2,6],[3,7]   # pillars
    ]
    for e in edges:
        ax.plot3D(c[e,0], c[e,1], c[e,2], color=color, linewidth=linewidth)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg_from_yaml_file(CFG_FILE, cfg)
    logger = common_utils.create_logger()

    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=1, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=CKPT_PATH, logger=logger, to_cpu=False)
    model.cuda().eval()

    # 1. Scan the whole test set to find the frames with the most detections
    frame_stats = []
    print(f"Scanning {len(test_set)} frames to find the busiest ones...")
    with torch.no_grad():
        for i in range(len(test_set)):
            try:
                data_dict = test_set[i]
                batch_dict = test_set.collate_batch([data_dict])
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = model(batch_dict)

                pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
                mask = pred_scores > 0.3
                n_pred = np.sum(mask)
                frame_stats.append((i, n_pred))
            except Exception as e:
                print(f"Skipping frame {i} due to error: {e}")
                continue

    # 2. Sort by detection count (descending) and take top 10
    frame_stats.sort(key=lambda x: x[1], reverse=True)
    top_frames = frame_stats[:10]
    
    print("\nTop 10 Frames by Detection Count:")
    for f_idx, count in top_frames:
        print(f"  Frame {f_idx}: {count} detections")

    # 3. Generate screenshots for these top 10 frames
    saved = 0
    with torch.no_grad():
        for i, count in top_frames:
            data_dict = test_set[i]
            batch_dict = test_set.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)

            points = batch_dict['points'][:, 1:4].cpu().numpy()
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()

            mask = pred_scores > 0.3
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]

            # === Figure ===
            fig = plt.figure(figsize=(16, 10), facecolor='black')
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('black')
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_edgecolor('black')
            ax.tick_params(colors='white', labelsize=7)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')

            # Downsample points
            n_vis = min(len(points), 60000)
            idx = np.random.choice(len(points), n_vis, replace=False)
            p = points[idx]

            # Color points by height
            colors = np.zeros((n_vis, 3))
            mask_ground = p[:, 2] < 2.0
            mask_mid = (p[:, 2] >= 2.0) & (p[:, 2] < 15.0)
            mask_high = p[:, 2] >= 15.0
            colors[mask_ground] = [0.7, 0.15, 0.15]  # Red ground
            colors[mask_mid]    = [0.15, 0.3, 0.7]    # Blue vegetation
            colors[mask_high]   = [0.2, 0.7, 0.2]     # Green tall

            ax.scatter(p[:,0], p[:,1], p[:,2], s=0.15, c=colors, alpha=0.7)

            # Draw predicted boxes colored by class
            for box, label in zip(pred_boxes, pred_labels):
                cls = CLASS_NAMES[label - 1]
                draw_3d_box(ax, box, CLASS_COLORS[cls], linewidth=2.0)

            # Perspective
            ax.set_box_aspect([1, 1, 0.3])
            ax.view_init(elev=25, azim=-50)

            # Auto-zoom around detections
            cx = pred_boxes[:, 0].mean()
            cy = pred_boxes[:, 1].mean()
            spread = max(pred_boxes[:, 3:5].max() * 2, 70)
            ax.set_xlim(cx - spread, cx + spread)
            ax.set_ylim(cy - spread, cy + spread)
            ax.set_zlim(-5, 90)

            ax.set_xlabel('X (m)', fontsize=8)
            ax.set_ylabel('Y (m)', fontsize=8)
            ax.set_zlabel('Z (m)', fontsize=8)

            # Legend
            legend_elements = [
                Line2D([0],[0], color=CLASS_COLORS['Antenna'],      lw=2, label='Antenna'),
                Line2D([0],[0], color=CLASS_COLORS['Cable'],         lw=2, label='Cable'),
                Line2D([0],[0], color=CLASS_COLORS['Electric_Pole'], lw=2, label='Electric Pole'),
                Line2D([0],[0], color=CLASS_COLORS['Wind_Turbine'],  lw=2, label='Wind Turbine'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                      facecolor='black', edgecolor='white', labelcolor='white')

            # Title
            cls_counts = {c: 0 for c in CLASS_NAMES}
            for l in pred_labels:
                cls_counts[CLASS_NAMES[l-1]] += 1
            count_str = ", ".join([f"{c}: {n}" for c, n in cls_counts.items() if n > 0])
            ax.set_title(f"Rank #{saved+1} Busy Frame (Index {i}) | {count} detections | {count_str}",
                        color='white', fontsize=11, pad=10)

            save_path = os.path.join(OUTPUT_DIR, f'top_frame_{saved+1:02d}.png')
            plt.savefig(save_path, dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved {save_path} ({count} detections)")
            saved += 1

    print(f"\nDone! {saved} frames saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
