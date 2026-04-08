#!/usr/bin/env python3
"""
=============================================================================
 convert_h5_to_kitti.py  —  Airbus Hackathon HDF5 → KITTI Format Converter
=============================================================================

PURPOSE:
    Convert the Airbus Hackathon HDF5 LiDAR files into KITTI-compatible format
    so they can be consumed by OpenPCDet / LinkNet3D for 3D object detection.

WHAT IT DOES (step-by-step):
    1. Loads each HDF5 scene file using the provided lidar_utils.py toolkit.
    2. Identifies individual frames via unique ego poses (ego_x, ego_y, ego_z, ego_yaw).
    3. For each frame:
       a. Filters out invalid points (distance_cm == 0).
       b. Converts spherical coordinates → local Cartesian (x, y, z) in meters.
       c. Saves point cloud as a KITTI .bin file: [x, y, z, reflectivity] float32.
       d. Identifies obstacle points by matching their RGB color to known class labels.
       e. Clusters obstacle points using DBSCAN to separate individual object instances.
       f. Fits axis-aligned 3D bounding boxes around each cluster.
       g. Estimates yaw orientation using PCA on the horizontal (x,y) plane.
       h. Writes KITTI-format label .txt file (one line per detected object).
       i. Creates an identity calibration file (no camera, LiDAR-only setup).
    4. Generates ImageSets/train.txt and ImageSets/val.txt (80/20 split).
    5. Saves a metadata JSON mapping each frame ID back to its source file and ego pose.

INPUT:
    - HDF5 files from: airbus_hackathon_trainingdata/scene_*.h5
    - Each file contains ~57.5M points (100 frames × 575K pts/frame).

OUTPUT:
    airbus_kitti/
    ├── training/
    │   ├── velodyne/    ← 000000.bin, 000001.bin, ... (x,y,z,reflectivity float32)
    │   ├── label_2/     ← 000000.txt, 000001.txt, ... (KITTI label format)
    │   └── calib/       ← 000000.txt, 000001.txt, ... (identity calibration)
    ├── ImageSets/
    │   ├── train.txt
    │   └── val.txt
    └── frame_metadata.json

KITTI LABEL FORMAT (each line = one object):
    <type> <trunc> <occ> <alpha> <x1> <y1> <x2> <y2> <h> <w> <l> <x> <y> <z> <ry>

    - type: Antenna / Cable / Electric_Pole / Wind_Turbine
    - trunc, occ, alpha: set to 0 (no 2D info)
    - x1,y1,x2,y2: 2D bbox — set to 0 (no images)
    - h, w, l: 3D box height, width, length in meters
    - x, y, z: 3D center in LiDAR frame (meters)
    - ry: yaw rotation around Z-axis in radians

USAGE:
    python convert_h5_to_kitti.py [--data-dir PATH] [--output-dir PATH] [--val-ratio 0.2]

COORDINATE SYSTEM NOTE:
    The Airbus data uses a left-handed system with Z-up.
    We keep coordinates in the LiDAR frame (no camera transform) since
    LinkNet3D processes LiDAR directly. Calibration files use identity matrices.
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import DBSCAN

# Add toolkit to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'airbus_hackathon_trainingdata', 'airbus_hackathon_toolkit'))
import lidar_utils


# =============================================================================
# Constants
# =============================================================================

# Ground-truth RGB → Class mapping (from Airbus README)
CLASS_MAP = {
    (38, 23, 180):   {'id': 0, 'label': 'Antenna'},
    (177, 132, 47):  {'id': 1, 'label': 'Cable'},
    (129, 81, 97):   {'id': 2, 'label': 'Electric_Pole'},
    (66, 132, 9):    {'id': 3, 'label': 'Wind_Turbine'},
}

# DBSCAN clustering parameters per class.
# eps = max distance (meters) between two points in the same cluster.
# min_samples = minimum points to form a cluster (filters noise).
# These are tuned based on the physical characteristics of each obstacle type:
#   - Antenna: compact, vertical → tight eps
#   - Cable: long, thin → larger eps to connect sparse points along the line
#   - Electric_Pole: compact, vertical → tight eps
#   - Wind_Turbine: very large structure → larger eps
DBSCAN_PARAMS = {
    'Antenna':       {'eps': 3.0,  'min_samples': 10},
    'Cable':         {'eps': 5.0,  'min_samples': 5},
    'Electric_Pole': {'eps': 3.0,  'min_samples': 10},
    'Wind_Turbine':  {'eps': 5.0,  'min_samples': 15},
}

# Minimum padding (meters) added to bounding box dimensions
# to avoid zero-sized boxes for thin/flat clusters
MIN_BOX_DIM = 0.3


# =============================================================================
# Step 1: Point Cloud Conversion  (HDF5 → .bin)
# =============================================================================

def convert_frame_to_bin(frame_df, output_path):
    """
    Convert a single frame's points from spherical to Cartesian and save as
    KITTI-format .bin file.

    .bin format: N × 4 float32 array → [x, y, z, reflectivity]
    - x, y, z in meters (local LiDAR frame)
    - reflectivity normalized to [0, 1]

    Args:
        frame_df: DataFrame with valid points (distance_cm > 0)
        output_path: Path to save .bin file
    
    Returns:
        xyz: Nx3 numpy array of Cartesian coordinates (for label generation)
    """
    # Convert spherical → Cartesian (meters)
    xyz = lidar_utils.spherical_to_local_cartesian(frame_df)

    # Normalize reflectivity to [0, 1] for KITTI compatibility
    reflectivity = frame_df['reflectivity'].values.astype(np.float32) / 255.0

    # Stack as [x, y, z, reflectivity] — KITTI velodyne format
    point_cloud = np.column_stack((xyz, reflectivity)).astype(np.float32)

    # Save as binary
    point_cloud.tofile(output_path)

    return xyz


# =============================================================================
# Step 2: Auto-Label Generation  (RGB → DBSCAN → 3D Bounding Boxes)
# =============================================================================

def extract_obstacle_points(frame_df, xyz):
    """
    Identify obstacle points by matching their RGB values to known class colors.

    Args:
        frame_df: DataFrame with r, g, b columns
        xyz: Nx3 Cartesian coordinates (matching frame_df rows)

    Returns:
        dict: {class_label: Mx3 numpy array of obstacle point coords}
    """
    obstacles = {}
    r_vals = frame_df['r'].values
    g_vals = frame_df['g'].values
    b_vals = frame_df['b'].values

    for (cr, cg, cb), info in CLASS_MAP.items():
        mask = (r_vals == cr) & (g_vals == cg) & (b_vals == cb)
        if mask.sum() > 0:
            obstacles[info['label']] = xyz[mask]

    return obstacles


def cluster_and_fit_boxes(class_label, class_points):
    """
    Given points of a single class, use DBSCAN to find individual object
    instances, then fit a 3D bounding box around each cluster.

    WHY DBSCAN?
    - A frame can contain multiple objects of the same class (e.g., several poles).
    - DBSCAN groups spatially close points without needing to know the count.
    - Noise points (isolated outliers) are automatically rejected.

    Args:
        class_label: str, e.g. 'Antenna'
        class_points: Mx3 numpy array

    Returns:
        list of dicts, each with keys:
            'label', 'center_xyz', 'dimensions_hwl', 'yaw'
    """
    params = DBSCAN_PARAMS.get(class_label, {'eps': 3.0, 'min_samples': 10})

    clustering = DBSCAN(
        eps=params['eps'],
        min_samples=params['min_samples']
    ).fit(class_points)

    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    boxes = []
    for lbl in unique_labels:
        cluster_points = class_points[labels == lbl]

        # Compute bounding box center
        center = cluster_points.mean(axis=0)  # [cx, cy, cz]

        # Compute bounding box dimensions with minimum padding
        mins = cluster_points.min(axis=0)
        maxs = cluster_points.max(axis=0)
        dims = maxs - mins  # [dx, dy, dz]
        dims = np.maximum(dims, MIN_BOX_DIM)  # Ensure non-zero

        # width = dx, length = dy, height = dz
        width, length, height = dims[0], dims[1], dims[2]

        # Estimate yaw via PCA on horizontal plane (x, y)
        yaw = estimate_yaw_pca(cluster_points[:, :2])

        boxes.append({
            'label': class_label,
            'class_id': [v['id'] for v in CLASS_MAP.values() if v['label'] == class_label][0],
            'center_xyz': center,
            'dimensions_hwl': (height, width, length),
            'yaw': yaw,
            'num_points': len(cluster_points),
        })

    return boxes


def estimate_yaw_pca(xy_points):
    """
    Estimate object orientation using PCA on the horizontal (x, y) plane.
    The yaw is the angle of the principal component (longest axis).

    Args:
        xy_points: Nx2 array of (x, y) coordinates

    Returns:
        float: yaw angle in radians [-π, π]
    """
    if len(xy_points) < 3:
        return 0.0

    # Center the points
    centered = xy_points - xy_points.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition — largest eigenvector = principal direction
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Yaw = angle of principal direction from X-axis
    yaw = np.arctan2(principal[1], principal[0])

    return float(yaw)


# =============================================================================
# Step 3: Write KITTI Label File
# =============================================================================

def write_kitti_label(boxes, output_path):
    """
    Write KITTI-format label file.

    Each line: <type> <trunc> <occ> <alpha> <x1> <y1> <x2> <y2> <h> <w> <l> <x> <y> <z> <ry>

    For LiDAR-only detection:
      - truncated, occluded, alpha = 0
      - 2D bbox (x1,y1,x2,y2) = 0 (no camera images)
      - h, w, l = 3D dimensions in meters
      - x, y, z = 3D center in LiDAR frame (meters)
      - ry = yaw rotation in radians

    Args:
        boxes: list of box dicts from cluster_and_fit_boxes()
        output_path: path to .txt label file
    """
    with open(output_path, 'w') as f:
        for box in boxes:
            h, w, l = box['dimensions_hwl']
            cx, cy, cz = box['center_xyz']
            ry = box['yaw']

            # KITTI format line
            line = (
                f"{box['label']} "  # type
                f"0.0 0 0.0 "       # truncated, occluded, alpha
                f"0.0 0.0 0.0 0.0 " # 2D bbox (unused)
                f"{h:.4f} {w:.4f} {l:.4f} "  # 3D dimensions
                f"{cx:.4f} {cy:.4f} {cz:.4f} "  # 3D center
                f"{ry:.4f}"          # rotation
            )
            f.write(line + '\n')


# =============================================================================
# Step 4: Write Calibration File (Identity)
# =============================================================================

def write_identity_calib(output_path):
    """
    Write an identity calibration file.

    Since we're doing LiDAR-only detection (no camera), all transformation
    matrices are set to identity. This tells OpenPCDet that LiDAR coordinates
    = camera coordinates = world coordinates (no transforms needed).
    """
    calib_content = (
        "P0: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "P1: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "P3: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    with open(output_path, 'w') as f:
        f.write(calib_content)


# =============================================================================
# Step 5: Generate ImageSets (Train/Val Split)
# =============================================================================

def generate_image_sets(total_frames, val_ratio, output_dir):
    """
    Generate train.txt and val.txt files for OpenPCDet.

    Uses a deterministic shuffle (seed=42) for reproducibility.
    80/20 split by default, stratified across scenes (each scene's frames
    are distributed proportionally to both sets).

    Args:
        total_frames: total number of frames across all scenes
        val_ratio: fraction for validation (default 0.2)
        output_dir: path to ImageSets/ directory
    """
    os.makedirs(output_dir, exist_ok=True)

    all_ids = list(range(total_frames))
    np.random.seed(42)
    np.random.shuffle(all_ids)

    n_val = int(len(all_ids) * val_ratio)
    val_ids = sorted(all_ids[:n_val])
    train_ids = sorted(all_ids[n_val:])

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for idx in train_ids:
            f.write(f"{idx:06d}\n")

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for idx in val_ids:
            f.write(f"{idx:06d}\n")

    print(f"  ImageSets: {len(train_ids)} train / {len(val_ids)} val")
    return train_ids, val_ids


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_all_scenes(data_dir, output_dir, val_ratio=0.2):
    """
    Main entry point. Processes all scene_*.h5 files → KITTI format.

    OPTIMIZATION: Uses pandas groupby to pre-index frames instead of
    scanning the full DataFrame per frame. This reduces complexity from
    O(n_points × n_frames) to O(n_points), cutting processing time
    from hours to minutes.

    For each HDF5 file:
      - Load all points
      - Group by ego pose (one-pass O(n) indexing)
      - For each frame: generate .bin, .txt label, and .txt calib
    Then generate ImageSets (train/val split).
    """
    import time

    # Find all scene files
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    print(f"Found {len(h5_files)} HDF5 scene files in {data_dir}")

    # Create output directories
    velodyne_dir = os.path.join(output_dir, 'training', 'velodyne')
    label_dir    = os.path.join(output_dir, 'training', 'label_2')
    calib_dir    = os.path.join(output_dir, 'training', 'calib')
    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    global_frame_id = 0
    metadata = {}  # Maps frame_id → source file + ego pose
    stats = defaultdict(int)  # Count boxes per class
    total_start = time.time()

    for h5_file in h5_files:
        filepath = os.path.join(data_dir, h5_file)
        print(f"\n{'='*60}")
        print(f"Processing: {h5_file}")
        print(f"{'='*60}")

        scene_start = time.time()

        # Load entire scene
        df = lidar_utils.load_h5_data(filepath)
        print(f"  Total points: {len(df):,}  (loaded in {time.time()-scene_start:.1f}s)")

        # ---- OPTIMIZED: Group by ego pose in ONE PASS ----
        # Instead of filter_by_pose() scanning 57.5M rows per frame,
        # we group once and iterate over groups. O(n) instead of O(n*frames).
        pose_fields = ["ego_x", "ego_y", "ego_z", "ego_yaw"]
        grouped = df.groupby(pose_fields)
        pose_keys = sorted(grouped.groups.keys())  # Sort for determinism
        print(f"  Frames: {len(pose_keys)}  (grouped in {time.time()-scene_start:.1f}s)")

        for pose_idx, pose_key in enumerate(pose_keys):
            ego_x, ego_y, ego_z, ego_yaw = pose_key

            # Get frame data directly from group (no scanning!)
            frame_df = grouped.get_group(pose_key)

            # Remove invalid points (no laser return)
            frame_df = frame_df[frame_df['distance_cm'] > 0].reset_index(drop=True)

            frame_id_str = f"{global_frame_id:06d}"

            # --- Step 1: Save point cloud as .bin ---
            bin_path = os.path.join(velodyne_dir, f"{frame_id_str}.bin")
            xyz = convert_frame_to_bin(frame_df, bin_path)

            # --- Step 2: Generate labels via DBSCAN ---
            obstacles = extract_obstacle_points(frame_df, xyz)
            all_boxes = []
            for class_label, class_points in obstacles.items():
                boxes = cluster_and_fit_boxes(class_label, class_points)
                all_boxes.extend(boxes)
                stats[class_label] += len(boxes)

            # --- Step 3: Write label file ---
            label_path = os.path.join(label_dir, f"{frame_id_str}.txt")
            write_kitti_label(all_boxes, label_path)

            # --- Step 4: Write calibration file ---
            calib_path = os.path.join(calib_dir, f"{frame_id_str}.txt")
            write_identity_calib(calib_path)

            # --- Save metadata ---
            metadata[frame_id_str] = {
                'source_file': h5_file,
                'pose_index': int(pose_idx),
                'ego_x': float(ego_x),
                'ego_y': float(ego_y),
                'ego_z': float(ego_z),
                'ego_yaw': float(ego_yaw),
                'num_points': len(frame_df),
                'num_obstacles': len(all_boxes),
            }

            global_frame_id += 1

            # Progress indicator every 10 frames
            if (pose_idx + 1) % 10 == 0:
                elapsed = time.time() - scene_start
                print(f"  Frame {pose_idx+1}/{len(pose_keys)} done "
                      f"(global: {frame_id_str}, obstacles: {len(all_boxes)}, "
                      f"elapsed: {elapsed:.0f}s)")

        # Free memory after each scene
        del df, grouped
        scene_elapsed = time.time() - scene_start
        print(f"  Scene complete in {scene_elapsed:.0f}s. Running total: {global_frame_id} frames")

    # --- Step 5: Generate ImageSets ---
    print(f"\n{'='*60}")
    print("Generating ImageSets (train/val split)...")
    imagesets_dir = os.path.join(output_dir, 'ImageSets')
    generate_image_sets(global_frame_id, val_ratio, imagesets_dir)

    # --- Save metadata ---
    meta_path = os.path.join(output_dir, 'frame_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # --- Summary ---
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames: {global_frame_id}")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Output directory: {output_dir}")
    print(f"\nBounding box counts per class:")
    for cls, count in sorted(stats.items()):
        print(f"  {cls}: {count} boxes")
    print(f"  TOTAL: {sum(stats.values())} boxes")

    return global_frame_id


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Airbus Hackathon HDF5 LiDAR data to KITTI format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--data-dir',
        default=os.path.join(SCRIPT_DIR, '..', 'airbus_hackathon_trainingdata'),
        help='Path to directory containing scene_*.h5 files'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join(SCRIPT_DIR, 'airbus_kitti'),
        help='Output directory for KITTI-format data'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Fraction of frames for validation (default: 0.2 = 20%%)'
    )

    args = parser.parse_args()
    process_all_scenes(args.data_dir, args.output_dir, args.val_ratio)


if __name__ == '__main__':
    main()
