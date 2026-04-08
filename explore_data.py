#!/usr/bin/env python3
"""
Phase 1: Data Exploration
Explore the Airbus HDF5 LiDAR files to understand structure, frame counts,
class distributions, and coordinate ranges.
"""

import sys
import os
import h5py
import numpy as np
import pandas as pd
from collections import Counter

# Add toolkit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'airbus_hackathon_trainingdata', 'airbus_hackathon_toolkit'))
import lidar_utils

# ---- Config ----
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'airbus_hackathon_trainingdata')

# Class color mapping from README
CLASS_COLORS = {
    (38, 23, 180): ('Antenna', 0),
    (177, 132, 47): ('Cable', 1),
    (129, 81, 97): ('Electric_Pole', 2),
    (66, 132, 9): ('Wind_Turbine', 3),
}

def explore_file(filepath):
    """Explore a single HDF5 file."""
    filename = os.path.basename(filepath)
    print(f"\n{'='*70}")
    print(f"FILE: {filename} ({os.path.getsize(filepath)/1e6:.1f} MB)")
    print(f"{'='*70}")

    # 1. Raw HDF5 structure
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            ds = f[key]
            print(f"\n  Dataset: '{key}'")
            print(f"    Shape: {ds.shape}")
            print(f"    Dtype: {ds.dtype}")
            if ds.dtype.names:
                print(f"    Fields: {list(ds.dtype.names)}")
                for name in ds.dtype.names:
                    print(f"      - {name}: {ds.dtype[name]}")

    # 2. Load as DataFrame
    df = lidar_utils.load_h5_data(filepath)
    print(f"\n  Total points: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # 3. Unique poses (frames)
    poses = lidar_utils.get_unique_poses(df)
    if poses is not None:
        print(f"\n  Unique frames (poses): {len(poses)}")
        print(f"  Points per frame: min={poses['num_points'].min():,}, "
              f"max={poses['num_points'].max():,}, "
              f"mean={poses['num_points'].mean():,.0f}")

    # 4. Coordinate ranges
    print(f"\n  Coordinate ranges:")
    print(f"    distance_cm: [{df['distance_cm'].min()}, {df['distance_cm'].max()}]")
    print(f"    azimuth_raw: [{df['azimuth_raw'].min()}, {df['azimuth_raw'].max()}]")
    print(f"    elevation_raw: [{df['elevation_raw'].min()}, {df['elevation_raw'].max()}]")
    if 'reflectivity' in df.columns:
        print(f"    reflectivity: [{df['reflectivity'].min()}, {df['reflectivity'].max()}]")

    # 5. Valid points (distance > 0)
    valid_mask = df['distance_cm'] > 0
    n_valid = valid_mask.sum()
    print(f"\n  Valid points (distance>0): {n_valid:,} / {len(df):,} ({100*n_valid/len(df):.1f}%)")

    # 6. RGB / Class distribution
    if all(c in df.columns for c in ['r', 'g', 'b']):
        # Only look at valid points
        df_valid = df[valid_mask]
        rgb_tuples = list(zip(df_valid['r'].values, df_valid['g'].values, df_valid['b'].values))
        rgb_counts = Counter(rgb_tuples)

        print(f"\n  RGB class distribution (valid points only):")
        obstacle_points = 0
        for rgb, count in rgb_counts.most_common(20):
            if rgb in CLASS_COLORS:
                label, cid = CLASS_COLORS[rgb]
                print(f"    RGB{rgb} → {label} (class {cid}): {count:,} points")
                obstacle_points += count
            else:
                print(f"    RGB{rgb} → BACKGROUND/OTHER: {count:,} points")

        print(f"\n  Total obstacle points: {obstacle_points:,} / {n_valid:,} ({100*obstacle_points/n_valid:.2f}%)")

    # 7. Convert one frame to Cartesian and check ranges
    if poses is not None and len(poses) > 0:
        first_pose = poses.iloc[0]
        frame_df = lidar_utils.filter_by_pose(df, first_pose)
        frame_df = frame_df[frame_df['distance_cm'] > 0]
        xyz = lidar_utils.spherical_to_local_cartesian(frame_df)

        print(f"\n  Cartesian ranges (frame 0, {len(xyz):,} valid points):")
        print(f"    X: [{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}] m")
        print(f"    Y: [{xyz[:,1].min():.1f}, {xyz[:,1].max():.1f}] m")
        print(f"    Z: [{xyz[:,2].min():.1f}, {xyz[:,2].max():.1f}] m")

    return len(poses) if poses is not None else 0


def main():
    h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])
    print(f"Found {len(h5_files)} HDF5 files in {DATA_DIR}")

    total_frames = 0
    for h5_file in h5_files:
        filepath = os.path.join(DATA_DIR, h5_file)
        n_frames = explore_file(filepath)
        total_frames += n_frames

    print(f"\n{'='*70}")
    print(f"TOTAL FRAMES ACROSS ALL FILES: {total_frames}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
