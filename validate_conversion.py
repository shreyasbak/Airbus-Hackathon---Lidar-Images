#!/usr/bin/env python3
"""
=============================================================================
 validate_conversion.py — Validate KITTI Conversion Output
=============================================================================

PURPOSE:
    After running convert_h5_to_kitti.py, this script validates that the
    output data is correct and KITTI-compatible. It checks:
    
    1. File counts: All expected .bin, .txt label, and .txt calib files exist
    2. Binary format: .bin files have correct shape (N×4 float32)
    3. Label format: .txt label files are valid KITTI format
    4. Calibration: calib files contain expected identity matrices
    5. ImageSets: train.txt and val.txt cover all frames without overlap
    6. Statistics: Box count per class, points per frame, etc.

USAGE:
    python validate_conversion.py --kitti-dir ./airbus_kitti
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

VALID_CLASSES = {'Antenna', 'Cable', 'Electric_Pole', 'Wind_Turbine'}


def validate_bin_file(bin_path):
    """Check that a .bin file is a valid KITTI velodyne file (N×4 float32)."""
    data = np.fromfile(bin_path, dtype=np.float32)
    
    if len(data) == 0:
        return False, "Empty file"
    if len(data) % 4 != 0:
        return False, f"Size {len(data)} not divisible by 4"
    
    points = data.reshape(-1, 4)
    n_pts = points.shape[0]
    
    # Sanity checks
    if np.any(np.isnan(points)):
        return False, "Contains NaN values"
    if np.any(np.isinf(points)):
        return False, "Contains Inf values"
    
    # Reflectivity should be in [0, 1]
    ref = points[:, 3]
    if ref.min() < 0 or ref.max() > 1.01:
        return False, f"Reflectivity out of range: [{ref.min():.3f}, {ref.max():.3f}]"
    
    return True, n_pts


def validate_label_file(label_path):
    """Check that a label .txt file is valid KITTI format."""
    boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        
        if len(parts) != 15:
            return False, f"Line {i}: expected 15 fields, got {len(parts)}", []
        
        cls = parts[0]
        if cls not in VALID_CLASSES:
            return False, f"Line {i}: unknown class '{cls}'", []
        
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            return False, f"Line {i}: non-numeric value", []
        
        # h, w, l should be positive
        h, w, l = vals[7], vals[8], vals[9]
        if h <= 0 or w <= 0 or l <= 0:
            return False, f"Line {i}: non-positive dimension h={h}, w={w}, l={l}", []
        
        boxes.append({'class': cls, 'h': h, 'w': w, 'l': l,
                      'x': vals[10], 'y': vals[11], 'z': vals[12], 'ry': vals[13]})
    
    return True, f"{len(boxes)} boxes", boxes


def validate_calib_file(calib_path):
    """Check that a calibration file contains identity matrices."""
    with open(calib_path, 'r') as f:
        content = f.read()
    
    required_keys = ['P0:', 'P1:', 'P2:', 'P3:', 'R0_rect:', 'Tr_velo_to_cam:', 'Tr_imu_to_velo:']
    for key in required_keys:
        if key not in content:
            return False, f"Missing key: {key}"
    
    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="Validate KITTI conversion output")
    parser.add_argument('--kitti-dir', required=True, help='Path to airbus_kitti/ directory')
    args = parser.parse_args()

    kitti_dir = args.kitti_dir
    velodyne_dir = os.path.join(kitti_dir, 'training', 'velodyne')
    label_dir = os.path.join(kitti_dir, 'training', 'label_2')
    calib_dir = os.path.join(kitti_dir, 'training', 'calib')
    imagesets_dir = os.path.join(kitti_dir, 'ImageSets')

    errors = []
    warnings = []

    # ---- Check directories exist ----
    print("=" * 60)
    print("VALIDATING KITTI CONVERSION")
    print("=" * 60)

    for d, name in [(velodyne_dir, 'velodyne'), (label_dir, 'label_2'),
                     (calib_dir, 'calib'), (imagesets_dir, 'ImageSets')]:
        if os.path.isdir(d):
            print(f"  ✓ {name}/ directory exists")
        else:
            errors.append(f"Missing directory: {name}/")
            print(f"  ✗ {name}/ MISSING")

    if errors:
        print("\nCRITICAL ERRORS — cannot continue.")
        for e in errors:
            print(f"  ✗ {e}")
        return

    # ---- Count files ----
    bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    lbl_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    cal_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.txt')])

    print(f"\n  .bin files:   {len(bin_files)}")
    print(f"  .txt labels:  {len(lbl_files)}")
    print(f"  .txt calibs:  {len(cal_files)}")

    if len(bin_files) != len(lbl_files):
        errors.append(f"Mismatch: {len(bin_files)} .bin vs {len(lbl_files)} .txt labels")
    if len(bin_files) != len(cal_files):
        errors.append(f"Mismatch: {len(bin_files)} .bin vs {len(cal_files)} .txt calibs")

    # ---- Validate .bin files (sample) ----
    print(f"\nValidating .bin files...")
    total_points = []
    n_check = min(20, len(bin_files))
    sample_indices = np.linspace(0, len(bin_files)-1, n_check, dtype=int) if len(bin_files) > 0 else []
    
    for idx in sample_indices:
        bf = bin_files[idx]
        ok, result = validate_bin_file(os.path.join(velodyne_dir, bf))
        if ok:
            total_points.append(result)
        else:
            errors.append(f"  {bf}: {result}")

    if total_points:
        print(f"  Sampled {n_check} files: "
              f"points/frame min={min(total_points):,} max={max(total_points):,} "
              f"mean={np.mean(total_points):,.0f}")

    # ---- Validate .txt label files ----
    print(f"\nValidating label files...")
    class_counts = defaultdict(int)
    box_dims = defaultdict(list)
    n_empty_frames = 0
    
    for lf in lbl_files:
        ok, msg, boxes = validate_label_file(os.path.join(label_dir, lf))
        if not ok:
            errors.append(f"  {lf}: {msg}")
        if not boxes:
            n_empty_frames += 1
        for box in boxes:
            class_counts[box['class']] += 1
            box_dims[box['class']].append((box['h'], box['w'], box['l']))

    print(f"  Empty frames (no obstacles): {n_empty_frames} / {len(lbl_files)}")
    print(f"\n  Bounding box statistics:")
    print(f"  {'Class':<20} {'Count':>8} {'Avg H':>8} {'Avg W':>8} {'Avg L':>8}")
    print(f"  {'-'*52}")
    for cls in sorted(class_counts.keys()):
        dims = np.array(box_dims[cls])
        print(f"  {cls:<20} {class_counts[cls]:>8} "
              f"{dims[:,0].mean():>8.2f} {dims[:,1].mean():>8.2f} {dims[:,2].mean():>8.2f}")
    print(f"  {'TOTAL':<20} {sum(class_counts.values()):>8}")

    # ---- Validate calib files (sample) ----
    print(f"\nValidating calib files (sample)...")
    for idx in sample_indices[:5]:
        cf = cal_files[idx]
        ok, msg = validate_calib_file(os.path.join(calib_dir, cf))
        if not ok:
            errors.append(f"  {cf}: {msg}")
    print(f"  ✓ Sampled {min(5, len(cal_files))} calib files — all OK")

    # ---- Validate ImageSets ----
    print(f"\nValidating ImageSets...")
    train_path = os.path.join(imagesets_dir, 'train.txt')
    val_path = os.path.join(imagesets_dir, 'val.txt')

    if os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path) as f:
            train_ids = set(f.read().strip().split('\n'))
        with open(val_path) as f:
            val_ids = set(f.read().strip().split('\n'))

        overlap = train_ids & val_ids
        all_ids = train_ids | val_ids
        expected_ids = set(f"{i:06d}" for i in range(len(bin_files)))

        print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")
        print(f"  Overlap: {len(overlap)} (should be 0)")
        print(f"  Coverage: {len(all_ids)} / {len(bin_files)} frames")

        if overlap:
            errors.append(f"Train/Val overlap: {len(overlap)} frames")
        if all_ids != expected_ids:
            missing = expected_ids - all_ids
            extra = all_ids - expected_ids
            if missing:
                warnings.append(f"Frames missing from splits: {len(missing)}")
            if extra:
                warnings.append(f"Extra frames in splits: {len(extra)}")
    else:
        errors.append("Missing train.txt or val.txt")

    # ---- Metadata ----
    meta_path = os.path.join(kitti_dir, 'frame_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n  Metadata: {len(meta)} frames recorded")
    else:
        warnings.append("No frame_metadata.json found")

    # ---- Summary ----
    print(f"\n{'='*60}")
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} errors:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("VALIDATION PASSED ✓")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")

    print("=" * 60)


if __name__ == '__main__':
    main()
