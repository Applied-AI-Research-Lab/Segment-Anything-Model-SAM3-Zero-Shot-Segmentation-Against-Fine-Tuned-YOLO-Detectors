#!/usr/bin/env python3
"""
Convert Instance Segmentation Masks to YOLO Polygon Format

Critical preprocessing step for YOLO11 training. Converts instance segmentation
masks (where each object has a unique pixel value) to YOLO's polygon annotation
format required for training.

Key technical details:
- Instance masks: pixels have values 1, 2, 3, ... for each object
- YOLO format: "class_id x1 y1 x2 y2 ..." (normalized 0-1 coordinates)
- Each instance becomes a separate annotation line in .txt files
- Polygon simplification reduces file size while preserving shape accuracy

This conversion is essential because:
1. YOLO11 trains on polygon annotations, not pixel masks
2. Polygons are more compact and efficient than full masks
3. Enables data augmentation and faster training convergence
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def mask_to_polygon(binary_mask: np.ndarray, epsilon_factor: float = 0.001) -> list:
    """
    Convert binary instance mask to normalized polygon coordinates.

    This function implements robust polygon extraction with:
    - Contour finding using OpenCV's optimized algorithm
    - Polygon simplification to reduce annotation file size
    - Coordinate normalization for YOLO's 0-1 coordinate system
    - Minimum vertex filtering to ensure valid polygons

    Args:
        binary_mask: Binary mask (0=background, 1=object) for single instance
        epsilon_factor: Douglas-Peucker simplification factor. Higher values
                       create simpler polygons (fewer vertices) but may lose
                       detail. 0.001 is good balance for most applications.

    Returns:
        List of normalized coordinates [x1, y1, x2, y2, ...] or empty list
        if polygon extraction fails or results in too few vertices.
    """
    # Extract object contours using OpenCV
    # RETR_EXTERNAL ensures only outer contours (no holes)
    # CHAIN_APPROX_SIMPLE reduces vertices while preserving shape
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return []  # No object found in mask

    # Select largest contour in case of multiple disconnected components
    # This handles cases where an object might be split by preprocessing
    contour = max(contours, key=cv2.contourArea)

    # Simplify polygon using Douglas-Peucker algorithm
    # Reduces vertex count while maintaining shape within epsilon tolerance
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Convert to YOLO's normalized coordinate format
    h, w = binary_mask.shape
    polygon = []
    for point in approx:
        x, y = point[0]
        # Normalize coordinates to [0, 1] range as required by YOLO
        polygon.extend([x / w, y / h])

    # Filter out degenerate polygons (need at least 3 points = 6 coordinates)
    return polygon if len(polygon) >= 6 else []


def convert_instance_mask_to_yolo(
    mask_path: Path,
    output_path: Path,
    class_id: int = 0
):
    """
    Convert instance segmentation mask to YOLO polygon format.
    
    Args:
        mask_path: Path to instance mask PNG
        output_path: Path to save .txt labels
        class_id: Class ID for all instances (default: 0 for single-class)
    
    Returns:
        Number of instances converted
    """
    # Read instance mask
    instance_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if instance_mask is None:
        print(f"Warning: Could not read {mask_path}")
        return 0
    
    # Get unique instance IDs (excluding background=0)
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    # Convert each instance to polygon
    labels = []
    for instance_id in instance_ids:
        # Create binary mask for this instance
        binary_mask = (instance_mask == instance_id).astype(np.uint8)
        
        # Convert to polygon
        polygon = mask_to_polygon(binary_mask)
        
        # Skip if polygon is invalid
        if len(polygon) < 6:  # Need at least 3 points
            continue
        
        # Format: class_id x1 y1 x2 y2 ...
        label_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in polygon)
        labels.append(label_line)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(labels))
    
    return len(labels)


def convert_dataset_split(
    masks_dir: Path,
    labels_dir: Path,
    class_id: int = 0
):
    """
    Convert all instance masks in a directory to YOLO format.
    
    Args:
        masks_dir: Directory containing instance mask PNGs
        labels_dir: Directory to save .txt label files
        class_id: Class ID for all instances
    
    Returns:
        Tuple of (num_images, num_instances)
    """
    masks_dir = Path(masks_dir)
    labels_dir = Path(labels_dir)
    
    # Get all mask files
    mask_files = sorted(masks_dir.glob("*.png"))
    
    print(f"\nConverting {len(mask_files)} instance masks to YOLO polygons...")
    total_instances = 0
    
    for mask_path in tqdm(mask_files, desc=f"Processing {masks_dir.name}"):
        # Create output path with same name but .txt extension
        label_path = labels_dir / mask_path.with_suffix('.txt').name
        
        # Convert mask to YOLO format
        num_instances = convert_instance_mask_to_yolo(mask_path, label_path, class_id)
        total_instances += num_instances
    
    print(f"Converted {len(mask_files)} images with {total_instances} total instances")
    print(f"Labels saved to: {labels_dir}")
    
    return len(mask_files), total_instances


def main():
    parser = argparse.ArgumentParser(
        description="Convert instance segmentation masks to YOLO polygon format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., minneapple, weedsgalore)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data/)"
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class ID for instances (default: 0)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to convert (default: train val test)"
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_dir) / args.dataset
    
    print("=" * 80)
    print("Converting Instance Masks to YOLO Polygon Format")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {data_root}")
    print(f"Class ID: {args.class_id}")
    print(f"Splits: {', '.join(args.splits)}")
    print()
    
    total_images = 0
    total_instances = 0
    
    # Convert each split
    for split in args.splits:
        masks_dir = data_root / split / "masks"
        labels_dir = data_root / split / "labels"
        
        if not masks_dir.exists():
            print(f"\nWARNING: Skipping {split}: {masks_dir} not found")
            continue
        
        num_images, num_instances = convert_dataset_split(
            masks_dir, labels_dir, args.class_id
        )
        total_images += num_images
        total_instances += num_instances
    
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"Total images: {total_images}")
    print(f"Total instances: {total_instances}")
    print(f"Average instances per image: {total_instances / total_images:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
