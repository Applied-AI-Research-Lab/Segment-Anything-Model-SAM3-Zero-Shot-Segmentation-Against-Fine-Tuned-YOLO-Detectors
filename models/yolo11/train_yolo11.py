#!/usr/bin/env python3
"""
YOLO11 Training Script for Instance Segmentation

This script implements a complete training pipeline for YOLO11 segmentation models
on agricultural datasets. Key design decisions:

- Uses YOLO11's native polygon-based training instead of mask-based training
  for better performance and compatibility with the Ultralytics ecosystem
- Implements proper data validation to ensure training data integrity
- Follows best practices for hyperparameter tuning and model saving
- Supports multiple dataset formats (MinneApple, WeedsGalore)

Usage:
    python3 train_yolo11.py --dataset minneapple --epochs 100
    python3 train_yolo11.py --dataset weedsgalore --epochs 100
"""

import argparse
import yaml
from pathlib import Path
import shutil
from ultralytics import YOLO
import sys
import cv2
import numpy as np


def create_dataset_yaml(dataset_name: str, data_dir: Path, output_path: Path) -> Path:
    """
    Create YOLO format dataset configuration file.

    YOLO11 uses YAML configuration files to define dataset structure, classes,
    and paths. This approach provides flexibility for different dataset layouts
    while maintaining consistency across training runs.

    Args:
        dataset_name: Name of dataset (minneapple or weedsgalore)
        data_dir: Path to dataset root (contains train/val/test)
        output_path: Where to save the YAML file

    Returns:
        Path to created YAML file
    """
    # Dataset-specific class definitions
    # Single-class segmentation is common in agricultural applications
    # where we focus on detecting one type of object (apples or weeds)
    if dataset_name == "minneapple":
        class_names = ["apple"]
    elif dataset_name == "weedsgalore":
        class_names = ["weed"]
    else:
        # Fallback for unknown datasets - allows extensibility
        class_names = ["object"]

    # YOLO11 segmentation configuration
    # Key insight: YOLO11 can train directly on polygon annotations,
    # which are more efficient than pixel-wise masks for training
    # The 'path' field is absolute to avoid path resolution issues
    config = {
        "path": str(data_dir.absolute()),  # Dataset root - must be absolute path
        "train": "train/images",  # Relative paths from 'path'
        "val": "val/images",
        "test": "test/images",

        # Class configuration - critical for proper loss computation
        "nc": len(class_names),  # Number of classes
        "names": class_names     # Class names for logging/display
    }
    
    # YOLO11 segmentation automatically uses:
    # - train/labels/*.txt for training polygon annotations (YOLO format)
    # - val/labels/*.txt for validation annotations
    # - test/labels/*.txt for test annotations
    # Polygon format is preferred over masks for better training efficiency
    # and compatibility with the Ultralytics training pipeline
    
    # Save YAML file
    yaml_path = output_path / f"{dataset_name}_yolo11.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created dataset config: {yaml_path}")
    return yaml_path


def verify_masks_structure(data_dir: Path, dataset_name: str):
    """
    Verify that YOLO polygon labels exist for training.

    Critical validation step: YOLO11 requires polygon annotations in .txt format,
    not raw PNG masks. This function ensures data integrity before training begins,
    preventing runtime failures and ensuring reproducible results.

    Polygon format advantages:
    - More compact storage than pixel masks
    - Faster training due to reduced I/O
    - Better numerical stability in loss computation
    - Compatible with data augmentation pipelines

    Args:
        data_dir: Dataset root directory
        dataset_name: Name of dataset
    """
    print("\nVerifying YOLO polygon labels...")
    print("   (Instance masks must be converted to polygon format)")
    
    missing_labels = False
    
    # Validate each data split independently
    # This allows partial datasets (e.g., test-only evaluation)
    for split in ['train', 'val', 'test']:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        
        if not images_dir.exists():
            print(f"WARNING: No images found for {split} split")
            continue
        
        if not labels_dir.exists():
            print(f"ERROR: No labels directory found: {labels_dir}")
            missing_labels = True
            continue
        
        # Count files to detect mismatches early
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        label_files = list(labels_dir.glob("*.txt"))
        
        print(f"\n  {split.upper()} split:")
        print(f"    Images: {len(image_files)}")
        print(f"    Labels: {len(label_files)}")
        
        if len(label_files) == 0:
            print(f"    ERROR: No label files found!")
            missing_labels = True
        elif len(label_files) != len(image_files):
            print(f"    WARNING: Image/label count mismatch!")
            # Don't fail here - allow partial datasets for flexibility
        else:
            # Detailed validation: count total instances
            # This helps estimate training complexity and memory requirements
            total_instances = 0
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    total_instances += len(lines)
                    # Could add validation for malformed polygons here
            print(f"    Verified: {len(label_files)} label files")
            print(f"    Total instances: {total_instances}")    # Fail fast if training data is missing
    # This prevents wasted compute time on incomplete datasets
    if missing_labels:
        print("\n" + "="*80)
        print("ERROR: YOLO polygon labels not found!")
        print("="*80)
        print("\nYou must convert instance masks to YOLO polygon format first:")
        print(f"  python3 models/yolo11/convert_masks_to_yolo.py --dataset {dataset_name}")
        print("\nOr run the conversion script:")
        print(f"  ./convert_masks.sh {dataset_name}")
        print("="*80)
        raise RuntimeError("YOLO labels not found - run conversion first")


def train_yolo11(
    dataset_yaml: Path,
    dataset_name: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    model_dir: Path = None,
    model_variant: str = "n"
):
    """
    Train YOLO11 segmentation model with optimized hyperparameters.

    This function implements a production-ready training pipeline with:
    - Pre-trained COCO weights for transfer learning
    - Aggressive data augmentation to combat overfitting on small datasets
    - Early stopping and checkpointing for efficient training
    - Comprehensive logging and visualization

    Key training decisions:
    - Lower learning rate (0.002) for small agricultural datasets
    - Strong augmentation (mosaic, copy-paste, mixup) for generalization
    - Early stopping patience of 50 epochs to prevent overfitting
    - Regular checkpointing every 10 epochs for recovery

    Args:
        dataset_yaml: Path to dataset YAML config
        dataset_name: Name of dataset
        epochs: Number of training epochs (early stopping may terminate earlier)
        batch_size: Global batch size (distributed across GPUs if multiple)
        imgsz: Input image size (640 is good balance of speed vs accuracy)
        device: Device to train on (0 for GPU, cpu for CPU, 0,1 for multi-GPU)
        model_dir: Directory to save trained model
        model_variant: Model variant (n/s/m/l/x) - nano for speed, xlarge for accuracy

    Returns:
        Path to trained model weights (best.pt)
    """
    print("\n" + "="*80)
    print("Training YOLO11 Segmentation Model")
    print("="*80)

    # Model selection based on computational constraints
    # Nano variant: fast inference, good for edge deployment
    # Medium/Large: better accuracy for complex scenes
    model_name = f'yolo11{model_variant}-seg.pt'
    variant_names = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "xlarge"}
    print(f"\nLoading YOLO11{model_variant}-seg ({variant_names.get(model_variant, model_variant)}) model (pre-trained on COCO)...")
    model = YOLO(model_name)
    print("Model loaded")
    
    # Training configuration display
    print("\nStarting training...")
    print(f"  Model: YOLO11{model_variant}-seg ({variant_names.get(model_variant, model_variant)})")
    print(f"  Dataset: {dataset_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} (global - distributed across GPUs)")
    print(f"  Image size: {imgsz}")
    print(f"  Device: {device}")
    if ',' in device:
        gpu_count = len(device.split(','))
        per_gpu_batch = batch_size // gpu_count
        print(f"  Per-GPU batch: {per_gpu_batch} ({gpu_count} GPUs)")
    print()    import time
    training_start = time.time()

    # Advanced augmentation strategy for agricultural segmentation
    # These parameters were tuned to address overfitting on small datasets
    # and improve performance on variable field conditions
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(model_dir / "training"),
        name=dataset_name,
        exist_ok=True,
        patience=50,  # Early stopping - prevents overfitting on small datasets
        save=True,
        save_period=10,  # Checkpoint every 10 epochs for recovery
        val=True,  # Validate during training to monitor generalization
        plots=True,  # Generate training curves and validation plots
        verbose=True,
        # Optimized hyperparameters for small agricultural datasets
        lr0=0.002,  # Conservative learning rate to prevent divergence
        mosaic=0.5,  # Composite images from 4 sources - improves context learning
        copy_paste=0.5,  # Instance-level augmentation - better for object detection
        mixup=0.3,  # Blend images with labels - reduces overfitting
        degrees=15,  # Rotation augmentation - handles different camera angles
        hsv_h=0.015,  # Color jittering for lighting invariance
        hsv_s=0.7,   # Saturation variation
        hsv_v=0.4,   # Brightness variation
    )

    training_end = time.time()
    training_time = training_end - training_start

    print("\nTraining completed!")
    print(f"Total training time: {training_time/3600:.2f} hours ({training_time/60:.2f} minutes)")
    
    # Get path to best model
    best_model = model_dir / "training" / dataset_name / "weights" / "best.pt"
    last_model = model_dir / "training" / dataset_name / "weights" / "last.pt"
    
    # Copy best model to model directory
    final_model = model_dir / "best.pt"
    if best_model.exists():
        shutil.copy(best_model, final_model)
        print(f"\nBest model saved to: {final_model}")
    
    # Also save last model
    final_last = model_dir / "last.pt"
    if last_model.exists():
        shutil.copy(last_model, final_last)
        print(f"Last model saved to: {final_last}")
    
    # Save training metadata (including timing)
    import json
    metadata = {
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "device": device,
        "training_time_seconds": training_time,
        "training_time_hours": training_time / 3600,
        "training_time_minutes": training_time / 60,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add final metrics if available
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        metadata["final_mAP50"] = float(metrics.get('metrics/mAP50(B)', 0))
        metadata["final_mAP50_95"] = float(metrics.get('metrics/mAP50-95(B)', 0))
    
    metadata_file = model_dir / "training_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Training metadata saved to: {metadata_file}")
    
    # Print training summary
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    
    # Get metrics from results
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"Final mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Final mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    
    print(f"\nTraining logs: {model_dir / 'training' / dataset_name}")
    print("="*80)
    
    return final_model


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 segmentation model on MinneApple or WeedsGalore"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["minneapple", "weedsgalore"],
        help="Dataset to train on"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset directory (default: data/<dataset>)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to train on: 0 for GPU, cpu for CPU (default: 0)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="ft/yolo11",
        help="Directory to save trained models (default: ft/yolo11)"
    )
    
    parser.add_argument(
        "--model-variant",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO11 model variant: n(nano), s(small), m(medium), l(large), x(xlarge) (default: n)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("data") / args.dataset
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify dataset exists
    if not data_dir.exists():
        print(f"ERROR: Dataset not found: {data_dir}")
        print(f"   Please run: python3 download_dataset.py --dataset {args.dataset}")
        sys.exit(1)
    
    # Verify train/val splits exist
    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        print(f"ERROR: Train/val splits not found in {data_dir}")
        print(f"   Please ensure dataset has train/ and val/ directories")
        sys.exit(1)
    
    print("="*80)
    print(f"YOLO11 Training - {args.dataset}")
    print("="*80)
    print(f"Dataset: {data_dir}")
    print(f"Model output: {model_dir}")
    print("="*80)
    
    # Verify PNG masks are ready (no conversion needed!)
    verify_masks_structure(data_dir, args.dataset)
    
    # Create dataset YAML
    dataset_yaml = create_dataset_yaml(args.dataset, data_dir, model_dir)
    
    # Train model
    trained_model = train_yolo11(
        dataset_yaml=dataset_yaml,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        model_dir=model_dir,
        model_variant=args.model_variant
    )
    
    print("\nTraining complete!")
    print(f"Best model: {trained_model}")
    print(f"Training logs: {model_dir / 'training' / args.dataset}")


if __name__ == "__main__":
    main()
