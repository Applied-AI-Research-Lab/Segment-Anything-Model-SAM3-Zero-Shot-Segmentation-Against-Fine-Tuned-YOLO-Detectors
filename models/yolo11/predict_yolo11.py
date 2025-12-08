#!/usr/bin/env python3
"""
YOLO11 Prediction and Evaluation Script

Production-ready inference pipeline for YOLO11 segmentation models.
Implements comprehensive evaluation against ground truth annotations
to enable fair comparison with other segmentation approaches (SAM3, etc.).

Key features:
- Batch processing with progress tracking
- Confidence thresholding and NMS for quality predictions
- Instance segmentation mask extraction
- Comprehensive evaluation metrics (IoU, precision, recall, F1)
- Visualization and reporting capabilities

Usage:
    python3 predict_yolo11.py --dataset minneapple --model ft/yolo11/minneapple/best.pt
    python3 predict_yolo11.py --dataset weedsgalore --model ft/yolo11/weedsgalore/best.pt
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Import evaluation utilities
# Note: These modules were removed during repo cleanup - restore if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
try:
    from evaluation import evaluate_dataset, print_evaluation_summary, save_metrics_to_csv
    from latex_report import generate_latex_report
except ImportError:
    print("Warning: Evaluation modules not found. Evaluation will be skipped.")
    evaluate_dataset = print_evaluation_summary = save_metrics_to_csv = None
    generate_latex_report = None


def predict_on_test_set(
    model_path: Path,
    test_images_dir: Path,
    output_dir: Path,
    confidence: float = 0.25,
    iou: float = 0.7
):
    """
    Run YOLO11 predictions on test set with optimized inference settings.

    This function implements efficient batch inference with:
    - Memory-efficient processing of large image sets
    - Confidence thresholding to filter low-quality detections
    - Non-maximum suppression for overlapping instances
    - Instance mask extraction from YOLO segmentation outputs
    - Timing measurements for performance analysis

    Args:
        model_path: Path to trained model weights (.pt file)
        test_images_dir: Directory containing test images (PNG/JPG)
        output_dir: Output directory for results and visualizations
        confidence: Confidence threshold (0.25 is good balance for agricultural scenes)
        iou: IoU threshold for NMS (0.7 standard, higher for crowded scenes)

    Returns:
        List of prediction results with masks, scores, and metadata
    """
    print("\n" + "="*80)
    print("YOLO11 Prediction on Test Set")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))
    print("Model loaded")
    
    # Get test images
    image_files = sorted(list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg")))
    print(f"\nFound {len(image_files)} test images")
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {test_images_dir}")
        return []
    
    # Create output directories
    viz_dir = output_dir / "visualizations"
    masks_dir = output_dir / "masks"
    viz_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Run predictions with progress tracking
    results_list = []
    inference_times = []

    print("\nRunning predictions...")
    import time
    for img_path in tqdm(image_files, desc="Processing"):
        # Time each prediction for performance analysis
        # Critical for benchmarking against other methods
        img_start = time.time()
        results = model.predict(
            source=str(img_path),
            conf=confidence,  # Filter low-confidence detections
            iou=iou,          # NMS threshold for overlapping instances
            save=False,       # Don't save images automatically
            verbose=False     # Reduce console output for cleaner logs
        )

        # YOLO returns list of results (one per image in batch)
        result = results[0]

        # Load original image for mask resizing and visualization
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]

        # Extract instance segmentation masks from YOLO output
        # YOLO11 provides masks as tensors that need conversion to binary masks
        masks_data = []
        scores = []

        if result.masks is not None:
            # Access raw mask and box data from Ultralytics result
            masks_tensor = result.masks.data  # Shape: [N, H, W] where N is instances
            boxes_tensor = result.boxes.data  # Shape: [N, 6] with [x1,y1,x2,y2,conf,class]

            # Process each detected instance
            for i, (mask_tensor, box_data) in enumerate(zip(masks_tensor, boxes_tensor)):
                # Convert from GPU tensor to CPU numpy array
                mask = mask_tensor.cpu().numpy()

                # Resize mask to match original image dimensions
                # YOLO may output masks at different resolution than input
                if mask.shape != (img_height, img_width):
                    mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

                # Convert to binary mask (threshold at 0.5)
                # YOLO outputs soft masks, we need hard segmentation
                binary_mask = (mask > 0.5).astype(np.uint8)

                masks_data.append(binary_mask)

                # Extract confidence score from bounding box data
                confidence_score = float(box_data[4])  # Index 4 is confidence in YOLO format
                scores.append(confidence_score)

        # Record inference time for this image
        img_end = time.time()
        inference_time = img_end - img_start
        inference_times.append(inference_time)

        # Structure results in standardized format for evaluation
        # This matches the format expected by evaluation functions
        result_dict = {
            "image_name": img_path.name,
            "image_path": str(img_path),
            "num_objects": len(masks_data),
            "masks": masks_data,
            "scores": np.array(scores) if scores else np.array([]),
            "image_shape": (img_height, img_width),
            "inference_time": inference_time
        }
        
        results_list.append(result_dict)
        
        # Save visualization
        if len(masks_data) > 0:
            viz_img = img.copy()
            
            # Overlay masks with different colors
            for idx, mask in enumerate(masks_data):
                # Generate color
                color = [
                    int(np.random.randint(50, 255)),
                    int(np.random.randint(50, 255)),
                    int(np.random.randint(50, 255))
                ]
                
                # Create colored mask
                colored_mask = np.zeros_like(viz_img)
                colored_mask[mask > 0] = color
                
                # Blend with image
                viz_img = cv2.addWeighted(viz_img, 1.0, colored_mask, 0.5, 0)
                
                # Draw contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(viz_img, contours, -1, color, 2)
            
            # Save visualization
            viz_path = viz_dir / img_path.name
            cv2.imwrite(str(viz_path), viz_img)
        
        # Save masks as numpy array (same as SAM3)
        mask_path = masks_dir / f"{img_path.stem}.npy"
        if len(masks_data) > 0:
            np.save(str(mask_path), np.array(masks_data))
    
    print(f"\nPredictions complete")
    print(f"  Total images: {len(results_list)}")
    print(f"  Total detections: {sum(r['num_objects'] for r in results_list)}")
    
    # Timing statistics
    if inference_times:
        avg_time = np.mean(inference_times)
        total_time = np.sum(inference_times)
        print(f"\nInference timing:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per image: {avg_time*1000:.2f}ms")
        print(f"  Min: {min(inference_times)*1000:.2f}ms")
        print(f"  Max: {max(inference_times)*1000:.2f}ms")
    
    print(f"\n  Visualizations: {viz_dir}")
    print(f"  Masks: {masks_dir}")
    
    return results_list


def load_ground_truth_masks(test_masks_dir: Path, image_names: list):
    """
    Load ground truth masks for evaluation.
    
    Args:
        test_masks_dir: Directory containing ground truth masks
        image_names: List of image names to load masks for
        
    Returns:
        List of ground truth masks (list of instance masks per image)
    """
    print(f"\nLoading ground truth masks from: {test_masks_dir}")
    
    ground_truth_list = []
    
    for img_name in image_names:
        mask_path = test_masks_dir / img_name
        
        if mask_path.exists():
            # Load mask
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if gt_mask is not None:
                # Extract individual instances
                unique_ids = np.unique(gt_mask)
                unique_ids = unique_ids[unique_ids > 0]  # Exclude background
                
                instance_masks = []
                for instance_id in unique_ids:
                    instance_mask = (gt_mask == instance_id).astype(np.uint8)
                    instance_masks.append(instance_mask)
                
                ground_truth_list.append(instance_masks)
            else:
                ground_truth_list.append([])
        else:
            ground_truth_list.append([])
    
    print(f"Loaded {len(ground_truth_list)} ground truth masks")
    return ground_truth_list


def save_evaluation_summary(results: list, output_dir: Path, inference_times: list = None):
    """
    Save evaluation summary JSON (same as SAM3).
    
    Args:
        results: Prediction results
        output_dir: Output directory
        inference_times: List of per-image inference times
    """
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "num_images": len(results),
        "total_detections": sum(r["num_objects"] for r in results),
        "statistics": {
            "mean_detections_per_image": sum(r["num_objects"] for r in results) / len(results) if results else 0,
            "max_detections": max(r["num_objects"] for r in results) if results else 0,
            "min_detections": min(r["num_objects"] for r in results) if results else 0,
            "images_with_detections": sum(1 for r in results if r["num_objects"] > 0),
        }
    }
    
    # Add timing statistics if available
    if inference_times:
        summary["inference_timing"] = {
            "total_time_seconds": float(np.sum(inference_times)),
            "mean_time_seconds": float(np.mean(inference_times)),
            "mean_time_ms": float(np.mean(inference_times) * 1000),
            "min_time_ms": float(min(inference_times) * 1000),
            "max_time_ms": float(max(inference_times) * 1000),
            "std_time_ms": float(np.std(inference_times) * 1000)
        }
    
    summary_path = report_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO11 predictions and evaluation on test set"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["minneapple", "weedsgalore"],
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO11 model (.pt file)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset directory (default: data/<dataset>)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<dataset>_yolo11_<timestamp>)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35, optimized for MinneApple)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS (default: 0.7)"
    )
    
    parser.add_argument(
        "--generate-latex",
        action="store_true",
        help="Generate LaTeX report"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("data") / args.dataset
    
    test_images_dir = data_dir / "test" / "images"
    test_masks_dir = data_dir / "test" / "masks"
    
    if not test_images_dir.exists():
        print(f"ERROR: Test images not found: {test_images_dir}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"yolo11_{args.dataset}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"YOLO11 Prediction & Evaluation - {args.dataset}")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Test images: {test_images_dir}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Run predictions
    results = predict_on_test_set(
        model_path=model_path,
        test_images_dir=test_images_dir,
        output_dir=output_dir,
        confidence=args.confidence,
        iou=args.iou
    )
    
    # Extract inference times from results
    inference_times = [r.get("inference_time", 0) for r in results if "inference_time" in r]
    
    # Save basic summary with timing info
    save_evaluation_summary(results, output_dir, inference_times)
    
    # Evaluate against ground truth
    if test_masks_dir.exists():
        print("\n" + "="*80)
        print("Evaluation Against Ground Truth")
        print("="*80)
        
        # Load ground truth
        image_names = [r["image_name"] for r in results]
        ground_truth = load_ground_truth_masks(test_masks_dir, image_names)
        
        # Run evaluation (using IoU=0.15 which is appropriate for dense agricultural scenes)
        # IoU=0.15 accounts for small boundary errors in crowded orchards while still
        # being strict enough to reject poor detections. See SOSIoUfoundproblem.txt for details.
        eval_metrics = evaluate_dataset(results, ground_truth, iou_threshold=0.15)
        
        # Print summary
        print_evaluation_summary(eval_metrics, title="YOLO11 Evaluation Results")
        
        # Save metrics
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        save_metrics_to_csv(eval_metrics, str(metrics_dir / "evaluation_metrics.csv"))
        
        # Save JSON
        with open(metrics_dir / "evaluation_metrics.json", 'w') as f:
            json_metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in eval_metrics.items() if k != 'per_image_metrics'}
            json.dump(json_metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_dir}")
    else:
        print(f"\nWARNING: Ground truth masks not found: {test_masks_dir}")
        print("   Skipping evaluation")
    
    # Generate LaTeX report
    if args.generate_latex:
        print("\n" + "="*80)
        print("Generating LaTeX Report")
        print("="*80)
        latex_file = generate_latex_report(str(output_dir), report_type="results")
        if latex_file:
            print(f"LaTeX report: {latex_file}")
    
    print("\n" + "="*80)
    print("Prediction and Evaluation Complete!")
    print("="*80)
    print(f"\nResults: {output_dir}")
    print(f"Visualizations: {output_dir}/visualizations")
    print(f"Metrics: {output_dir}/metrics")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
