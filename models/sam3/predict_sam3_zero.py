"""
SAM3 Zero-Shot Segmentation Pipeline

Complete end-to-end workflow for zero-shot segmentation using Meta's SAM3 model.
Implements a production-ready pipeline with:

1. Intelligent data loading and preprocessing
2. Zero-shot segmentation using text prompts
3. Advanced prompting strategies for improved results
4. Comprehensive evaluation against ground truth
5. Rich visualization and reporting capabilities
6. Batch processing for scalability

Key architectural decisions:
- Zero-shot approach eliminates need for task-specific training
- Text prompts provide semantic guidance for segmentation
- Batch processing enables efficient handling of large datasets
- Modular design allows easy extension and customization

This pipeline serves as a strong baseline for segmentation tasks and
demonstrates SAM3's capabilities in agricultural applications.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import numpy as np

# Import pipeline components
# Note: These modules were removed during repo cleanup - restore if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from data_loader import MinneAppleDataset, download_sample_orchard_images
    from segmentation import SAM3Segmenter, segment_image
    from batch_processing import BatchProcessor, batch_segment_images
    from evaluation import evaluate_dataset, print_evaluation_summary, save_metrics_to_csv
    from visualization import visualize_segmentation
    from advanced_prompting import run_advanced_prompting_examples
    from latex_report import generate_latex_report
except ImportError:
    print("Warning: Pipeline modules not found. Some features will be unavailable.")
    # Define dummy functions to prevent crashes
    MinneAppleDataset = download_sample_orchard_images = None
    SAM3Segmenter = segment_image = None
    BatchProcessor = batch_segment_images = None
    evaluate_dataset = print_evaluation_summary = save_metrics_to_csv = None
    visualize_segmentation = None
    run_advanced_prompting_examples = None
    generate_latex_report = None


def setup_workflow(data_dir: str = "data/minneapple"):
    """
    Initialize and validate the segmentation workflow.

    This function handles data discovery and preparation with:
    - Flexible dataset structure detection (supports multiple layouts)
    - Automatic sample data generation for testing
    - Dataset integrity validation
    - Graceful fallback for missing components

    The design supports both:
    - Standard split directories (train/val/test with images/ subdirs)
    - Legacy flat directory structures
    - Automatic sample data creation for development

    Args:
        data_dir: Root directory for dataset. Supports multiple structures:
                 - Standard: data/minneapple/test/ (with images/, masks/)
                 - Legacy: data/minneapple/ (flat structure)
                 - Auto-generated: Creates sample data if none exists
    """
    print("\n" + "="*80)
    print("PHASE 1: Setup and Data Preparation")
    print("="*80)

    data_path = Path(data_dir)

    # Flexible dataset structure detection
    # Supports both new split-based and legacy flat structures
    images_dir = data_path / "images"

    if not images_dir.exists():
        # Fallback to legacy structure or trigger sample generation
        print(f"WARNING: No 'images' subdirectory found in {data_dir}")
        print(f"   Trying legacy structure...")
        dataset = MinneAppleDataset(data_dir=data_dir)
    else:
        # New structure: data_dir already points to split (e.g., data/minneapple/test)
        dataset = MinneAppleDataset(data_dir=data_dir)
    
    # Check if data exists
    images = dataset.list_images()
    
    if len(images) == 0:
        print("\nNo images found. Creating sample data...")
        dataset.create_sample_data(num_samples=5)
        download_sample_orchard_images(save_dir=str(dataset.images_dir))
        images = dataset.list_images()
    
    # Display dataset info
    info = dataset.get_dataset_info()
    print("\nDataset Information:")
    print(f"  Location: {info['images_dir']}")
    print(f"  Number of images: {info['num_images']}")
    print(f"  Sample images: {', '.join(info['sample_images'][:3])}")
    
    return dataset


def run_segmentation(
    dataset: MinneAppleDataset,
    text_prompt: str = "apple",
    max_images: int = None,
    output_dir: str = "results"
):
    """
    Execute zero-shot segmentation using SAM3 with text prompts.

    This function orchestrates the core segmentation workflow:
    - Initializes batch processing pipeline for efficiency
    - Applies text-guided segmentation to all images
    - Generates visualizations and saves segmentation masks
    - Returns structured results for downstream evaluation

    The zero-shot approach leverages SAM3's pre-trained knowledge to
    segment objects based on natural language descriptions, eliminating
    the need for task-specific training data.

    Args:
        dataset: MinneAppleDataset instance with loaded images
        text_prompt: Natural language description of target objects
                    (e.g., "apple", "weed", "fruit"). More specific prompts
                    generally yield better results.
        max_images: Optional limit on number of images to process.
                   Useful for testing pipeline on subset of data.
        output_dir: Directory for saving results, masks, and visualizations

    Returns:
        List of segmentation results with masks, confidence scores,
        and metadata for each processed image
    """
    print("\n" + "="*80)
    print("PHASE 2: SAM3 Zero-Shot Segmentation")
    print("="*80)

    # Initialize batch processing for efficient pipeline execution
    # BatchProcessor handles memory management, progress tracking,
    # and parallel processing when available
    processor = BatchProcessor(output_dir=output_dir)

    # Execute segmentation with text guidance
    # This leverages SAM3's zero-shot capabilities - no training required
    results = processor.process_directory(
        image_dir=dataset.images_dir,
        text_prompt=text_prompt,      # Semantic guidance for segmentation
        visualize=True,               # Generate visual overlays
        save_masks=True,              # Save binary masks for evaluation
        max_images=max_images         # Optional processing limit
    )

    return results


def generate_evaluation_report(
    results: list,
    output_dir: str = "results"
):
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: List of segmentation results
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("PHASE 3: Evaluation Report")
    print("="*80)
    
    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate summary statistics
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_images": len(results),
        "total_detections": sum(r["num_objects"] for r in results),
        "statistics": {
            "mean_detections_per_image": sum(r["num_objects"] for r in results) / len(results) if results else 0,
            "max_detections": max(r["num_objects"] for r in results) if results else 0,
            "min_detections": min(r["num_objects"] for r in results) if results else 0,
            "images_with_detections": sum(1 for r in results if r["num_objects"] > 0),
        },
        "per_image_results": []
    }
    
    for r in results:
        summary["per_image_results"].append({
            "image_name": r.get("image_name", "unknown"),
            "num_objects": r["num_objects"],
            "mean_score": float(r["scores"].mean()) if len(r["scores"]) > 0 else 0.0,
            "max_score": float(r["scores"].max()) if len(r["scores"]) > 0 else 0.0,
        })
    
    # Save summary
    summary_path = report_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate HTML report
    html_report = generate_html_report(summary, results, output_dir)
    html_path = report_dir / "report.html"
    with open(html_path, 'w') as f:
        f.write(html_report)
    
    print(f"\nEvaluation report saved to: {report_dir}")
    print(f"  - evaluation_summary.json")
    print(f"  - report.html")
    
    # Print summary to console
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nTotal Images: {summary['num_images']}")
    print(f"Total Detections: {summary['total_detections']}")
    print(f"Mean Detections/Image: {summary['statistics']['mean_detections_per_image']:.2f}")
    print(f"Images with Detections: {summary['statistics']['images_with_detections']}")
    print("="*80)
    
    return summary


def generate_html_report(summary: dict, results: list, output_dir: str) -> str:
    """Generate HTML evaluation report."""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SAM3 Segmentation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .stat-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .stat-value {{
                font-size: 36px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .stat-label {{
                font-size: 14px;
                opacity: 0.9;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .gallery {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .gallery-item {{
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .gallery-item img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .gallery-item-info {{
                padding: 10px;
                background-color: #f9f9f9;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SAM3 Segmentation Report</h1>
            <p class="timestamp">Generated: {summary['timestamp']}</p>
            
            <h2>Summary Statistics</h2>
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Images</div>
                    <div class="stat-value">{summary['num_images']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Detections</div>
                    <div class="stat-value">{summary['total_detections']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Mean Detections/Image</div>
                    <div class="stat-value">{summary['statistics']['mean_detections_per_image']:.1f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Images with Detections</div>
                    <div class="stat-value">{summary['statistics']['images_with_detections']}</div>
                </div>
            </div>
            
            <h2>📋 Per-Image Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Image</th>
                        <th>Objects Detected</th>
                        <th>Mean Score</th>
                        <th>Max Score</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for item in summary['per_image_results']:
        html += f"""
                    <tr>
                        <td>{item['image_name']}</td>
                        <td>{item['num_objects']}</td>
                        <td>{item['mean_score']:.3f}</td>
                        <td>{item['max_score']:.3f}</td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
            
            <h2>Visualizations</h2>
            <p>Segmentation visualizations are saved in the <code>visualizations/</code> directory.</p>
            
            <h2>Output Files</h2>
            <ul>
                <li><strong>visualizations/</strong> - Segmentation visualizations</li>
                <li><strong>masks/</strong> - Binary mask arrays (NumPy format)</li>
                <li><strong>metrics/</strong> - Evaluation metrics (CSV/JSON)</li>
                <li><strong>report/</strong> - This report and summary files</li>
            </ul>
            
            <h2>About SAM3</h2>
            <p>This report was generated using <strong>SAM3 (Segment Anything Model 3)</strong> from Meta AI. 
            SAM3 is a unified foundation model for promptable segmentation that can detect and segment objects 
            using text prompts or visual cues.</p>
            
            <p><strong>Key Features:</strong></p>
            <ul>
                <li>Open-vocabulary segmentation (270K+ concepts)</li>
                <li>Text-based and visual prompting</li>
                <li>Instance and semantic segmentation</li>
                <li>75-80% human-level performance</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html


def main():
    """Main workflow execution."""
    parser = argparse.ArgumentParser(
        description="SAM3 Segmentation Workflow for MinneApple and WeedsGalore Datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/minneapple",
        help="Directory for dataset split (e.g., data/minneapple/test or data/weedsgalore/test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for results"
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="apple",
        help="Text prompt for segmentation"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip data setup phase"
    )
    parser.add_argument(
        "--advanced-prompting",
        action="store_true",
        help="Run advanced prompting examples"
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Process all images in dataset (ignores --max-images)"
    )
    parser.add_argument(
        "--comprehensive-evaluation",
        action="store_true",
        help="Perform comprehensive evaluation with ground truth if available"
    )
    parser.add_argument(
        "--generate-latex",
        action="store_true",
        help="Generate LaTeX report for academic papers"
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Directory containing ground truth annotations for evaluation"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SAM3 SEGMENTATION WORKFLOW")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Text prompt: '{args.text_prompt}'")
    
    # Handle full dataset flag
    if args.full_dataset:
        print(f"  Mode: FULL DATASET (all images)")
        args.max_images = None  # Override max_images
    else:
        print(f"  Max images: {args.max_images or 'All'}")
    
    if args.comprehensive_evaluation:
        print(f"  Evaluation: COMPREHENSIVE (with ground truth)")
    if args.generate_latex:
        print(f"  LaTeX Report: ENABLED")
    if args.gt_dir:
        print(f"  Ground Truth: {args.gt_dir}")
    
    print("="*80)
    
    try:
        # Phase 1: Setup
        if not args.skip_setup:
            dataset = setup_workflow(data_dir=args.data_dir)
        else:
            dataset = MinneAppleDataset(data_dir=args.data_dir)
        
        # Phase 2: Segmentation
        results = run_segmentation(
            dataset=dataset,
            text_prompt=args.text_prompt,
            max_images=args.max_images,
            output_dir=args.output_dir
        )
        
        # Phase 3: Report Generation
        summary = generate_evaluation_report(
            results=results,
            output_dir=args.output_dir
        )
        
        # Phase 3.1: Comprehensive Evaluation with Ground Truth (if requested)
        if args.comprehensive_evaluation:
            print("\n" + "="*80)
            print("PHASE 3.1: Comprehensive Evaluation with Ground Truth")
            print("="*80)
            
            # Determine ground truth directory
            # New structure: data_dir already points to split (e.g., data/minneapple/test)
            # So masks are in data_dir/masks/
            if args.gt_dir:
                gt_dir = args.gt_dir
            else:
                # Check if data_dir/masks exists (new structure)
                gt_dir_new = Path(args.data_dir) / "masks"
                if gt_dir_new.exists():
                    gt_dir = str(gt_dir_new)
                else:
                    # Fallback for old structure
                    gt_dir = str(Path(args.data_dir) / "masks")
            
            if Path(gt_dir).exists():
                print(f"Loading ground truth from: {gt_dir}")
                
                # Load ground truth masks
                ground_truth_list = []
                annotations_dir = Path(gt_dir)
                
                for result in results:
                    image_name = result.get("image_name", "unknown")
                    # Look for corresponding annotation file
                    # MinneApple uses .png masks with same name
                    mask_file = annotations_dir / image_name
                    
                    if mask_file.exists():
                        import cv2
                        gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if gt_mask is not None:
                            # MinneApple masks are instance segmentation masks
                            # Each unique non-zero value represents a different apple
                            unique_ids = np.unique(gt_mask)
                            unique_ids = unique_ids[unique_ids > 0]  # Exclude background (0)
                            
                            # Create individual binary masks for each instance
                            instance_masks = []
                            for instance_id in unique_ids:
                                instance_mask = (gt_mask == instance_id).astype(np.uint8)
                                instance_masks.append(instance_mask)
                            
                            ground_truth_list.append(instance_masks)
                        else:
                            ground_truth_list.append([])
                    else:
                        ground_truth_list.append([])
                
                if len(ground_truth_list) > 0:
                    # Run evaluation
                    eval_metrics = evaluate_dataset(results, ground_truth_list, iou_threshold=0.5)
                    
                    # Print evaluation summary
                    print_evaluation_summary(eval_metrics, title="Ground Truth Evaluation")
                    
                    # Save metrics
                    metrics_dir = Path(args.output_dir) / "metrics"
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save to CSV
                    save_metrics_to_csv(eval_metrics, str(metrics_dir / "evaluation_metrics.csv"))
                    
                    # Save to JSON
                    import json
                    with open(metrics_dir / "evaluation_metrics.json", 'w') as f:
                        # Convert numpy types to Python types for JSON serialization
                        json_metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                                       for k, v in eval_metrics.items() if k != 'per_image_metrics'}
                        json.dump(json_metrics, f, indent=2)
                    
                    print(f"\nEvaluation metrics saved to: {metrics_dir}")
                else:
                    print("WARNING: No ground truth annotations found")
            else:
                print(f"WARNING: Ground truth directory not found: {gt_dir}")
                print("    Skipping comprehensive evaluation")
        
        # Phase 3.5: Generate LaTeX Report (if requested)
        if args.generate_latex:
            print("\n" + "="*80)
            print("PHASE 3.5: Generating LaTeX Report for Paper")
            print("="*80)
            latex_file = generate_latex_report(args.output_dir, report_type="results")
            if latex_file:
                print(f"LaTeX report generated: {latex_file}")
            else:
                print("WARNING: LaTeX report generation skipped (no metrics available)")
        
        # Optional: Advanced Prompting
        if args.advanced_prompting:
            images = dataset.list_images()
            if images:
                print("\n" + "="*80)
                print("PHASE 4: Advanced Prompting Examples")
                print("="*80)
                run_advanced_prompting_examples(
                    str(images[0]),
                    output_dir=f"{args.output_dir}/advanced_prompting"
                )
        
        # Final Summary
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {args.output_dir}/")
        print(f"\nView the full report at: {args.output_dir}/report/report.html")
        print("\nNext steps:")
        print("  1. Review visualizations in visualizations/")
        print("  2. Check the HTML report for detailed analysis")
        print("  3. Use masks/ directory for further processing")
        print("  4. Run with --advanced-prompting for more examples")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Error during workflow: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're logged into Hugging Face:")
        print("     huggingface-cli login")
        print("  2. Accept SAM3 license:")
        print("     https://huggingface.co/facebook/sam3")
        print("  3. Check that all dependencies are installed:")
        print("     pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
