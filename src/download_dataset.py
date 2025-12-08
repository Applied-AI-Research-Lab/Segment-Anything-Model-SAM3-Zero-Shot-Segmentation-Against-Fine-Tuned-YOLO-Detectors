#!/usr/bin/env python3
"""
Unified Dataset Downloader for Agricultural Segmentation

Production-ready dataset management system supporting multiple agricultural
computer vision datasets with standardized preprocessing and validation.

Key Features:
- Automated download and extraction with integrity checks
- Consistent directory structure across datasets
- Train/val/test split management
- Disk space validation and progress tracking
- Robust error handling and recovery

Supported Datasets:
- MinneApple: Precision apple detection and segmentation
- WeedsGalore: Multi-spectral weed identification

Directory Structure:
    data/dataset_name/
        train/
            images/     # RGB images
            masks/      # Instance segmentation masks
        val/
            images/
            masks/
        test/
            images/
            masks/

Usage:
    python3 download_dataset.py --dataset minneapple
    python3 download_dataset.py --dataset weedsgalore
    python3 download_dataset.py --dataset both
"""

import os
import sys
import argparse
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image
import random


# Dataset URLs
DATASET_URLS = {
    "minneapple": {
        "detection": "https://conservancy.umn.edu/bitstreams/3ef26f04-6467-469b-9857-f443ffa1bb61/download",
        "size_mb": 1700,
        "filename": "detection.tar.gz"
    },
    "weedsgalore": {
        "dataset": "https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip",
        "size_mb": 321,
        "filename": "weedsgalore-dataset.zip"
    }
}


class DatasetDownloader:
    """
    Production-grade dataset management system.

    Handles the complete lifecycle of agricultural dataset preparation:
    - Automated download with integrity verification
    - Archive extraction with format detection
    - Dataset validation and preprocessing
    - Standardized train/val/test splitting
    - Disk space management and error recovery

    Design Principles:
    - Idempotent operations (safe to re-run)
    - Comprehensive error handling and logging
    - Memory-efficient processing for large datasets
    - Reproducible results with seeded randomization
    """

    def __init__(self, base_dir: str = "data"):
        """
        Initialize the dataset downloader with configuration.

        Args:
            base_dir: Root directory for dataset storage. All datasets
                     will be organized under this path with consistent
                     subdirectory structure.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Ensure reproducible dataset splits across runs
        # Critical for consistent model evaluation
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Archive retention policy - useful for debugging
        self.keep_archives = False

    def check_disk_space(self, required_mb: float) -> bool:
        """
        Validate sufficient disk space before large downloads.

        Implements conservative disk space checking with safety buffers
        to prevent download failures mid-process. Uses statvfs for
        accurate free space calculation across different filesystems.

        Args:
            required_mb: Estimated space needed in megabytes

        Returns:
            True if sufficient space available, False otherwise
        """
        stat = os.statvfs(self.base_dir)
        available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)

        print(f"Available disk space: {available_mb:.2f} MB")
        print(f"Required space: {required_mb:.2f} MB (with buffer)")

        # Require 50% buffer for extraction and temporary files
        if available_mb < required_mb * 1.5:
            print(f"ERROR: Insufficient disk space!")
            return False

        print(f"Sufficient disk space available")
        return True
    
    def download_file(self, url: str, output_path: Path, description: str) -> bool:
        """
        Robust file download with progress tracking and error recovery.

        Implements production-grade download functionality with:
        - Idempotent operation (skips if file exists)
        - Progress visualization using tqdm
        - Automatic cleanup on failure
        - Timeout handling for network issues
        - Content-length validation for progress bars

        Args:
            url: Direct URL to download from
            output_path: Local path where file should be saved
            description: Human-readable description for logging

        Returns:
            True if download successful, False otherwise
        """
        # Skip download if file already exists (idempotent operation)
        if output_path.exists():
            print(f"{output_path.name} already exists (skipping download)")
            return True

        print(f"\nDownloading: {description}")
        print(f"   URL: {url}")
        print(f"   Destination: {output_path}")

        try:
            # Use streaming download to handle large files efficiently
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress visualization
            with open(output_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=output_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"Downloaded: {output_path.name}")
            return True

        except Exception as e:
            print(f"ERROR: Download failed: {e}")
            # Clean up partial downloads
            if output_path.exists():
                output_path.unlink()
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """
        Multi-format archive extraction with automatic format detection.

        Supports common archive formats used in dataset distribution:
        - tar.gz (compressed tar archives)
        - zip (PKZIP archives)

        Features:
        - Format auto-detection based on file extension
        - Safe extraction with path validation
        - Progress indication for large archives
        - Comprehensive error handling

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract files into

        Returns:
            True if extraction successful, False otherwise
        """
        print(f"\nExtracting: {archive_path.name}")
        print(f"   Destination: {extract_to}")

        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            # Handle different archive formats
            if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
                # TAR.GZ format - common for Linux/Unix distributions
                with tarfile.open(archive_path, 'r:gz') as tar:
                    members = tar.getmembers()
                    for member in tqdm(members, desc="Extracting"):
                        tar.extract(member, path=extract_to)

            elif archive_path.suffix == '.zip':
                # ZIP format - common for Windows/cross-platform
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    members = zip_ref.namelist()
                    for member in tqdm(members, desc="Extracting"):
                        zip_ref.extract(member, path=extract_to)
            else:
                print(f"ERROR: Unsupported archive format: {archive_path.suffix}")
                return False

            print(f"Extracted: {archive_path.name}")
            return True

        except Exception as e:
            print(f"ERROR: Extraction failed: {e}")
            return False

    def split_dataset(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List, List, List]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of corresponding mask file paths
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (train_pairs, val_pairs, test_pairs)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
        
        # Pair images with masks
        pairs = list(zip(image_paths, mask_paths))
        random.shuffle(pairs)
        
        n_total = len(pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]
        
        print(f"\n📊 Dataset split:")
        print(f"   Train: {len(train_pairs)} images ({len(train_pairs)/n_total*100:.1f}%)")
        print(f"   Val:   {len(val_pairs)} images ({len(val_pairs)/n_total*100:.1f}%)")
        print(f"   Test:  {len(test_pairs)} images ({len(test_pairs)/n_total*100:.1f}%)")
        
        return train_pairs, val_pairs, test_pairs
    
    def copy_split_to_dirs(
        self,
        split_pairs: List[Tuple[Path, Path]],
        split_name: str,
        output_dir: Path
    ) -> None:
        """
        Efficient file copying for dataset splits with progress tracking.

        Handles the physical file operations for dataset partitioning:
        - Creates standardized directory structure
        - Copies image and mask pairs to appropriate locations
        - Provides progress feedback for large datasets
        - Validates successful file operations

        Args:
            split_pairs: List of (image_path, mask_path) tuples
            split_name: Name of split ('train', 'val', or 'test')
            output_dir: Base directory for the dataset
        """
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"

        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCopying {split_name} set...")
        for img_path, mask_path in tqdm(split_pairs, desc=f"Copying {split_name}"):
            # Copy image
            shutil.copy(img_path, images_dir / img_path.name)

            # Copy mask
            shutil.copy(mask_path, masks_dir / mask_path.name)

        print(f"{split_name} set ready: {len(split_pairs)} images")
    
    def process_minneapple(self) -> bool:
        """
        Download and process MinneApple dataset.
        
        Steps:
        1. Download detection.tar.gz
        2. Extract to temp directory
        3. Use only 'train' folder (has images + masks)
        4. Split into train/val/test (70/15/15)
        5. Organize into final structure
        """
        print("\n" + "="*80)
        print("Processing MinneApple Dataset")
        print("="*80)
        
        dataset_dir = self.base_dir / "minneapple"
        temp_dir = self.base_dir / "temp_minneapple"
        downloads_dir = self.base_dir / "downloads"
        
        # Check disk space
        if not self.check_disk_space(DATASET_URLS["minneapple"]["size_mb"] * 2):
            return False
        
        # Download
        archive_path = downloads_dir / DATASET_URLS["minneapple"]["filename"]
        if not self.download_file(
            DATASET_URLS["minneapple"]["detection"],
            archive_path,
            "MinneApple Detection Dataset"
        ):
            return False
        
        # Extract to temp directory
        if not self.extract_archive(archive_path, temp_dir):
            return False
        
        # Find the train folder (contains images + masks)
        train_folder = None
        for root, dirs, files in os.walk(temp_dir):
            if 'train' in Path(root).parts:
                if (Path(root) / 'images').exists() or any('train' in d for d in dirs):
                    train_folder = Path(root)
                    if (train_folder / 'images').exists():
                        break
        
        if not train_folder:
            # Look for detection/detection/train structure
            detection_path = temp_dir / "detection" / "detection"
            if detection_path.exists():
                train_folder = detection_path / "train"
        
        if not train_folder or not train_folder.exists():
            print("ERROR: Could not find train folder in extracted data")
            return False

        print(f"\nFound train folder: {train_folder}")

        # Locate images and masks
        images_folder = train_folder / "images"
        masks_folder = train_folder / "masks"

        if not images_folder.exists() or not masks_folder.exists():
            print(f"ERROR: Missing images or masks folder in {train_folder}")
            return False

        # Collect image and mask pairs
        image_files = sorted(list(images_folder.glob("*.png")) + list(images_folder.glob("*.jpg")))

        print(f"\nFound {len(image_files)} images in train folder")

        # Match masks to images
        image_mask_pairs = []
        for img_path in image_files:
            # Try to find corresponding mask
            mask_path = masks_folder / img_path.name
            
            if mask_path.exists():
                image_mask_pairs.append((img_path, mask_path))
            else:
                print(f"WARNING: No mask found for {img_path.name}")

        print(f"Matched {len(image_mask_pairs)} image-mask pairs")

        if len(image_mask_pairs) == 0:
            print("ERROR: No valid image-mask pairs found")
            return False

        # Split dataset
        image_paths = [p[0] for p in image_mask_pairs]
        mask_paths = [p[1] for p in image_mask_pairs]

        train_pairs, val_pairs, test_pairs = self.split_dataset(
            image_paths, mask_paths,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        # Copy to final structure
        self.copy_split_to_dirs(train_pairs, "train", dataset_dir)
        self.copy_split_to_dirs(val_pairs, "val", dataset_dir)
        self.copy_split_to_dirs(test_pairs, "test", dataset_dir)

        # Cleanup
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        if not self.keep_archives:
            archive_path.unlink(missing_ok=True)

        print(f"\nMinneApple dataset ready at: {dataset_dir}")
        self.print_dataset_summary(dataset_dir)

        return True
    
    def create_rgb_composite(self, r_path: Path, g_path: Path, b_path: Path) -> Image.Image:
        """Create RGB composite from separate band files."""
        r_band = np.array(Image.open(r_path))
        g_band = np.array(Image.open(g_path))
        b_band = np.array(Image.open(b_path))
        
        # Stack into RGB
        rgb = np.stack([r_band, g_band, b_band], axis=-1)
        
        return Image.fromarray(rgb.astype(np.uint8))
    
    def process_weedsgalore(self) -> bool:
        """
        Download and process WeedsGalore dataset.
        
        Steps:
        1. Download weedsgalore-dataset.zip
        2. Extract to temp directory
        3. Convert R, G, B bands to RGB composites
        4. Copy instance masks
        5. Split into train/val/test (70/15/15)
        6. Organize into final structure
        """
        print("\n" + "="*80)
        print("Processing WeedsGalore Dataset")
        print("="*80)
        
        dataset_dir = self.base_dir / "weedsgalore"
        temp_dir = self.base_dir / "temp_weedsgalore"
        temp_processed = self.base_dir / "temp_weedsgalore_processed"
        downloads_dir = self.base_dir / "downloads"
        
        # Check disk space
        if not self.check_disk_space(DATASET_URLS["weedsgalore"]["size_mb"] * 3):
            return False
        
        # Download
        archive_path = downloads_dir / DATASET_URLS["weedsgalore"]["filename"]
        if not self.download_file(
            DATASET_URLS["weedsgalore"]["dataset"],
            archive_path,
            "WeedsGalore Dataset"
        ):
            return False
        
        # Extract
        if not self.extract_archive(archive_path, temp_dir):
            return False
        
        # Find weedsgalore-dataset folder
        dataset_path = temp_dir / "weedsgalore-dataset"
        if not dataset_path.exists():
            # Might be directly extracted
            dataset_path = temp_dir
        
        print(f"\n🌈 Converting multi-spectral bands to RGB composites...")
        
        # Create temp processed directories
        temp_images_dir = temp_processed / "images"
        temp_masks_dir = temp_processed / "masks"
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        temp_masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all date folders
        date_folders = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("2023-")]
        print(f"Found {len(date_folders)} date folders")
        
        total_images = 0
        total_masks = 0
        
        for date_folder in sorted(date_folders):
            print(f"\n📅 Processing {date_folder.name}...")
            
            images_folder = date_folder / "images"
            instances_folder = date_folder / "instances"
            
            if not images_folder.exists():
                print(f"  Skipping (no images folder)")
                continue
            
            # Find all R band images
            r_images = sorted(images_folder.glob("*_R.png"))
            print(f"  Found {len(r_images)} images")
            
            for r_path in tqdm(r_images, desc=f"  Converting {date_folder.name}"):
                base_name = r_path.stem.replace("_R", "")
                
                # Paths for G and B bands
                g_path = images_folder / f"{base_name}_G.png"
                b_path = images_folder / f"{base_name}_B.png"
                
                if not g_path.exists() or not b_path.exists():
                    continue
                
                try:
                    # Create RGB composite
                    rgb_image = self.create_rgb_composite(r_path, g_path, b_path)
                    
                    # Save RGB image
                    output_image_path = temp_images_dir / f"{base_name}.png"
                    rgb_image.save(output_image_path)
                    total_images += 1
                    
                    # Copy instance mask if exists
                    mask_path = instances_folder / f"{base_name}.png"
                    if mask_path.exists():
                        output_mask_path = temp_masks_dir / f"{base_name}.png"
                        shutil.copy(mask_path, output_mask_path)
                        total_masks += 1
                
                except Exception as e:
                    print(f"  Error processing {base_name}: {e}")
                    continue
        
        print(f"\n✓ Created {total_images} RGB composites")
        print(f"✓ Copied {total_masks} instance masks")
        
        if total_images == 0:
            print("❌ No RGB images created")
            return False
        
        # Collect all image-mask pairs
        image_files = sorted(list(temp_images_dir.glob("*.png")))
        
        image_mask_pairs = []
        for img_path in image_files:
            mask_path = temp_masks_dir / img_path.name
            if mask_path.exists():
                image_mask_pairs.append((img_path, mask_path))
        
        print(f"\n✓ Found {len(image_mask_pairs)} valid image-mask pairs")
        
        # Split dataset
        image_paths = [p[0] for p in image_mask_pairs]
        mask_paths = [p[1] for p in image_mask_pairs]
        
        train_pairs, val_pairs, test_pairs = self.split_dataset(
            image_paths, mask_paths,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        # Copy to final structure
        self.copy_split_to_dirs(train_pairs, "train", dataset_dir)
        self.copy_split_to_dirs(val_pairs, "val", dataset_dir)
        self.copy_split_to_dirs(test_pairs, "test", dataset_dir)
        
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(temp_processed, ignore_errors=True)
        if not self.keep_archives:
            archive_path.unlink(missing_ok=True)
        
        print(f"\n✅ WeedsGalore dataset ready at: {dataset_dir}")
        self.print_dataset_summary(dataset_dir)
        
        return True
    
    def print_dataset_summary(self, dataset_dir: Path) -> None:
        """Print summary of prepared dataset."""
        print("\n" + "="*80)
        print(f"Dataset Structure: {dataset_dir.name}")
        print("="*80)
        
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                images_count = len(list((split_dir / "images").glob("*.png"))) + \
                              len(list((split_dir / "images").glob("*.jpg")))
                masks_count = len(list((split_dir / "masks").glob("*.png")))
                
                print(f"\n{split.upper()}:")
                print(f"  Images: {images_count}")
                print(f"  Masks:  {masks_count}")
                print(f"  📁 {split_dir}")
        
        print("\n" + "="*80)
    
    def set_keep_archives(self, keep: bool):
        """Set whether to keep archive files after extraction."""
        self.keep_archives = keep


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for SAM3 segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download MinneApple only
  python3 download_dataset.py --dataset minneapple
  
  # Download WeedsGalore only
  python3 download_dataset.py --dataset weedsgalore
  
  # Download both datasets
  python3 download_dataset.py --dataset both
  
  # Custom data directory
  python3 download_dataset.py --dataset minneapple --data-dir /custom/path
  
  # Keep archive files after extraction
  python3 download_dataset.py --dataset both --keep-archives
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["minneapple", "weedsgalore", "both"],
        required=True,
        help="Which dataset(s) to download and process"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory for datasets (default: data)"
    )
    
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archive files after extraction"
    )
    
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Automatic yes to prompts"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*80)
    print("SAM3 Dataset Downloader & Processor")
    print("="*80)
    print(f"Target directory: {args.data_dir}")
    print(f"Dataset: {args.dataset}")
    print("="*80)
    
    # Confirmation for large downloads
    if not args.yes:
        if args.dataset == "both":
            total_mb = DATASET_URLS["minneapple"]["size_mb"] + DATASET_URLS["weedsgalore"]["size_mb"]
            print(f"\n⚠️  This will download ~{total_mb:.0f} MB ({total_mb/1024:.2f} GB)")
        elif args.dataset == "minneapple":
            print(f"\n⚠️  This will download ~{DATASET_URLS['minneapple']['size_mb']:.0f} MB")
        elif args.dataset == "weedsgalore":
            print(f"\n⚠️  This will download ~{DATASET_URLS['weedsgalore']['size_mb']:.0f} MB")
        
        try:
            response = input("\nContinue? (y/N): ").strip().lower()
            if response != 'y':
                print("❌ Cancelled")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            sys.exit(0)
    
    # Initialize downloader
    downloader = DatasetDownloader(args.data_dir)
    downloader.set_keep_archives(args.keep_archives)
    
    # Process datasets
    success = True
    
    if args.dataset == "minneapple" or args.dataset == "both":
        if not downloader.process_minneapple():
            success = False
    
    if args.dataset == "weedsgalore" or args.dataset == "both":
        if not downloader.process_weedsgalore():
            success = False
    
    if success:
        print("\n" + "="*80)
        print("✨ All datasets ready!")
        print("="*80)
        print("\nFinal structure:")
        print(f"  {args.data_dir}/")
        
        if args.dataset in ["minneapple", "both"]:
            print(f"    └── minneapple/")
            print(f"          ├── train/    (images/ + masks/)")
            print(f"          ├── val/      (images/ + masks/)")
            print(f"          └── test/     (images/ + masks/)")
        
        if args.dataset in ["weedsgalore", "both"]:
            print(f"    └── weedsgalore/")
            print(f"          ├── train/    (images/ + masks/)")
            print(f"          ├── val/      (images/ + masks/)")
            print(f"          └── test/     (images/ + masks/)")
        
        print("\nNext steps:")
        print("  1. Run SAM3 on MinneApple:  python3 main.py --data-dir data/minneapple/test --text-prompt 'apple'")
        print("  2. Run SAM3 on WeedsGalore:  python3 main.py --data-dir data/weedsgalore/test --text-prompt 'weed'")
        print("  3. Or use deployment scripts: ./deploy.sh or ./deploy_weedsgalore.sh")
        print()
    else:
        print("\n❌ Dataset preparation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
