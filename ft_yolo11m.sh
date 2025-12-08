#!/bin/bash
###############################################################################
# YOLO11-Medium Fine-tuning Script
# 
# Trains YOLO11m-seg (medium) model on MinneApple dataset.
# Fine-tuned models are saved to ft/yolo11m/{dataset}/
#
# Usage:
#   ./ft_yolo11m.sh minneapple     # Train on MinneApple
#   ./ft_yolo11m.sh weedsgalore    # Train on WeedsGalore
#
# Model: yolo11m-seg (medium variant - better accuracy than nano)
# Expected: ~40-50 MB model, F1 ≈ 62-66% (+10-12% over nano)
#
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() { echo -e "${GREEN}SUCCESS: $1${NC}"; }
print_error() { echo -e "${RED}ERROR: $1${NC}"; }
print_warning() { echo -e "${YELLOW}WARNING: $1${NC}"; }
print_info() { echo -e "${BLUE}INFO: $1${NC}"; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

###############################################################################
# Parse Arguments
###############################################################################

DATASET="${1:-minneapple}"

if [[ "$DATASET" != "minneapple" && "$DATASET" != "weedsgalore" ]]; then
    print_error "Invalid dataset: $DATASET"
    echo "Usage: $0 [minneapple|weedsgalore]"
    exit 1
fi

###############################################################################
# System Check
###############################################################################

clear
print_header "YOLO11-Medium Fine-tuning Pipeline"
echo "Model: yolo11m-seg (medium variant)"
echo "Dataset: $DATASET"
echo

print_header "System Check"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python: $PYTHON_VERSION"

if command -v nvidia-smi &> /dev/null; then
    print_success "CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "CUDA not detected - will use CPU (much slower)"
fi

echo

###############################################################################
# Validate Dataset
###############################################################################

print_header "Validate Dataset"

DATA_DIR="data/$DATASET"

if [ ! -d "$DATA_DIR/train" ] || [ ! -d "$DATA_DIR/val" ] || [ ! -d "$DATA_DIR/test" ]; then
    print_error "Dataset not found: $DATASET"
    print_info "Please download dataset first:"
    echo "    ./datasets.sh $DATASET"
    exit 1
fi

print_success "Dataset validated: $DATASET"
echo

###############################################################################
# Train YOLO11-Medium Model
###############################################################################

print_header "Train YOLO11-Medium Model"

EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
MODEL_DIR="ft/yolo11m/${DATASET}"
MODEL_PATH="${MODEL_DIR}/best.pt"

print_info "Training Configuration:"
echo "  Model: yolo11m-seg (medium)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo "  Output: $MODEL_DIR"
echo

if [ -f "$MODEL_PATH" ]; then
    print_warning "Model already exists: $MODEL_PATH"
    read -p "   Retrain model? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Using existing model: $MODEL_PATH"
        exit 0
    fi
fi

print_info "Starting training on $DATASET..."
echo

python3 models/yolo11/train_yolo11.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --imgsz "$IMG_SIZE" \
    --device "0,1" \
    --model-dir "$MODEL_DIR" \
    --model-variant "m"

if [ -f "$MODEL_PATH" ]; then
    print_success "Model trained: $MODEL_PATH"
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    print_info "Model size: $MODEL_SIZE"
else
    print_error "Training failed"
    exit 1
fi

echo

###############################################################################
# Summary
###############################################################################

print_header "YOLO11-Medium Training Complete!"

echo
print_info "Model: $MODEL_PATH"
print_info "Size: $(du -h "$MODEL_PATH" | cut -f1)"
echo
print_info "Next Steps:"
echo "  1. Run predictions: ./prediction_yolo11m.sh $DATASET"
echo "  2. Compare with nano model results"
echo

print_header "Done!"
