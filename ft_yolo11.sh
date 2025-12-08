#!/bin/bash
###############################################################################
# YOLO11 Fine-tuning Script
# 
# Trains YOLO11 models on MinneApple and/or WeedsGalore datasets.
# Fine-tuned models are saved to ft/yolo11/{dataset}/
#
# Usage:
#   ./ft_yolo11.sh minneapple     # Train on MinneApple only
#   ./ft_yolo11.sh weedsgalore    # Train on WeedsGalore only
#   ./ft_yolo11.sh both           # Train on both datasets
#
# Requirements:
#   - Python 3.7+
#   - CUDA-capable GPU (recommended)
#   - ~5GB disk space per dataset
#
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

# Function to print colored output
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

###############################################################################
# Parse Arguments
###############################################################################

DATASET_ARG="${1:-both}"

# Validate argument
if [[ "$DATASET_ARG" != "minneapple" && "$DATASET_ARG" != "weedsgalore" && "$DATASET_ARG" != "both" ]]; then
    print_error "Invalid argument: $DATASET_ARG"
    echo "Usage: $0 [minneapple|weedsgalore|both]"
    exit 1
fi

# Determine which datasets to process
if [ "$DATASET_ARG" = "both" ]; then
    DATASETS=("minneapple" "weedsgalore")
else
    DATASETS=("$DATASET_ARG")
fi

###############################################################################
# Phase 1: System Check
###############################################################################

clear
print_header "YOLO11 Fine-tuning Pipeline"
echo "This script trains YOLO11 models for instance segmentation"
echo

print_header "Phase 1: System Check"

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python: $PYTHON_VERSION"

# Check available disk space
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
print_success "Available disk space: $AVAILABLE_SPACE"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    print_success "CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "CUDA not detected - will use CPU (much slower)"
fi

echo

###############################################################################
# Phase 2: Install Dependencies
###############################################################################

print_header "Phase 2: Install Dependencies"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install requirements
if [ -f "requirements.txt" ]; then
    print_info "Installing base dependencies..."
    pip install -r requirements.txt
    print_success "Base requirements installed"
fi

# Install YOLO11 requirements
print_info "Installing YOLO11 (ultralytics)..."
pip install ultralytics opencv-python
print_success "YOLO11 requirements installed"

echo

###############################################################################
# Phase 3: Validate Datasets
###############################################################################

print_header "Phase 3: Validate Datasets"

for DATASET in "${DATASETS[@]}"; do
    DATA_DIR="data/$DATASET"
    
    if [ ! -d "$DATA_DIR/train" ] || [ ! -d "$DATA_DIR/val" ] || [ ! -d "$DATA_DIR/test" ]; then
        print_error "Dataset not found: $DATASET"
        print_info "Please download datasets first:"
        echo "    ./datasets.sh $DATASET"
        exit 1
    fi
    
    print_success "Dataset validated: $DATASET"
done

echo

###############################################################################
# Phase 4: Train YOLO11 Models
###############################################################################

print_header "Phase 4: Train YOLO11 Models"

# Training parameters
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640

print_info "Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo

TRAINED_MODELS=()

for DATASET in "${DATASETS[@]}"; do
    MODEL_PATH="ft/yolo11/${DATASET}/best.pt"
    
    echo
    print_header "Training: $DATASET"
    
    if [ -f "$MODEL_PATH" ]; then
        print_warning "Model already exists: $MODEL_PATH"
        read -p "   Retrain model? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Using existing model: $MODEL_PATH"
            TRAINED_MODELS+=("$MODEL_PATH")
            continue
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
        --model-dir "ft/yolo11/${DATASET}"
    
    if [ -f "$MODEL_PATH" ]; then
        print_success "Model trained: $MODEL_PATH"
        MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
        print_info "Model size: $MODEL_SIZE"
        TRAINED_MODELS+=("$MODEL_PATH")
    else
        print_error "Training failed for $DATASET"
        exit 1
    fi
    
    echo
done

###############################################################################
# Phase 5: Training Summary
###############################################################################

print_header "Phase 5: Training Summary"

SUMMARY_FILE="ft/yolo11/TRAINING_SUMMARY_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "ft/yolo11"

{
    echo "============================================================================"
    echo "YOLO11 TRAINING SUMMARY"
    echo "============================================================================"
    echo ""
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Python: $PYTHON_VERSION"
    echo ""
    echo "Training Configuration:"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Image size: ${IMG_SIZE}x${IMG_SIZE}"
    echo ""
    echo "Datasets Trained: ${DATASETS[*]}"
    echo ""
    
    for i in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$i]}"
        MODEL_PATH="${TRAINED_MODELS[$i]}"
        
        echo "----------------------------------------------------------------------------"
        echo "Dataset: $DATASET"
        echo "----------------------------------------------------------------------------"
        
        if [ -f "$MODEL_PATH" ]; then
            MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
            echo "  Model: $MODEL_PATH"
            echo "  Size: $MODEL_SIZE"
            echo ""
            
            # Check for training metadata
            METADATA_FILE="ft/yolo11/${DATASET}/training_metadata.json"
            if [ -f "$METADATA_FILE" ]; then
                echo "  Training completed successfully"
                echo "  Metadata: $METADATA_FILE"
            fi
        else
            echo "  ❌ Model not found"
        fi
        echo ""
    done
    
    echo "============================================================================"
    echo "Next Steps:"
    echo "============================================================================"
    echo ""
    echo "Run predictions using trained models:"
    echo "  ./prediction_yolo11.sh ${DATASET_ARG}"
    echo ""
    echo "Models saved in:"
    for DATASET in "${DATASETS[@]}"; do
        echo "  - ft/yolo11/${DATASET}/best.pt"
    done
    echo ""
    echo "============================================================================"
    
} | tee "$SUMMARY_FILE"

print_success "Summary saved to: $SUMMARY_FILE"

###############################################################################
# Training Complete
###############################################################################

echo
print_header "YOLO11 Training Complete!"

echo
print_info "Trained Models:"
for MODEL_PATH in "${TRAINED_MODELS[@]}"; do
    echo "  - $MODEL_PATH"
done

echo
print_info "Next Steps:"
echo "  1. Run predictions: ./prediction_yolo11.sh ${DATASET_ARG}"
echo "  2. Compare with SAM3 results"
echo "  3. Evaluate model performance"

echo
print_header "Done!"
echo
