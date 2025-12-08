#!/bin/bash
###############################################################################
# YOLO11 Prediction Script
# 
# Runs predictions using trained YOLO11 models on test sets.
# Generates visualizations, masks, and evaluation metrics.
#
# Usage:
#   ./prediction_yolo11.sh minneapple     # Predict MinneApple only
#   ./prediction_yolo11.sh weedsgalore    # Predict WeedsGalore only
#   ./prediction_yolo11.sh both           # Predict both datasets
#
# Requirements:
#   - Trained models in ft/yolo11/{dataset}/best.pt
#   - Test datasets in data/{dataset}/test/
#   - GPU with CUDA support (recommended)
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

# Get timestamp for output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

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
print_header "YOLO11 Prediction Pipeline"
echo "This script runs predictions using trained YOLO11 models"
echo

print_header "Phase 1: System Check"

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python: $PYTHON_VERSION"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    print_success "CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "CUDA not detected - will use CPU (slower)"
fi

# Check PyTorch
if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    print_success "PyTorch available"
else
    print_error "PyTorch not found"
    exit 1
fi

# Check ultralytics
if python3 -c "import ultralytics" 2>/dev/null; then
    print_success "Ultralytics (YOLO11) available"
else
    print_error "Ultralytics not found"
    print_info "Install with: pip install ultralytics"
    exit 1
fi

echo

###############################################################################
# Phase 2: Validate Trained Models
###############################################################################

print_header "Phase 2: Validate Trained Models"

for DATASET in "${DATASETS[@]}"; do
    MODEL_PATH="ft/yolo11/${DATASET}/best.pt"
    
    if [ ! -f "$MODEL_PATH" ]; then
        print_error "Model not found: $MODEL_PATH"
        print_info "Please run ft_yolo11.sh first to train the model"
        exit 1
    fi
    
    print_success "Model found: $MODEL_PATH"
    
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    print_info "Model size: $MODEL_SIZE"
done

echo

###############################################################################
# Phase 3: Validate Test Datasets
###############################################################################

print_header "Phase 3: Validate Test Datasets"

for DATASET in "${DATASETS[@]}"; do
    DATA_DIR="data/$DATASET"
    TEST_IMAGES_DIR="$DATA_DIR/test/images"
    TEST_MASKS_DIR="$DATA_DIR/test/masks"
    
    if [ ! -d "$TEST_IMAGES_DIR" ]; then
        print_error "Test images not found: $TEST_IMAGES_DIR"
        print_info "Please download datasets first:"
        echo "    ./datasets.sh $DATASET"
        exit 1
    fi
    
    # Count test images
    TEST_IMAGE_COUNT=$(find "$TEST_IMAGES_DIR" -name "*.png" -o -name "*.jpg" | wc -l | tr -d ' ')
    print_success "Found $TEST_IMAGE_COUNT test images for $DATASET"
    
    # Check masks
    if [ -d "$TEST_MASKS_DIR" ]; then
        TEST_MASK_COUNT=$(find "$TEST_MASKS_DIR" -name "*.png" | wc -l | tr -d ' ')
        print_success "Found $TEST_MASK_COUNT ground truth masks"
    else
        print_warning "Ground truth masks not found (evaluation will be limited)"
    fi
done

echo

###############################################################################
# Phase 4: Run Predictions
###############################################################################

print_header "Phase 4: Run Predictions"

RESULT_DIRS=()

for DATASET in "${DATASETS[@]}"; do
    MODEL_PATH="ft/yolo11/${DATASET}/best.pt"
    OUTPUT_DIR="results/yolo11_${DATASET}_${TIMESTAMP}"
    
    echo
    print_header "Predicting: $DATASET"
    print_info "Model: $MODEL_PATH"
    print_info "Output: $OUTPUT_DIR"
    echo
    
    python3 models/yolo11/predict_yolo11.py \
        --dataset "$DATASET" \
        --model "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --generate-latex
    
    if [ -d "$OUTPUT_DIR" ]; then
        print_success "Predictions complete: $OUTPUT_DIR"
        RESULT_DIRS+=("$OUTPUT_DIR")
        
        # Create zip archive of results
        print_info "Creating zip archive..."
        RESULTS_ZIP="results/$(basename $OUTPUT_DIR).zip"
        cd results && zip -r "$(basename $RESULTS_ZIP)" "$(basename $OUTPUT_DIR)" > /dev/null 2>&1
        cd ..
        
        if [ -f "$RESULTS_ZIP" ]; then
            ZIP_SIZE=$(du -h "$RESULTS_ZIP" | cut -f1)
            print_success "Results archived: $(basename $RESULTS_ZIP) ($ZIP_SIZE)"
        else
            print_warning "Failed to create zip archive"
        fi
    else
        print_error "Prediction failed for $DATASET"
        exit 1
    fi
    
    echo
done

###############################################################################
# Phase 5: Generate Summary Report
###############################################################################

print_header "Phase 5: Generate Summary Report"

SUMMARY_FILE="results/YOLO11_PREDICTION_SUMMARY_${TIMESTAMP}.txt"

{
    echo "============================================================================"
    echo "YOLO11 PREDICTION SUMMARY"
    echo "============================================================================"
    echo ""
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Python: $PYTHON_VERSION"
    echo ""
    echo "Datasets Processed: ${DATASETS[*]}"
    echo ""
    
    for i in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$i]}"
        RESULT_DIR="${RESULT_DIRS[$i]}"
        
        echo "----------------------------------------------------------------------------"
        echo "Dataset: $DATASET"
        echo "----------------------------------------------------------------------------"
        echo ""
        
        if [ -d "$RESULT_DIR" ]; then
            echo "Results Directory: $RESULT_DIR"
            echo ""
            
            # Model info
            MODEL_PATH="ft/yolo11/${DATASET}/best.pt"
            if [ -f "$MODEL_PATH" ]; then
                MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
                echo "Model: $MODEL_PATH"
                echo "Model Size: $MODEL_SIZE"
            fi
            echo ""
            
            # Metrics
            METRICS_FILE="$RESULT_DIR/metrics/evaluation_metrics.json"
            if [ -f "$METRICS_FILE" ]; then
                echo "Evaluation Metrics:"
                echo "-------------------"
                
                if command -v jq &> /dev/null; then
                    MEAN_IOU=$(jq -r '.mean_iou // "N/A"' "$METRICS_FILE")
                    F1_SCORE=$(jq -r '.f1_score // "N/A"' "$METRICS_FILE")
                    PRECISION=$(jq -r '.precision // "N/A"' "$METRICS_FILE")
                    RECALL=$(jq -r '.recall // "N/A"' "$METRICS_FILE")
                else
                    MEAN_IOU=$(python3 -c "import json; print(f\"{json.load(open('$METRICS_FILE'))['mean_iou']:.4f}\")" 2>/dev/null || echo "N/A")
                    F1_SCORE=$(python3 -c "import json; print(f\"{json.load(open('$METRICS_FILE'))['f1_score']:.4f}\")" 2>/dev/null || echo "N/A")
                    PRECISION=$(python3 -c "import json; print(f\"{json.load(open('$METRICS_FILE'))['precision']:.4f}\")" 2>/dev/null || echo "N/A")
                    RECALL=$(python3 -c "import json; print(f\"{json.load(open('$METRICS_FILE'))['recall']:.4f}\")" 2>/dev/null || echo "N/A")
                fi
                
                echo "  Mean IoU:  $MEAN_IOU"
                echo "  F1 Score:  $F1_SCORE"
                echo "  Precision: $PRECISION"
                echo "  Recall:    $RECALL"
                echo ""
            fi
            
            # Summary
            SUMMARY_JSON="$RESULT_DIR/report/evaluation_summary.json"
            if [ -f "$SUMMARY_JSON" ]; then
                if command -v jq &> /dev/null; then
                    NUM_IMAGES=$(jq -r '.num_images // "N/A"' "$SUMMARY_JSON")
                    TOTAL_DETS=$(jq -r '.total_detections // "N/A"' "$SUMMARY_JSON")
                else
                    NUM_IMAGES=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['num_images'])" 2>/dev/null || echo "N/A")
                    TOTAL_DETS=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['total_detections'])" 2>/dev/null || echo "N/A")
                fi
                
                echo "Processing Statistics:"
                echo "---------------------"
                echo "  Images processed:    $NUM_IMAGES"
                echo "  Total detections:    $TOTAL_DETS"
                echo ""
            fi
            
            # File counts
            if [ -d "$RESULT_DIR/visualizations" ]; then
                VIZ_COUNT=$(find "$RESULT_DIR/visualizations" -type f | wc -l | xargs)
                echo "  Visualizations saved: $VIZ_COUNT"
            fi
            
            if [ -d "$RESULT_DIR/masks" ]; then
                MASK_COUNT=$(find "$RESULT_DIR/masks" -type f | wc -l | xargs)
                echo "  Masks saved:          $MASK_COUNT"
            fi
            
            echo ""
        else
            echo "ERROR: Results directory not found"
            echo ""
        fi
    done
    
    echo "============================================================================"
    echo "Next Steps:"
    echo "============================================================================"
    echo ""
    echo "1. View visualizations in: results/yolo11_*/visualizations/"
    echo "2. Check metrics in: results/yolo11_*/metrics/"
    echo "3. Compare with SAM3 results:"
    echo "   - Zero-shot SAM3: results/*sam3*<dataset>*/"
    echo "   - Full FT SAM3: results/sam3_full_*/"
    echo "   - LoRA FT SAM3: results/sam3_lora_*/"
    echo "4. Generate comparison reports"
    echo ""
    echo "============================================================================"
    
} | tee "$SUMMARY_FILE"

print_success "Summary saved to: $SUMMARY_FILE"

###############################################################################
# Prediction Complete
###############################################################################

echo
print_header "YOLO11 Prediction Complete!"

echo
print_info "Results:"
for RESULT_DIR in "${RESULT_DIRS[@]}"; do
    echo "  - $RESULT_DIR"
done

echo
print_info "Summary: $SUMMARY_FILE"

echo
print_info "Compare with SAM3 models:"
echo "  • Zero-shot SAM3: ./prediction_sam3_zero.sh"
echo "  • Full FT SAM3: ./prediction_sam3_full.sh"
echo "  • LoRA FT SAM3: ./prediction_sam3_lora.sh"

echo
print_header "Done!"
echo
