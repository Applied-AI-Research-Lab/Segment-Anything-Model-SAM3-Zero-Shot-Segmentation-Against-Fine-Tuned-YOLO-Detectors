#!/bin/bash
###############################################################################
# YOLO11-Medium Prediction Script
# 
# Runs predictions using trained YOLO11m-seg (medium) model.
#
# Usage:
#   ./prediction_yolo11m.sh minneapple
#   ./prediction_yolo11m.sh weedsgalore
#
# Model: yolo11m-seg (medium variant)
# Expected: Better accuracy than nano (~62-66% F1)
#
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() { echo -e "${GREEN}SUCCESS: $1${NC}"; }
print_error() { echo -e "${RED}ERROR: $1${NC}"; }
print_info() { echo -e "${BLUE}INFO: $1${NC}"; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

DATASET="${1:-minneapple}"

if [[ "$DATASET" != "minneapple" && "$DATASET" != "weedsgalore" ]]; then
    print_error "Invalid dataset: $DATASET"
    echo "Usage: $0 [minneapple|weedsgalore]"
    exit 1
fi

clear
print_header "YOLO11-Medium Prediction Pipeline"
echo "Model: yolo11m-seg (medium)"
echo "Dataset: $DATASET"
echo

###############################################################################
# Validate Model and Dataset
###############################################################################

print_header "Validation"

MODEL_PATH="ft/yolo11m/${DATASET}/best.pt"
DATA_DIR="data/$DATASET"

if [ ! -f "$MODEL_PATH" ]; then
    print_error "Model not found: $MODEL_PATH"
    print_info "Train the model first:"
    echo "    ./ft_yolo11m.sh $DATASET"
    exit 1
fi

if [ ! -d "$DATA_DIR/test" ]; then
    print_error "Test dataset not found: $DATA_DIR/test"
    exit 1
fi

print_success "Model: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
print_success "Test data: $DATA_DIR/test"

echo

###############################################################################
# Run Predictions
###############################################################################

print_header "Running Predictions"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/yolo11m_${DATASET}_${TIMESTAMP}"

print_info "Output directory: $OUTPUT_DIR"
echo

python3 models/yolo11/predict_yolo11.py \
    --dataset "$DATASET" \
    --model "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --generate-latex

if [ -d "$OUTPUT_DIR" ]; then
    print_success "Predictions complete"
    
    # Show results summary
    if [ -f "$OUTPUT_DIR/metrics/evaluation_metrics.json" ]; then
        echo
        print_header "Results Summary"
        python3 -c "
import json
with open('$OUTPUT_DIR/metrics/evaluation_metrics.json') as f:
    metrics = json.load(f)
print(f\"  F1 Score: {metrics['dataset_f1']:.1%}\")
print(f\"  Precision: {metrics['dataset_precision']:.1%}\")
print(f\"  Recall: {metrics['dataset_recall']:.1%}\")
print(f\"  Mean IoU: {metrics['mean_image_iou']:.1%}\")
"
    fi
    
    echo
    print_info "Detailed results: $OUTPUT_DIR"
else
    print_error "Prediction failed"
    exit 1
fi

echo
print_header "Done!"
