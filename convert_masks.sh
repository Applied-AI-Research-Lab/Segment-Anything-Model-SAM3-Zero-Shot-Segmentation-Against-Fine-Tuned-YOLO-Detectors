#!/bin/bash
###############################################################################
# Convert Instance Masks to YOLO Polygon Format
#
# Converts instance segmentation masks (where each object has a unique pixel 
# value) to YOLO polygon format (separate polygon annotation per instance).
#
# Usage:
#   ./convert_masks.sh minneapple
#   ./convert_masks.sh weedsgalore
#   ./convert_masks.sh both
#
# This must be run BEFORE training YOLO11 models.
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

if [[ "$DATASET" != "minneapple" && "$DATASET" != "weedsgalore" && "$DATASET" != "both" ]]; then
    print_error "Invalid dataset: $DATASET"
    echo "Usage: $0 [minneapple|weedsgalore|both]"
    exit 1
fi

clear
print_header "Instance Mask to YOLO Polygon Conversion"

convert_dataset() {
    local dataset=$1
    
    print_info "Converting $dataset masks to YOLO polygon format..."
    echo
    
    if [ ! -d "data/$dataset" ]; then
        print_error "Dataset not found: data/$dataset"
        print_info "Download dataset first:"
        echo "    ./datasets.sh $dataset"
        return 1
    fi
    
    python3 models/yolo11/convert_masks_to_yolo.py \
        --dataset "$dataset" \
        --class-id 0
    
    if [ $? -eq 0 ]; then
        print_success "Conversion complete for $dataset"
    else
        print_error "Conversion failed for $dataset"
        return 1
    fi
}

if [[ "$DATASET" == "both" ]]; then
    convert_dataset "minneapple"
    echo
    convert_dataset "weedsgalore"
else
    convert_dataset "$DATASET"
fi

echo
print_header "Done!"
echo
print_info "Next steps:"
echo "  1. Train YOLO11 model: ./ft_yolo11.sh $DATASET"
echo "  2. Run predictions: ./prediction_yolo11.sh $DATASET"
