#!/bin/bash
###############################################################################
# Dataset Management Script
# 
# Downloads and prepares MinneApple and/or WeedsGalore datasets for training
# and evaluation. Handles data validation and organization.
#
# Usage:
# chmod +x *.sh
#   ./datasets.sh minneapple     # Download MinneApple only
#   ./datasets.sh weedsgalore    # Download WeedsGalore only
#   ./datasets.sh both           # Download both datasets
#   ./datasets.sh check          # Check dataset status
#
# Output:
#   - data/minneapple/{train,val,test}/{images,masks}
#   - data/weedsgalore/{train,val,test}/{images,masks}
#
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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
    echo -e "${CYAN}INFO: $1${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

###############################################################################
# Parse Arguments
###############################################################################

DATASET_ARG="${1:-both}"

# Validate argument
if [[ "$DATASET_ARG" != "minneapple" && "$DATASET_ARG" != "weedsgalore" && "$DATASET_ARG" != "both" && "$DATASET_ARG" != "check" ]]; then
    print_error "Invalid argument: $DATASET_ARG"
    echo ""
    echo "Usage: $0 [minneapple|weedsgalore|both|check]"
    echo ""
    echo "Commands:"
    echo "  minneapple   - Download MinneApple dataset only"
    echo "  weedsgalore  - Download WeedsGalore dataset only"
    echo "  both         - Download both datasets"
    echo "  check        - Check status of existing datasets"
    echo ""
    exit 1
fi

###############################################################################
# Dataset Status Check Function
###############################################################################

check_dataset_status() {
    local dataset=$1
    local data_dir="data/$dataset"
    
    echo ""
    print_info "Checking dataset: $dataset"
    echo "  Location: $data_dir"
    
    if [ ! -d "$data_dir" ]; then
        echo "  Status: ${RED}Not downloaded${NC}"
        return 1
    fi
    
    # Check splits
    local all_splits_exist=true
    for split in train val test; do
        if [ -d "$data_dir/$split/images" ] && [ -d "$data_dir/$split/masks" ]; then
            local img_count=$(find "$data_dir/$split/images" -name "*.png" -o -name "*.jpg" | wc -l | tr -d ' ')
            local mask_count=$(find "$data_dir/$split/masks" -name "*.png" | wc -l | tr -d ' ')
            echo "  $split: ${GREEN}OK${NC} ($img_count images, $mask_count masks)"
        else
            echo "  $split: ${RED}MISSING${NC}"
            all_splits_exist=false
        fi
    done
    
    # Check COCO format (for SAM3)
    if [ -d "$data_dir/../sam3_coco/$dataset" ]; then
        echo "  SAM3 COCO format: ${GREEN}Available${NC}"
    else
        echo "  SAM3 COCO format: ${YELLOW}Not converted${NC}"
    fi
    
    # Overall status
    if [ "$all_splits_exist" = true ]; then
        local total_size=$(du -sh "$data_dir" 2>/dev/null | cut -f1)
        echo "  Total size: $total_size"
        echo "  Status: ${GREEN}Complete${NC}"
        return 0
    else
        echo "  Status: ${YELLOW}Incomplete${NC}"
        return 1
    fi
}

###############################################################################
# Check Mode
###############################################################################

if [ "$DATASET_ARG" = "check" ]; then
    clear
    print_header "Dataset Status Check"
    
    echo ""
    print_info "Checking all datasets in: data/"
    
    MINNEAPPLE_STATUS=1
    WEEDSGALORE_STATUS=1
    
    check_dataset_status "minneapple"
    MINNEAPPLE_STATUS=$?
    
    check_dataset_status "weedsgalore"
    WEEDSGALORE_STATUS=$?
    
    echo ""
    print_header "Summary"
    echo ""
    
    if [ $MINNEAPPLE_STATUS -eq 0 ] && [ $WEEDSGALORE_STATUS -eq 0 ]; then
        print_success "All datasets are ready!"
    else
        print_warning "Some datasets are missing or incomplete"
        echo ""
        echo "Download missing datasets with:"
        [ $MINNEAPPLE_STATUS -ne 0 ] && echo "  ./datasets.sh minneapple"
        [ $WEEDSGALORE_STATUS -ne 0 ] && echo "  ./datasets.sh weedsgalore"
        echo "  ./datasets.sh both"
    fi
    
    echo ""
    exit 0
fi

###############################################################################
# Phase 1: System Check
###############################################################################

clear
print_header "Dataset Download & Preparation"
echo "This script downloads and prepares datasets for training and evaluation"
echo ""

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

# Check required Python packages
print_info "Checking Python dependencies..."
MISSING_PACKAGES=()

for package in requests pillow opencv-python numpy tqdm; do
    if python3 -c "import ${package//-/_}" 2>/dev/null; then
        print_success "$package installed"
    else
        print_warning "$package not installed"
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    print_warning "Installing missing packages: ${MISSING_PACKAGES[*]}"
    pip install ${MISSING_PACKAGES[*]}
    print_success "Dependencies installed"
fi

echo ""

###############################################################################
# Phase 2: Determine Datasets to Download
###############################################################################

print_header "Phase 2: Dataset Selection"

if [ "$DATASET_ARG" = "both" ]; then
    DATASETS=("minneapple" "weedsgalore")
    print_info "Processing both datasets: MinneApple and WeedsGalore"
else
    DATASETS=("$DATASET_ARG")
    print_info "Processing single dataset: $DATASET_ARG"
fi

echo ""

# Check if datasets already exist
DATASETS_TO_DOWNLOAD=()
for dataset in "${DATASETS[@]}"; do
    if check_dataset_status "$dataset"; then
        echo ""
        read -p "  Dataset exists. Re-download? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            DATASETS_TO_DOWNLOAD+=("$dataset")
        else
            print_info "Skipping $dataset (already complete)"
        fi
    else
        DATASETS_TO_DOWNLOAD+=("$dataset")
    fi
done

if [ ${#DATASETS_TO_DOWNLOAD[@]} -eq 0 ]; then
    echo ""
    print_success "All requested datasets are already available!"
    print_info "Use './datasets.sh check' to verify dataset status"
    echo ""
    exit 0
fi

echo ""

###############################################################################
# Phase 3: Download and Prepare Datasets
###############################################################################

print_header "Phase 3: Download and Prepare Datasets"

DOWNLOAD_SUMMARY=()

for dataset in "${DATASETS_TO_DOWNLOAD[@]}"; do
    echo ""
    print_header "Processing: $dataset"
    echo ""
    
    START_TIME=$(date +%s)
    
    print_info "Downloading and preparing $dataset dataset..."
    print_info "This may take several minutes depending on your connection..."
    echo ""
    
    # Run download script
    if python3 src/download_dataset.py --dataset "$dataset" --yes; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        print_success "Dataset prepared: $dataset (${DURATION}s)"
        
        # Verify download
        DATA_DIR="data/$dataset"
        if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/val" ] && [ -d "$DATA_DIR/test" ]; then
            # Count files
            TRAIN_COUNT=$(find "$DATA_DIR/train/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
            VAL_COUNT=$(find "$DATA_DIR/val/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
            TEST_COUNT=$(find "$DATA_DIR/test/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
            TOTAL_COUNT=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))
            
            DATASET_SIZE=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
            
            print_success "Verification complete"
            echo "  Train: $TRAIN_COUNT images"
            echo "  Val:   $VAL_COUNT images"
            echo "  Test:  $TEST_COUNT images"
            echo "  Total: $TOTAL_COUNT images"
            echo "  Size:  $DATASET_SIZE"
            
            DOWNLOAD_SUMMARY+=("$dataset: OK ($TOTAL_COUNT images, $DATASET_SIZE, ${DURATION}s)")
        else
            print_warning "Dataset downloaded but directory structure incomplete"
            DOWNLOAD_SUMMARY+=("$dataset: Incomplete")
        fi
    else
        print_error "Failed to download $dataset dataset"
        DOWNLOAD_SUMMARY+=("$dataset: Failed")
    fi
    
    echo ""
done

###############################################################################
# Phase 4: Summary Report
###############################################################################

print_header "Phase 4: Download Summary"

SUMMARY_FILE="data/DATASET_DOWNLOAD_SUMMARY_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "============================================================================"
    echo "DATASET DOWNLOAD SUMMARY"
    echo "============================================================================"
    echo ""
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Python: $PYTHON_VERSION"
    echo ""
    echo "Datasets Processed:"
    echo "-------------------"
    
    for summary in "${DOWNLOAD_SUMMARY[@]}"; do
        echo "  $summary"
    done
    
    echo ""
    echo "Dataset Locations:"
    echo "------------------"
    
    for dataset in "${DATASETS[@]}"; do
        if [ -d "data/$dataset" ]; then
            DATASET_SIZE=$(du -sh "data/$dataset" 2>/dev/null | cut -f1)
            echo "  $dataset: data/$dataset/ ($DATASET_SIZE)"
            
            # Count splits
            for split in train val test; do
                if [ -d "data/$dataset/$split/images" ]; then
                    IMG_COUNT=$(find "data/$dataset/$split/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
                    echo "    - $split: $IMG_COUNT images"
                fi
            done
        fi
    done
    
    echo ""
    echo "============================================================================"
    echo "Next Steps:"
    echo "============================================================================"
    echo ""
    echo "Check dataset status:"
    echo "  ./datasets.sh check"
    echo ""
    echo "Train models:"
    echo "  ./ft_yolo11.sh ${DATASET_ARG}"
    echo "  ./ft_sam3_full.sh ${DATASET_ARG}"
    echo "  ./ft_sam3_lora.sh ${DATASET_ARG}"
    echo ""
    echo "Run predictions:"
    echo "  ./prediction_yolo11.sh ${DATASET_ARG}"
    echo "  ./prediction_sam3_zero.sh ${DATASET_ARG}"
    echo "  ./prediction_sam3_full.sh ${DATASET_ARG}"
    echo "  ./prediction_sam3_lora.sh ${DATASET_ARG}"
    echo ""
    echo "============================================================================"
    
} | tee "$SUMMARY_FILE"

echo ""
print_success "Summary saved to: $SUMMARY_FILE"

###############################################################################
# Complete
###############################################################################

echo ""
print_header "Dataset Preparation Complete!"

echo ""
print_info "Available Datasets:"
for dataset in "${DATASETS[@]}"; do
    if [ -d "data/$dataset" ]; then
        echo "  - $dataset"
    fi
done

echo ""
print_info "Use './datasets.sh check' to verify dataset status"

echo ""
print_header "Done!"
echo ""
