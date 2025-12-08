#!/bin/bash

################################################################################
# SAM3 Unified Deployment Script
# 
# Purpose: Single-command deployment for both MinneApple and WeedsGalore datasets
# 
# This script will:
#   1. Check system requirements (GPU, Python, disk space)
#   2. Install all dependencies
#   3. Download and prepare selected dataset(s)
#   4. Setup Hugging Face authentication
#   5. Run SAM3 segmentation on test split
#   6. Perform comprehensive evaluation with ground truth masks
#   7. Generate LaTeX-ready reports for papers
#
# Usage:
#   ./prediction_sam3_zero.sh minneapple
#   ./prediction_sam3_zero.sh weedsgalore
#   ./prediction_sam3_zero.sh both
#
# Requirements:
#   - Ubuntu 20.04+ with NVIDIA GPU (or macOS for development)
#   - At least 50GB free disk space
#   - Internet connection
#
# Example:
# chmod +x prediction_sam3_zero.sh
# # Process BOTH datasets
# export HF_TOKEN="" && ./prediction_sam3_zero.sh both
# # Process MinneApple ONLY
# export HF_TOKEN="" && ./prediction_sam3_zero.sh minneapple
# # Process WeedsGalore ONLY
# export HF_TOKEN="" && ./prediction_sam3_zero.sh weedsgalore
################################################################################

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/prediction_sam3_zero.sh_$(date +%Y%m%d_%H%M%S).log"

# Dataset selection (will be set from command line argument)
DATASET_MODE=""
DATASETS_TO_PROCESS=()

# Create log directory
mkdir -p "${LOG_DIR}"

################################################################################
# Logging functions
################################################################################

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*" | tee -a "${LOG_FILE}"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $*" | tee -a "${LOG_FILE}"
}

print_header() {
    echo ""
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo "$1" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo ""
}

################################################################################
# Phase 1: System Check
################################################################################

check_system() {
    print_header "PHASE 1: System Requirements Check"
    
    # Check OS
    log "Checking operating system..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS_NAME="macOS"
        OS_VERSION=$(sw_vers -productVersion)
        log "SUCCESS: OS: ${OS_NAME} ${OS_VERSION}"
    elif [[ -f /etc/os-release ]]; then
        OS_NAME=$(grep "^NAME=" /etc/os-release | cut -d'"' -f2)
        OS_VERSION=$(grep "^VERSION=" /etc/os-release | cut -d'"' -f2)
        log "SUCCESS: OS: ${OS_NAME} ${OS_VERSION}"
    else
        log_warning "Cannot detect OS"
    fi
    
    # Check GPU
    log "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
        log "SUCCESS: GPU detected: ${GPU_INFO}"
        
        # Get GPU memory
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if (( GPU_MEM < 16000 )); then
            log_warning "GPU memory is ${GPU_MEM}MB. Recommended: 24GB+ for optimal performance."
        fi
    else
        log_warning "No NVIDIA GPU detected. SAM3 will run slower on CPU."
    fi
    
    # Check Python
    log "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.8+."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version)
    log "SUCCESS: ${PYTHON_VERSION}"
    
    # Check disk space
    log "Checking disk space..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        AVAILABLE_SPACE=$(df -g "${PROJECT_DIR}" | tail -1 | awk '{print $4}')
    else
        AVAILABLE_SPACE=$(df -BG "${PROJECT_DIR}" | tail -1 | awk '{print $4}' | sed 's/G//')
    fi
    log "Available disk space: ${AVAILABLE_SPACE}GB"
    
    if (( AVAILABLE_SPACE < 50 )); then
        log_warning "Low disk space: ${AVAILABLE_SPACE}GB. Recommended: 50GB+"
    else
        log "SUCCESS: Disk space check passed (${AVAILABLE_SPACE}GB available)"
    fi
    
    # Check internet connection
    log "Checking internet connection..."
    if curl -s --connect-timeout 5 https://www.google.com > /dev/null 2>&1; then
        log "SUCCESS: Internet connection active"
    else
        log_error "No internet connection detected"
        exit 1
    fi
    
    log "SUCCESS: All system requirements met"
}

################################################################################
# Phase 2: Install Dependencies
################################################################################

install_dependencies() {
    print_header "PHASE 2: Installing Dependencies"
    
    # Install system packages (Linux only)
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log "Installing system dependencies..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            build-essential \
            libgl1 \
            libglib2.0-0 \
            wget \
            curl \
            git \
            unzip \
            > /dev/null 2>&1
        log "SUCCESS: System packages installed"
    fi
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
        log "SUCCESS: Virtual environment created"
    fi
    
    # Activate virtual environment
    log "Activating virtual environment..."
    source venv/bin/activate
    log "SUCCESS: Virtual environment activated"
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip -q
    log "SUCCESS: pip upgraded"
    
    # Install Python requirements
    log "Installing Python dependencies..."
    log_info "This may take 5-10 minutes on first run..."
    
    # Install PyTorch with CUDA support (or CPU for macOS)
    if command -v nvidia-smi &> /dev/null; then
        log "Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    else
        log "Installing PyTorch (CPU version)..."
        pip install torch torchvision torchaudio -q
    fi
    log "SUCCESS: PyTorch installed"
    
    # Install transformers from GitHub (required for SAM3 support)
    log "Installing transformers from GitHub for SAM3 support..."
    pip install git+https://github.com/huggingface/transformers.git -q
    log "SUCCESS: Transformers installed with SAM3 support"
    
    # Install other requirements
    log "Installing other dependencies..."
    pip install -r requirements.txt -q
    log "SUCCESS: All Python dependencies installed"
    
    # Verify PyTorch
    log "Verifying PyTorch installation..."
    if command -v nvidia-smi &> /dev/null; then
        python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'SUCCESS: PyTorch CUDA available: {torch.cuda.get_device_name(0)}')" | tee -a "${LOG_FILE}"
    else
        python3 -c "import torch; print('SUCCESS: PyTorch installed (CPU mode)')" | tee -a "${LOG_FILE}"
    fi
}

################################################################################
# Phase 3: Validate Datasets
################################################################################

validate_datasets() {
    print_header "PHASE 3: Validating Dataset(s)"
    
    log "Dataset mode: ${DATASET_MODE}"
    
    # Verify datasets exist
    for dataset in "${DATASETS_TO_PROCESS[@]}"; do
        DATA_DIR="${PROJECT_DIR}/data/${dataset}"
        
        if [[ ! -d "${DATA_DIR}/train" ]] || [[ ! -d "${DATA_DIR}/val" ]] || [[ ! -d "${DATA_DIR}/test" ]]; then
            log_error "Dataset not found: ${dataset}"
            log_error "Please download datasets first: ./datasets.sh ${DATASET_MODE}"
            exit 1
        fi
        
        TRAIN_COUNT=$(find "${DATA_DIR}/train/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
        VAL_COUNT=$(find "${DATA_DIR}/val/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
        TEST_COUNT=$(find "${DATA_DIR}/test/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
        TOTAL_COUNT=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))
        
        log "SUCCESS: ${dataset} dataset ready:"
        log "  Train: ${TRAIN_COUNT} images"
        log "  Val:   ${VAL_COUNT} images"
        log "  Test:  ${TEST_COUNT} images"
        log "  Total: ${TOTAL_COUNT} images"
    done
}

################################################################################
# Phase 4: Hugging Face Authentication
################################################################################

setup_huggingface() {
    print_header "PHASE 4: Hugging Face Authentication"
    
    # Check if token is provided via environment variable
    if [[ -n "${HF_TOKEN}" ]]; then
        log "Using Hugging Face token from HF_TOKEN environment variable..."
        python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>&1 | tee -a "${LOG_FILE}"
        
        if python3 -c "from huggingface_hub import whoami; whoami()" &> /dev/null; then
            HF_USER=$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null)
            log "SUCCESS: Successfully authenticated as: ${HF_USER}"
            return 0
        else
            log_error "Failed to authenticate with provided token"
            exit 1
        fi
    fi
    
    # Check if already logged in
    if python3 -c "from huggingface_hub import whoami; whoami()" &> /dev/null; then
        HF_USER=$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null || echo "unknown")
        log "SUCCESS: Already logged into Hugging Face as: ${HF_USER}"
        return 0
    fi
    
    log "Hugging Face authentication required for SAM3 model access"
    log ""
    log "Please follow these steps:"
    log "  1. Visit: https://huggingface.co/settings/tokens"
    log "  2. Create a token with 'read' access"
    log "  3. Accept SAM3 license at: https://huggingface.co/facebook/sam3"
    log ""
    log_warning "The script will now pause for Hugging Face login..."
    log ""
    
    # Login interactively using Python
    python3 -c "from huggingface_hub import login; login()"
    
    # Verify login
    if python3 -c "from huggingface_hub import whoami; whoami()" &> /dev/null; then
        HF_USER=$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")
        log "SUCCESS: Successfully logged in as: ${HF_USER}"
    else
        log_error "Hugging Face login failed"
        exit 1
    fi
}

################################################################################
# Phase 5: Run SAM3 Segmentation
################################################################################

run_segmentation() {
    print_header "PHASE 5: Running SAM3 Segmentation on Test Sets"
    
    for dataset in "${DATASETS_TO_PROCESS[@]}"; do
        DATA_DIR="${PROJECT_DIR}/data/${dataset}"
        RESULTS_DIR="${PROJECT_DIR}/results/sam3_${dataset}_$(date +%Y%m%d_%H%M%S)"
        
        # Determine text prompt based on dataset
        if [[ "${dataset}" == "minneapple" ]]; then
            TEXT_PROMPT="apple"
        elif [[ "${dataset}" == "weedsgalore" ]]; then
            TEXT_PROMPT="weed"
        else
            TEXT_PROMPT="object"
        fi
        
        log ""
        log "Processing ${dataset} dataset..."
        log "  Text prompt: '${TEXT_PROMPT}'"
        log "  Test set: ${DATA_DIR}/test"
        log "  Output: ${RESULTS_DIR}"
        log ""
        
        # Create results directory
        mkdir -p "${RESULTS_DIR}"
        
        # Run segmentation with comprehensive evaluation
        START_TIME=$(date +%s)
        
        python3 models/sam3/predict_sam3_zero.py \
            --data-dir "${DATA_DIR}/test" \
            --output-dir "${RESULTS_DIR}" \
            --text-prompt "${TEXT_PROMPT}" \
            --full-dataset \
            --comprehensive-evaluation \
            --generate-latex \
            --gt-dir "${DATA_DIR}/test/masks" \
            2>&1 | tee -a "${LOG_FILE}"
        
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))
        
        log "SUCCESS: ${dataset} segmentation completed in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
        
        # Count results
        VIZ_COUNT=$(find "${RESULTS_DIR}/visualizations" -name "*.png" 2>/dev/null | wc -l)
        MASK_COUNT=$(find "${RESULTS_DIR}/masks" -name "*.npy" 2>/dev/null | wc -l)
        
        log "  Results:"
        log "    - Visualizations: ${VIZ_COUNT}"
        log "    - Masks saved: ${MASK_COUNT}"
        log "    - Results directory: ${RESULTS_DIR}"
        
        # Create zip archive of results
        log ""
        log "Creating zip archive..."
        RESULTS_ZIP="${PROJECT_DIR}/results/$(basename ${RESULTS_DIR}).zip"
        cd "${PROJECT_DIR}/results" && zip -r "$(basename ${RESULTS_ZIP})" "$(basename ${RESULTS_DIR})" > /dev/null 2>&1
        cd "${PROJECT_DIR}"
        
        if [[ -f "${RESULTS_ZIP}" ]]; then
            ZIP_SIZE=$(du -h "${RESULTS_ZIP}" | cut -f1)
            log "SUCCESS: Results archived: $(basename ${RESULTS_ZIP}) (${ZIP_SIZE})"
        else
            log_warning "Failed to create zip archive"
        fi
        
        # Store results directory for summary
        if [[ "${dataset}" == "minneapple" ]]; then
            MINNEAPPLE_RESULTS="${RESULTS_DIR}"
        elif [[ "${dataset}" == "weedsgalore" ]]; then
            WEEDSGALORE_RESULTS="${RESULTS_DIR}"
        fi
    done
}

################################################################################
# Phase 6: Generate Final Report
################################################################################

generate_report() {
    print_header "PHASE 6: Generating Comprehensive Reports"
    
    log "Generating final reports for all processed datasets..."
    
    for dataset in "${DATASETS_TO_PROCESS[@]}"; do
        if [[ "${dataset}" == "minneapple" ]]; then
            RESULTS_DIR="${MINNEAPPLE_RESULTS}"
        elif [[ "${dataset}" == "weedsgalore" ]]; then
            RESULTS_DIR="${WEEDSGALORE_RESULTS}"
        fi
        
        log ""
        log "${dataset} results:"
        
        # Check for generated files
        if [[ -f "${RESULTS_DIR}/report/report.html" ]]; then
            log "  SUCCESS: HTML report: ${RESULTS_DIR}/report/report.html"
        fi
        
        if [[ -f "${RESULTS_DIR}/latex/results_section.tex" ]]; then
            log "  SUCCESS: LaTeX report: ${RESULTS_DIR}/latex/results_section.tex"
        fi
        
        if [[ -f "${RESULTS_DIR}/metrics/evaluation_metrics.csv" ]]; then
            log "  SUCCESS: Metrics CSV: ${RESULTS_DIR}/metrics/evaluation_metrics.csv"
        fi
    done
    
    # Create deployment summary
    create_deployment_summary
}

################################################################################
# Create Deployment Summary
################################################################################

create_deployment_summary() {
    SUMMARY_FILE="${PROJECT_DIR}/DEPLOYMENT_SUMMARY_SAM3.txt"
    
    cat > "${SUMMARY_FILE}" << EOF
================================================================================
SAM3 Unified Deployment Summary
================================================================================

Deployment Date: $(date)
Deployment Log: ${LOG_FILE}
Dataset Mode: ${DATASET_MODE}

System Information:
-------------------
EOF
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "OS: macOS $(sw_vers -productVersion)" >> "${SUMMARY_FILE}"
    else
        echo "OS: $(grep "^NAME=" /etc/os-release | cut -d'"' -f2)" >> "${SUMMARY_FILE}"
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "${SUMMARY_FILE}"
        echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)" >> "${SUMMARY_FILE}"
    else
        echo "GPU: None (CPU mode)" >> "${SUMMARY_FILE}"
    fi
    
    echo "Python: $(python3 --version)" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
    
    for dataset in "${DATASETS_TO_PROCESS[@]}"; do
        DATA_DIR="${PROJECT_DIR}/data/${dataset}"
        
        if [[ "${dataset}" == "minneapple" ]]; then
            RESULTS_DIR="${MINNEAPPLE_RESULTS}"
            DATASET_NAME="MinneApple Detection Dataset"
            TEXT_PROMPT="apple"
        elif [[ "${dataset}" == "weedsgalore" ]]; then
            RESULTS_DIR="${WEEDSGALORE_RESULTS}"
            DATASET_NAME="WeedsGalore Multi-spectral Weed Segmentation Dataset"
            TEXT_PROMPT="weed"
        fi
        
        cat >> "${SUMMARY_FILE}" << DATASETEOF
${DATASET_NAME}:
--------------------
Train Images: $(find "${DATA_DIR}/train/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
Val Images: $(find "${DATA_DIR}/val/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
Test Images: $(find "${DATA_DIR}/test/images" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
Location: ${DATA_DIR}
Evaluation Split: test
Text Prompt: '${TEXT_PROMPT}'

Results Location: ${RESULTS_DIR}
- Visualizations: ${RESULTS_DIR}/visualizations/
- Masks: ${RESULTS_DIR}/masks/
- Metrics: ${RESULTS_DIR}/metrics/
- Reports: ${RESULTS_DIR}/report/
- LaTeX: ${RESULTS_DIR}/latex/

DATASETEOF
    done
    
    cat >> "${SUMMARY_FILE}" << EOF

Next Steps:
-----------
1. View HTML reports in browser
2. Copy LaTeX reports to your paper
3. Review evaluation metrics (IoU, F1-score, precision, recall)
4. Compare results between datasets (if both processed)
5. Download results if on remote server:
   scp -r user@server:${PROJECT_DIR}/results ./local_results/

================================================================================
EOF

    log "SUCCESS: Deployment summary saved to: ${SUMMARY_FILE}"
    cat "${SUMMARY_FILE}" | tee -a "${LOG_FILE}"
}

################################################################################
# Main Execution
################################################################################

main() {
    # Parse command line argument
    if [[ $# -eq 0 ]]; then
        echo "Usage: $0 {minneapple|weedsgalore|both}"
        echo ""
        echo "Examples:"
        echo "  $0 minneapple      # Process MinneApple dataset only"
        echo "  $0 weedsgalore     # Process WeedsGalore dataset only"
        echo "  $0 both            # Process both datasets"
        exit 1
    fi
    
    DATASET_MODE="$1"
    
    # Validate argument
    case "${DATASET_MODE}" in
        minneapple)
            DATASETS_TO_PROCESS=("minneapple")
            ;;
        weedsgalore)
            DATASETS_TO_PROCESS=("weedsgalore")
            ;;
        both)
            DATASETS_TO_PROCESS=("minneapple" "weedsgalore")
            ;;
        *)
            echo "Error: Invalid argument '${DATASET_MODE}'"
            echo "Usage: $0 {minneapple|weedsgalore|both}"
            exit 1
            ;;
    esac
    
    print_header "SAM3 Unified Deployment - ${DATASET_MODE}"
    
    log "Starting automated deployment..."
    log "Project directory: ${PROJECT_DIR}"
    log "Log file: ${LOG_FILE}"
    log "Dataset(s): ${DATASETS_TO_PROCESS[*]}"
    log ""
    
    # Execute phases
    check_system
    install_dependencies
    validate_datasets
    setup_huggingface
    run_segmentation
    generate_report
    
    # Final message
    print_header "DEPLOYMENT COMPLETED SUCCESSFULLY!"
    
    log ""
    log "All operations completed successfully!"
    log ""
    
    for dataset in "${DATASETS_TO_PROCESS[@]}"; do
        if [[ "${dataset}" == "minneapple" ]]; then
            RESULTS_DIR="${MINNEAPPLE_RESULTS}"
        elif [[ "${dataset}" == "weedsgalore" ]]; then
            RESULTS_DIR="${WEEDSGALORE_RESULTS}"
        fi
        
        log "${dataset} results: ${RESULTS_DIR}"
        log "HTML report: ${RESULTS_DIR}/report/report.html"
        log "LaTeX report: ${RESULTS_DIR}/latex/results_section.tex"
        log ""
        
        # Display quick stats if available
        if [[ -f "${RESULTS_DIR}/metrics/evaluation_metrics.json" ]]; then
            log "Quick Stats (${dataset}):"
            python3 -c "
import json
with open('${RESULTS_DIR}/metrics/evaluation_metrics.json') as f:
    data = json.load(f)
    print(f'  Total Images: {data.get(\"num_images\", \"N/A\")}')
    print(f'  Total Detections: {data.get(\"total_predictions\", \"N/A\")}')
    print(f'  Mean IoU: {data.get(\"mean_image_iou\", 0):.3f}')
    print(f'  F1-Score: {data.get(\"dataset_f1\", 0):.3f}')
    print(f'  Precision: {data.get(\"precision\", 0):.3f}')
    print(f'  Recall: {data.get(\"recall\", 0):.3f}')
" 2>/dev/null | tee -a "${LOG_FILE}"
            log ""
        fi
    done
    
    log "Full deployment log: ${LOG_FILE}"
    log "Summary: ${PROJECT_DIR}/DEPLOYMENT_SUMMARY_SAM3.txt"
    log ""
    log "Ready for your paper!"
    log ""
}

# Run main function
main "$@"
