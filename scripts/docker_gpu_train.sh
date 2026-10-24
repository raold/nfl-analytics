#!/usr/bin/env bash
# docker_gpu_train.sh - GPU Training Orchestration Script for Docker
# Purpose: Sequential execution of GPU training jobs with monitoring and reporting
# Usage: bash scripts/docker_gpu_train.sh [--phase N] [--job JOB_NAME] [--all]

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/training"
MODELS_DIR="$PROJECT_ROOT/models"
REPORT_FILE="$LOG_DIR/training_report_$(date +%Y%m%d_%H%M%S).txt"

# Create log directory
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date +%Y-%m-%d\ %H:%M:%S) - $*" | tee -a "$REPORT_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date +%Y-%m-%d\ %H:%M:%S) - $*" | tee -a "$REPORT_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date +%Y-%m-%d\ %H:%M:%S) - $*" | tee -a "$REPORT_FILE" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +%Y-%m-%d\ %H:%M:%S) - $*" | tee -a "$REPORT_FILE"
}

print_header() {
    echo ""
    echo "================================================================================" | tee -a "$REPORT_FILE"
    echo "$1" | tee -a "$REPORT_FILE"
    echo "================================================================================" | tee -a "$REPORT_FILE"
    echo ""
}

check_gpu_available() {
    log_info "Checking GPU availability..."
    if docker exec nfl-analytics-gpu-trainer nvidia-smi &>/dev/null; then
        log_success "GPU is accessible in container"
        docker exec nfl-analytics-gpu-trainer nvidia-smi | tee -a "$REPORT_FILE"
        return 0
    else
        log_error "GPU is not accessible in container"
        return 1
    fi
}

get_gpu_memory() {
    docker exec nfl-analytics-gpu-trainer nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1 "/" $3 " GB"}'
}

get_gpu_utilization() {
    docker exec nfl-analytics-gpu-trainer nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
}

wait_for_cooldown() {
    local seconds=${1:-30}
    log_info "Cooling down GPU for $seconds seconds..."
    sleep "$seconds"
}

# ============================================================================
# Training Job Definitions
# ============================================================================

run_bnn_multioutput() {
    local job_name="BNN Multi-Output"
    local start_time=$(date +%s)

    print_header "$job_name Training"
    log_info "Starting $job_name training..."
    log_info "GPU Memory: $(get_gpu_memory), Utilization: $(get_gpu_utilization)%"

    local log_file="$LOG_DIR/bnn_multioutput_$(date +%Y%m%d_%H%M%S).log"

    if docker exec nfl-analytics-gpu-trainer \
        python nfl_analytics/models/bnn_multioutput_pytorch.py \
        --device cuda \
        --epochs 500 \
        --batch-size 256 \
        --learning-rate 0.001 \
        --checkpoint-dir models/bayesian/bnn_multioutput_checkpoints \
        --log-dir logs/training \
        --output-dir models/bayesian 2>&1 | tee "$log_file"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))

        log_success "$job_name completed in ${hours}h ${minutes}m"
        log_info "Final GPU Memory: $(get_gpu_memory), Utilization: $(get_gpu_utilization)%"

        # Check for model files
        if docker exec nfl-analytics-gpu-trainer test -f models/bayesian/bnn_multioutput_v1_final.pt; then
            log_success "Model saved: models/bayesian/bnn_multioutput_v1_final.pt"
        else
            log_warning "Expected model file not found"
        fi

        return 0
    else
        log_error "$job_name training failed"
        return 1
    fi
}

run_xgboost_gpu() {
    local job_name="XGBoost GPU v3"
    local start_time=$(date +%s)

    print_header "$job_name Training"
    log_info "Starting $job_name training..."
    log_info "GPU Memory: $(get_gpu_memory), Utilization: $(get_gpu_utilization)%"

    local log_file="$LOG_DIR/xgboost_gpu_$(date +%Y%m%d_%H%M%S).log"

    if docker exec nfl-analytics-gpu-trainer \
        python nfl_analytics/models/xgboost_gpu_v3.py \
        --features-csv data/processed/features/asof_team_features_v3.csv \
        --start-season 2006 \
        --end-season 2021 \
        --test-seasons 2022 2023 2024 2025 \
        --sweep \
        --output-dir models/xgboost/v3_sweep \
        --device cuda 2>&1 | tee "$log_file"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))

        log_success "$job_name completed in ${hours}h ${minutes}m"
        log_info "Final GPU Memory: $(get_gpu_memory), Utilization: $(get_gpu_utilization)%"

        # Check for sweep results
        if docker exec nfl-analytics-gpu-trainer test -f models/xgboost/v3_sweep/sweep_results.csv; then
            log_success "Sweep results saved: models/xgboost/v3_sweep/sweep_results.csv"
        else
            log_warning "Expected sweep results file not found"
        fi

        return 0
    else
        log_error "$job_name training failed"
        return 1
    fi
}

run_hierarchical_bayesian() {
    local job_name="Hierarchical Bayesian (R + Stan)"
    local start_time=$(date +%s)

    print_header "$job_name Training"
    log_info "Starting $job_name training..."
    log_info "Note: Stan uses CPU, GPU memory should be free"

    local log_file="$LOG_DIR/hierarchical_bayesian_$(date +%Y%m%d_%H%M%S).log"

    if docker exec nfl-analytics-gpu-trainer \
        Rscript R/train_and_save_passing_model.R 2>&1 | tee "$log_file"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))

        log_success "$job_name completed in ${hours}h ${minutes}m"

        # Check for model files
        if docker exec nfl-analytics-gpu-trainer test -f models/bayesian/passing_yards_hierarchical_v1.rds; then
            log_success "Model saved: models/bayesian/passing_yards_hierarchical_v1.rds"
        else
            log_warning "Expected model file not found"
        fi

        return 0
    else
        log_error "$job_name training failed"
        return 1
    fi
}

run_ensemble_creation() {
    local job_name="Ensemble Creation"
    local start_time=$(date +%s)

    print_header "$job_name"
    log_info "Creating ensemble from trained models..."

    local log_file="$LOG_DIR/ensemble_creation_$(date +%Y%m%d_%H%M%S).log"

    if docker exec nfl-analytics-gpu-trainer \
        python nfl_analytics/models/create_final_ensemble.py \
        --bayesian models/bayesian/bnn_multioutput_v1_final.pt \
        --xgboost models/xgboost/v3_sweep/best_model.json \
        --hierarchical models/bayesian/passing_yards_hierarchical_v1.rds \
        --output models/ensemble/v3_final.pkl 2>&1 | tee "$log_file"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))

        log_success "$job_name completed in ${minutes}m"
        return 0
    else
        log_error "$job_name failed"
        return 1
    fi
}

# ============================================================================
# Phase Execution
# ============================================================================

run_phase_1() {
    # Phase 1: Critical Models (BNN + XGBoost)
    print_header "PHASE 1: Critical Models"
    log_info "Estimated time: 5-7 hours"

    local jobs_completed=0
    local jobs_failed=0

    # Job 1: BNN Multi-Output
    if run_bnn_multioutput; then
        ((jobs_completed++))
        wait_for_cooldown 30
    else
        ((jobs_failed++))
    fi

    # Job 2: XGBoost GPU
    if run_xgboost_gpu; then
        ((jobs_completed++))
    else
        ((jobs_failed++))
    fi

    print_header "PHASE 1 SUMMARY"
    log_info "Jobs completed: $jobs_completed"
    log_info "Jobs failed: $jobs_failed"

    return $jobs_failed
}

run_phase_2() {
    # Phase 2: Hierarchical Bayesian Models
    print_header "PHASE 2: Hierarchical Bayesian Models"
    log_info "Estimated time: 2-3 hours"

    local jobs_completed=0
    local jobs_failed=0

    # Job 1: Hierarchical Bayesian
    if run_hierarchical_bayesian; then
        ((jobs_completed++))
    else
        ((jobs_failed++))
    fi

    print_header "PHASE 2 SUMMARY"
    log_info "Jobs completed: $jobs_completed"
    log_info "Jobs failed: $jobs_failed"

    return $jobs_failed
}

run_phase_3() {
    # Phase 3: Ensemble Creation
    print_header "PHASE 3: Ensemble Creation"
    log_info "Estimated time: 5-10 minutes"

    local jobs_completed=0
    local jobs_failed=0

    # Job 1: Ensemble Creation
    if run_ensemble_creation; then
        ((jobs_completed++))
    else
        ((jobs_failed++))
    fi

    print_header "PHASE 3 SUMMARY"
    log_info "Jobs completed: $jobs_completed"
    log_info "Jobs failed: $jobs_failed"

    return $jobs_failed
}

run_all_phases() {
    print_header "RUNNING ALL TRAINING PHASES"
    log_info "Total estimated time: 7-10 hours"

    local total_start=$(date +%s)
    local phases_failed=0

    run_phase_1 || ((phases_failed++))
    run_phase_2 || ((phases_failed++))
    run_phase_3 || ((phases_failed++))

    local total_end=$(date +%s)
    local total_duration=$((total_end - total_start))
    local total_hours=$((total_duration / 3600))
    local total_minutes=$(((total_duration % 3600) / 60))

    print_header "ALL PHASES COMPLETE"
    log_info "Total training time: ${total_hours}h ${total_minutes}m"
    log_info "Phases failed: $phases_failed"
    log_info "Report saved: $REPORT_FILE"

    return $phases_failed
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    print_header "NFL Analytics GPU Training Orchestration"
    log_info "Script started at $(date)"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Report file: $REPORT_FILE"

    # Check if container is running
    if ! docker ps | grep -q nfl-analytics-gpu-trainer; then
        log_error "GPU training container is not running"
        log_info "Start with: docker-compose -f docker-compose.gpu.yml up -d"
        exit 1
    fi

    # Check GPU availability
    if ! check_gpu_available; then
        log_error "GPU not available, exiting"
        exit 1
    fi

    # Parse arguments
    local phase=""
    local job=""
    local run_all=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --phase)
                phase="$2"
                shift 2
                ;;
            --job)
                job="$2"
                shift 2
                ;;
            --all)
                run_all=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--phase N] [--job JOB_NAME] [--all]"
                echo ""
                echo "Options:"
                echo "  --phase N       Run specific phase (1, 2, or 3)"
                echo "  --job JOB_NAME  Run specific job (bnn, xgboost, hierarchical, ensemble)"
                echo "  --all           Run all phases sequentially"
                echo "  --help          Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 --phase 1"
                echo "  $0 --job bnn"
                echo "  $0 --all"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Execute based on arguments
    if [[ "$run_all" == true ]]; then
        run_all_phases
    elif [[ -n "$phase" ]]; then
        case $phase in
            1)
                run_phase_1
                ;;
            2)
                run_phase_2
                ;;
            3)
                run_phase_3
                ;;
            *)
                log_error "Invalid phase: $phase (must be 1, 2, or 3)"
                exit 1
                ;;
        esac
    elif [[ -n "$job" ]]; then
        case $job in
            bnn)
                run_bnn_multioutput
                ;;
            xgboost)
                run_xgboost_gpu
                ;;
            hierarchical)
                run_hierarchical_bayesian
                ;;
            ensemble)
                run_ensemble_creation
                ;;
            *)
                log_error "Invalid job: $job (must be bnn, xgboost, hierarchical, or ensemble)"
                exit 1
                ;;
        esac
    else
        log_warning "No phase or job specified, running Phase 1 by default"
        run_phase_1
    fi

    local exit_code=$?

    print_header "TRAINING COMPLETE"
    log_info "Exit code: $exit_code"
    log_info "Report saved: $REPORT_FILE"

    exit $exit_code
}

# Run main function
main "$@"
