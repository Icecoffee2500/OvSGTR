#!/bin/bash
# =============================================================================
# TDA Phase 2 Experiments - GPU 2
# Phase 2B: Relation TDA - Positive (5) + Negative (6)
#
# Usage:
#   nohup bash scripts/tda_phase2_gpu2.sh > logs/tda_exp/gpu2_main.log 2>&1 &
# =============================================================================

GPU=2
CONFIG="./config/GroundingDINO_SwinT_OGC_full.py"
CHECKPOINT="./logs/ovsgtr_vg_swint_full/checkpoint_best_regular.pth"
LOG_DIR="./logs/tda_exp"
WANDB_PROJECT="tda-tuning"

mkdir -p "${LOG_DIR}"

COMPLETED=0
FAILED=0

run_exp() {
  local name="$1"
  shift

  echo "=========================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: ${name} (GPU ${GPU})"
  echo "=========================================="

  mkdir -p "${LOG_DIR}/${name}"

  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    --output_dir "${LOG_DIR}/${name}" \
    -c "${CONFIG}" \
    --data_path ./data \
    --eval \
    --resume "${CHECKPOINT}" \
    --dataset_file vg \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${name}" \
    --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False use_test_set=True \
    "$@" \
    > "${LOG_DIR}/${name}.log" 2>&1

  local ec=$?
  if [ $ec -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: ${name}"
    COMPLETED=$((COMPLETED + 1))
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${name} (exit code: ${ec})"
    FAILED=$((FAILED + 1))
  fi
  echo ""
}

# =============================================================================
# Phase 2B: Relation TDA - Positive Cache Only (continued)
# relation affinity p50~0.3
# =============================================================================

# pa=0.01, pb=0.5 → max_add≈0.035 (moderate broad)
run_exp "p2b_rln_pos_a01b05" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=0.5 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1

# pa=0.005, pb=1.0 → max_add≈0.013 (mild selective)
run_exp "p2b_rln_pos_a005b1" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1

# pa=0.01, pb=1.0 → max_add≈0.025 (moderate selective)
run_exp "p2b_rln_pos_a01b1" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1

# pa=0.01, pb=3.0 → max_add≈0.006 (moderate sharp)
run_exp "p2b_rln_pos_a01b3" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=3.0 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1

# pa=0.05, pb=3.0 → max_add≈0.030 (strong sharp)
run_exp "p2b_rln_pos_a05b3" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.05 tda_rln_pos_beta=3.0 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Phase 2B: Relation TDA - Negative Cache Only
# pos_alpha=0, entropy=0.2-0.35, score_threshold=0.1
# =============================================================================

# na=0.0005, nb=0.5 (very gentle)
run_exp "p2b_rln_neg_a0005b05" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.0005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# na=0.001, nb=0.5 (mild broad)
run_exp "p2b_rln_neg_a001b05" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# na=0.005, nb=0.5 (moderate broad)
run_exp "p2b_rln_neg_a005b05" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# na=0.001, nb=1.0 (mild selective)
run_exp "p2b_rln_neg_a001b1" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# na=0.005, nb=1.0 (moderate selective)
run_exp "p2b_rln_neg_a005b1" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# na=0.001, nb=3.0 (mild sharp)
run_exp "p2b_rln_neg_a001b3" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=3.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU} ALL DONE"
echo "  Completed: ${COMPLETED} / $((COMPLETED + FAILED))"
echo "  Failed:    ${FAILED}"
echo "=========================================="
