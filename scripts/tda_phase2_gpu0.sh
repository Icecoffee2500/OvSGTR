#!/bin/bash
# =============================================================================
# TDA Phase 2 Experiments - GPU 0
# Phase 2A: Object TDA - Positive cache sweep (7) + Negative cache sweep (4)
#
# Alpha ranges calibrated from SwinB experiments:
#   alpha=1.0, beta=5 → R@50=0.03 (catastrophic)
#   alpha=0.01, beta=5 → R@50=37.75 (slight drop)
#   alpha=0.001, beta=5 → R@50=38.02 (near baseline 38.06)
#
# Usage:
#   nohup bash scripts/tda_phase2_gpu0.sh > logs/tda_exp/gpu0_main.log 2>&1 &
# =============================================================================

GPU=0
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
# Phase 2A: Object TDA - Positive Cache Only
# neg_alpha=0, score_threshold=0.3, entropy=0.5-0.9
#
# Max logit addition = alpha * (shot_cap * exp(-(beta*(1-aff))))
# With shot_cap=5, affinity_p50=0.6:
#   beta=1: max_add = alpha * 3.35
#   beta=3: max_add = alpha * 1.50
#   beta=5: max_add = alpha * 0.68
# =============================================================================

# alpha=0.001, beta=1.0 → max_add≈0.003 (minimal effect)
run_exp "p2a_obj_pos_a001b1" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.001 tda_obj_pos_beta=1.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3

# alpha=0.005, beta=1.0 → max_add≈0.017 (mild broad)
run_exp "p2a_obj_pos_a005b1" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=1.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3

# alpha=0.01, beta=1.0 → max_add≈0.034 (moderate broad)
run_exp "p2a_obj_pos_a01b1" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.01 tda_obj_pos_beta=1.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3

# alpha=0.005, beta=3.0 → max_add≈0.008 (mild selective)
run_exp "p2a_obj_pos_a005b3" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=3.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3

# alpha=0.01, beta=3.0 → max_add≈0.015 (moderate selective)
run_exp "p2a_obj_pos_a01b3" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.01 tda_obj_pos_beta=3.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3

# alpha=0.01, beta=5.0 → max_add≈0.007 (moderate sharp, similar to SwinB 37.75)
run_exp "p2a_obj_pos_a01b5" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.01 tda_obj_pos_beta=5.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3

# alpha=0.05, beta=5.0 → max_add≈0.034 (strong sharp)
run_exp "p2a_obj_pos_a05b5" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.05 tda_obj_pos_beta=5.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3

# =============================================================================
# Phase 2A: Object TDA - Negative Cache Only (1/2)
# pos_alpha=0, entropy=0.5-0.9, score_threshold=0.3
# =============================================================================

# na=0.0005, nb=1.0 (very gentle)
run_exp "p2a_obj_neg_a0005b1" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.0005 tda_obj_neg_beta=1.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# na=0.001, nb=1.0 (mild broad)
run_exp "p2a_obj_neg_a001b1" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=1.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# na=0.005, nb=1.0 (moderate broad)
run_exp "p2a_obj_neg_a005b1" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=1.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# na=0.001, nb=3.0 (mild selective)
run_exp "p2a_obj_neg_a001b3" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=3.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# =============================================================================
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU} ALL DONE"
echo "  Completed: ${COMPLETED} / $((COMPLETED + FAILED))"
echo "  Failed:    ${FAILED}"
echo "=========================================="
