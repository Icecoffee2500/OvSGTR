#!/bin/bash
# =============================================================================
# TDA Phase 2 Experiments - GPU 1
# Phase 2A: Object neg (3) + both (2) + entropy (2) + misc (2) + Rln pos (2)
#
# Usage:
#   nohup bash scripts/tda_phase2_gpu1.sh > logs/tda_exp/gpu1_main.log 2>&1 &
# =============================================================================

GPU=1
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
# Phase 2A: Object TDA - Negative Cache Only (2/2)
# =============================================================================

# na=0.005, nb=3.0 (moderate selective)
run_exp "p2a_obj_neg_a005b3" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=3.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# na=0.001, nb=5.0 (mild sharp)
run_exp "p2a_obj_neg_a001b5" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# na=0.005, nb=5.0 (moderate sharp)
run_exp "p2a_obj_neg_a005b5" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# =============================================================================
# Phase 2A: Object TDA - Both Caches
# =============================================================================

# balanced mid: pa=0.005 pb=3, na=0.001 nb=3
run_exp "p2a_obj_both_mid" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=3.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=3.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# aggressive: pa=0.01 pb=5, na=0.005 nb=5
run_exp "p2a_obj_both_strong" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.01 tda_obj_pos_beta=5.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# =============================================================================
# Phase 2A: Object TDA - Entropy Threshold Variation
# (both caches, pa=0.005 pb=3, na=0.001 nb=3)
# =============================================================================

# wider entropy (captures more neg samples)
run_exp "p2a_obj_ent_04_08" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=3.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=3.0 \
  tda_obj_neg_entropy_lower=0.4 tda_obj_neg_entropy_upper=0.8 \
  tda_obj_score_threshold=0.3

# tighter high-entropy only
run_exp "p2a_obj_ent_06_095" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=3.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=3.0 \
  tda_obj_neg_entropy_lower=0.6 tda_obj_neg_entropy_upper=0.95 \
  tda_obj_score_threshold=0.3

# =============================================================================
# Phase 2A: Object TDA - Score Threshold & Shot Capacity
# (both caches, pa=0.005 pb=3, na=0.001 nb=3, entropy=0.5-0.9)
# =============================================================================

# lower score threshold
run_exp "p2a_obj_sthr01" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=3.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=3.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.1

# larger cache (pos=10, neg=5)
run_exp "p2a_obj_shot_lg" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=3.0 \
  tda_obj_pos_shot_capacity=10 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=3.0 \
  tda_obj_neg_shot_capacity=5 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# =============================================================================
# Phase 2B: Relation TDA - Positive Cache Only (start)
# score_threshold=0.1, relation affinity p50~0.3
#   beta=0.5: max_add = alpha * 3.50
#   beta=1.0: max_add = alpha * 2.50
#   beta=3.0: max_add = alpha * 0.60
# =============================================================================

# pa=0.001, pb=0.5 → max_add≈0.004 (minimal)
run_exp "p2b_rln_pos_a001b05" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.001 tda_rln_pos_beta=0.5 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1

# pa=0.005, pb=0.5 → max_add≈0.018 (mild broad)
run_exp "p2b_rln_pos_a005b05" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=0.5 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1

# =============================================================================
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU} ALL DONE"
echo "  Completed: ${COMPLETED} / $((COMPLETED + FAILED))"
echo "  Failed:    ${FAILED}"
echo "=========================================="
