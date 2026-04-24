#!/bin/bash
# =============================================================================
# TDA Phase 3 Experiments - GPU 0
# Combined Object_neg + Relation (HIGHEST PRIORITY)
#
# Phase 2 key findings:
#   Object: neg-only works (best: na=0.001, nb=5, R@50=35.066), pos hurts
#   Relation: neg best (na=0.005, nb=0.5, R@50=35.101), both also works
#   Phase 2C combined: all hurt because obj_pos was included!
#   → obj_neg_only + rln_neg/both was NEVER TESTED → biggest potential gain
#
# Usage:
#   nohup bash scripts/tda_phase3_gpu0.sh > logs/tda_exp/gpu0_p3.log 2>&1 &
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
# Combined: obj_neg(A) + rln_neg(A) — NEVER TESTED IN PHASE 2
# obj: pos_alpha=0, neg_alpha=0.001, neg_beta=5, ent=0.5-0.9, sthr=0.3
# rln: pos_alpha=0, neg_alpha=0.005, neg_beta=0.5, ent=0.2-0.35, sthr=0.1
# =============================================================================

# baseline combined: obj_neg best + rln_neg best, default shots
run_exp "p3_comb_nn_base" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# obj_neg best + rln_neg best, rln larger shot
run_exp "p3_comb_nn_rlns10" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_shot_capacity=10 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Combined: obj_neg(A) + rln_both_mid
# rln: pa=0.005, pb=1.0, na=0.001, nb=1.0
# =============================================================================

# default shots
run_exp "p3_comb_n_rbmid" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# rln larger shots (shot_lg was P2 #2)
run_exp "p3_comb_n_rbmid_s" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_pos_shot_capacity=10 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Combined: obj_neg(A) + rln_both_strong
# rln: pa=0.01, pb=3.0, na=0.005, nb=3.0
# =============================================================================

run_exp "p3_comb_n_rbstr" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=3.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=3.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

run_exp "p3_comb_n_rbstr_s" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=3.0 \
  tda_rln_pos_shot_capacity=10 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=3.0 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Combined: obj_neg(B/C) + rln variations
# Try P2 obj #2 and #3 with best rln configs
# =============================================================================

# obj_neg B (na=0.005, nb=5) + rln_neg best
run_exp "p3_comb_nB_rneg" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# obj_neg B + rln_both_mid
run_exp "p3_comb_nB_rbmid" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# obj_neg C (na=0.0005, nb=1) + rln_neg best
run_exp "p3_comb_nC_rneg" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.0005 tda_obj_neg_beta=1.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Combined: shot variations on best combined (neg+neg)
# =============================================================================

# both domains larger shots
run_exp "p3_comb_nn_os5rs5" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=5 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU} ALL DONE"
echo "  Completed: ${COMPLETED} / $((COMPLETED + FAILED))"
echo "  Failed:    ${FAILED}"
echo "=========================================="

# test