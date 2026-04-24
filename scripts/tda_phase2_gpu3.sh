#!/bin/bash
# =============================================================================
# TDA Phase 2 Experiments - GPU 3
# Phase 2B: Relation neg(1) + both(2) + ent(2) + misc(2)
# Phase 2C: Combined Object+Relation (4)
#
# Usage:
#   nohup bash scripts/tda_phase2_gpu3.sh > logs/tda_exp/gpu3_main.log 2>&1 &
# =============================================================================

GPU=3
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
# Phase 2B: Relation TDA - Negative Cache Only (last)
# =============================================================================

# na=0.005, nb=3.0 (moderate sharp)
run_exp "p2b_rln_neg_a005b3" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=3.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Phase 2B: Relation TDA - Both Caches
# =============================================================================

# balanced mid: pa=0.005 pb=1, na=0.001 nb=1
run_exp "p2b_rln_both_mid" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# aggressive: pa=0.01 pb=3, na=0.005 nb=3
run_exp "p2b_rln_both_strong" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=3.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=3.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Phase 2B: Relation TDA - Entropy Threshold Variation
# (both caches, pa=0.005 pb=1, na=0.001 nb=1)
# =============================================================================

# tighter low range
run_exp "p2b_rln_ent_015_030" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.15 tda_rln_neg_entropy_upper=0.30 \
  tda_rln_score_threshold=0.1

# wider range
run_exp "p2b_rln_ent_025_040" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.25 tda_rln_neg_entropy_upper=0.40 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Phase 2B: Relation TDA - Score Threshold & Shot Capacity
# (both caches, pa=0.005 pb=1, na=0.001 nb=1, entropy=0.2-0.35)
# =============================================================================

# higher score threshold (stricter)
run_exp "p2b_rln_sthr03" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.3

# larger cache (pos=10, neg=5)
run_exp "p2b_rln_shot_lg" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_pos_shot_capacity=10 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Phase 2C: Combined Object + Relation TDA
# Object: entropy=0.5-0.9, score_threshold=0.3
# Relation: entropy=0.2-0.35, score_threshold=0.1
# =============================================================================

# balanced mid-mid
run_exp "p2c_comb_mid" \
  tda_obj_enabled=True tda_rln_enabled=True \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=3.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=3.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# aggressive strong-strong
run_exp "p2c_comb_strong" \
  tda_obj_enabled=True tda_rln_enabled=True \
  tda_obj_pos_alpha=0.01 tda_obj_pos_beta=5.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=3.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=3.0 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# positive-only for both domains
run_exp "p2c_comb_posonly" \
  tda_obj_enabled=True tda_rln_enabled=True \
  tda_obj_pos_alpha=0.005 tda_obj_pos_beta=3.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3 \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1

# conservative soft-soft
run_exp "p2c_comb_soft" \
  tda_obj_enabled=True tda_rln_enabled=True \
  tda_obj_pos_alpha=0.001 tda_obj_pos_beta=1.0 \
  tda_obj_neg_alpha=0.0005 tda_obj_neg_beta=1.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_pos_alpha=0.001 tda_rln_pos_beta=0.5 \
  tda_rln_neg_alpha=0.0005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU} ALL DONE"
echo "  Completed: ${COMPLETED} / $((COMPLETED + FAILED))"
echo "  Failed:    ${FAILED}"
echo "=========================================="
