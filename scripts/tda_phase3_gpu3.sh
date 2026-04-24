#!/bin/bash
# =============================================================================
# TDA Phase 3 Experiments - GPU 3
# Relation mask sweep (3) + Combined with shot/mask variations (8)
#
# Usage:
#   nohup bash scripts/tda_phase3_gpu3.sh > logs/tda_exp/gpu3_p3.log 2>&1 &
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
# Relation Mask Sweep (continued) - Config A neg (na=0.005, nb=0.5)
# =============================================================================

run_exp "p3_rln_mask_l01" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_neg_mask_lower=0.1 tda_rln_neg_mask_upper=1.0 \
  tda_rln_score_threshold=0.1

# vary mask_upper (keep lower=0.03)
run_exp "p3_rln_mask_u05" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_neg_mask_lower=0.03 tda_rln_neg_mask_upper=0.5 \
  tda_rln_score_threshold=0.1

run_exp "p3_rln_mask_u03" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_neg_mask_lower=0.03 tda_rln_neg_mask_upper=0.3 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Combined Variations: obj_neg(A) + rln_neg(A) with mask changes
# Base: obj(na=0.001,nb=5,ent=0.5-0.9) + rln(na=0.005,nb=0.5,ent=0.2-0.35)
# =============================================================================

# obj mask_lower=0.05
run_exp "p3_cv_nn_oml005" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_neg_mask_lower=0.05 tda_obj_neg_mask_upper=1.0 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# obj mask_upper=0.5 (don't suppress high-confidence preds)
run_exp "p3_cv_nn_omu05" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_neg_mask_lower=0.03 tda_obj_neg_mask_upper=0.5 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# rln mask_lower=0.05
run_exp "p3_cv_nn_rml005" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_neg_mask_lower=0.05 tda_rln_neg_mask_upper=1.0 \
  tda_rln_score_threshold=0.1

# rln mask_upper=0.5
run_exp "p3_cv_nn_rmu05" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_neg_mask_lower=0.03 tda_rln_neg_mask_upper=0.5 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Combined Variations: obj_neg(A) + rln_both_mid with shots
# rln: pa=0.005, pb=1, na=0.001, nb=1
# =============================================================================

# rln_both_mid(shot=10/5) + obj shot=5
run_exp "p3_cv_rbm_os5" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=5 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_pos_shot_capacity=10 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# rln_both_mid(shot=10/5) + obj shot=10
run_exp "p3_cv_rbm_os10" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=10 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_pos_shot_capacity=10 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# combined neg+neg with obj shot=5, rln shot=10
run_exp "p3_cv_nn_os5rs10" \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=5 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_shot_capacity=10 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU} ALL DONE"
echo "  Completed: ${COMPLETED} / $((COMPLETED + FAILED))"
echo "  Failed:    ${FAILED}"
echo "=========================================="
