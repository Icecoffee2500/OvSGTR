#!/bin/bash
# =============================================================================
# TDA Phase 3 Experiments - GPU 1
# Object neg cache: shot_capacity sweep (6) + mask_threshold sweep (5)
#
# Usage:
#   nohup bash scripts/tda_phase3_gpu1.sh > logs/tda_exp/gpu1_p3.log 2>&1 &
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
# Object Neg Shot Sweep - Config A (P2 best: na=0.001, nb=5)
# Default shot=3 was tested in P2 (R@50=35.066)
# =============================================================================

run_exp "p3_obj_negA_s2" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=2 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

run_exp "p3_obj_negA_s5" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=5 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

run_exp "p3_obj_negA_s8" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=8 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

run_exp "p3_obj_negA_s10" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=10 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# =============================================================================
# Object Neg Shot Sweep - Config B (P2 #2: na=0.005, nb=5)
# =============================================================================

run_exp "p3_obj_negB_s5" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=5 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

run_exp "p3_obj_negB_s10" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.005 tda_obj_neg_beta=5.0 \
  tda_obj_neg_shot_capacity=10 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_score_threshold=0.3

# =============================================================================
# Object Neg Mask Sweep - Config A (na=0.001, nb=5, shot=3)
# Default: mask_lower=0.03, mask_upper=1.0
# =============================================================================

# vary mask_lower (keep upper=1.0)
run_exp "p3_obj_mask_l001" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_neg_mask_lower=0.01 tda_obj_neg_mask_upper=1.0 \
  tda_obj_score_threshold=0.3

run_exp "p3_obj_mask_l005" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_neg_mask_lower=0.05 tda_obj_neg_mask_upper=1.0 \
  tda_obj_score_threshold=0.3

run_exp "p3_obj_mask_l01" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_neg_mask_lower=0.1 tda_obj_neg_mask_upper=1.0 \
  tda_obj_score_threshold=0.3

# vary mask_upper (keep lower=0.03)
run_exp "p3_obj_mask_u05" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_neg_mask_lower=0.03 tda_obj_neg_mask_upper=0.5 \
  tda_obj_score_threshold=0.3

run_exp "p3_obj_mask_u03" \
  tda_obj_enabled=True tda_rln_enabled=False \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.001 tda_obj_neg_beta=5.0 \
  tda_obj_neg_entropy_lower=0.5 tda_obj_neg_entropy_upper=0.9 \
  tda_obj_neg_mask_lower=0.03 tda_obj_neg_mask_upper=0.3 \
  tda_obj_score_threshold=0.3

# =============================================================================
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU} ALL DONE"
echo "  Completed: ${COMPLETED} / $((COMPLETED + FAILED))"
echo "  Failed:    ${FAILED}"
echo "=========================================="
