#!/bin/bash
# =============================================================================
# TDA Phase 3 Experiments - GPU 2
# Relation: neg shot sweep (4) + both shot sweep (5) + mask sweep start (2)
#
# Usage:
#   nohup bash scripts/tda_phase3_gpu2.sh > logs/tda_exp/gpu2_p3.log 2>&1 &
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
# Relation Neg Shot Sweep (P2 best: na=0.005, nb=0.5, R@50=35.101)
# Default neg_shot=3 was tested in P2
# =============================================================================

run_exp "p3_rln_negA_s2" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_shot_capacity=2 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

run_exp "p3_rln_negA_s5" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

run_exp "p3_rln_negA_s8" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_shot_capacity=8 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

run_exp "p3_rln_negA_s10" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_shot_capacity=10 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Relation Both_Mid Shot Sweep
# P2 shot_lg (pos=10,neg=5) was R@50=35.098 (#2)
# P2 both_mid (pos=5,neg=3) was R@50=35.055
# pa=0.005, pb=1, na=0.001, nb=1, ent=0.2-0.35
# =============================================================================

# default shots for comparison with shot_lg
run_exp "p3_rln_bmid_p5n3" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_pos_shot_capacity=5 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_shot_capacity=3 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# larger pos, keep neg
run_exp "p3_rln_bmid_p15n5" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_pos_shot_capacity=15 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# both large
run_exp "p3_rln_bmid_p10n10" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.005 tda_rln_pos_beta=1.0 \
  tda_rln_pos_shot_capacity=10 \
  tda_rln_neg_alpha=0.001 tda_rln_neg_beta=1.0 \
  tda_rln_neg_shot_capacity=10 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Relation Both_Strong Shot Sweep
# P2 both_strong R@50=35.074 (#3)
# pa=0.01, pb=3, na=0.005, nb=3, ent=0.2-0.35
# =============================================================================

run_exp "p3_rln_bstr_p10n5" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=3.0 \
  tda_rln_pos_shot_capacity=10 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=3.0 \
  tda_rln_neg_shot_capacity=5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

run_exp "p3_rln_bstr_p15n8" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.01 tda_rln_pos_beta=3.0 \
  tda_rln_pos_shot_capacity=15 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=3.0 \
  tda_rln_neg_shot_capacity=8 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_score_threshold=0.1

# =============================================================================
# Relation Mask Sweep (start) - Config A neg (na=0.005, nb=0.5)
# Default: mask_lower=0.03, mask_upper=1.0
# =============================================================================

run_exp "p3_rln_mask_l001" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_neg_mask_lower=0.01 tda_rln_neg_mask_upper=1.0 \
  tda_rln_score_threshold=0.1

run_exp "p3_rln_mask_l005" \
  tda_obj_enabled=False tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.005 tda_rln_neg_beta=0.5 \
  tda_rln_neg_entropy_lower=0.2 tda_rln_neg_entropy_upper=0.35 \
  tda_rln_neg_mask_lower=0.05 tda_rln_neg_mask_upper=1.0 \
  tda_rln_score_threshold=0.1

# =============================================================================
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU} ALL DONE"
echo "  Completed: ${COMPLETED} / $((COMPLETED + FAILED))"
echo "  Failed:    ${FAILED}"
echo "=========================================="
