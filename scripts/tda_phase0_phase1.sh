#!/bin/bash
# =============================================================================
# TDA Hyperparameter Tuning: Phase 0 (Distribution Analysis) + Phase 1 (Baseline)
# 각 명령어를 별도 터미널에서 복사-붙여넣기로 실행하세요.
# =============================================================================

# ===== GPU 0: Phase 1 Baseline (TDA OFF) =====
CUDA_VISIBLE_DEVICES=0 python main.py \
  --output_dir ./logs/tda_exp/phase1_baseline \
  -c ./config/GroundingDINO_SwinT_OGC_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swint_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --wandb_project tda-tuning \
  --wandb_run_name phase1_baseline \
  --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False use_test_set=True


# ===== GPU 1: Phase 0 stats (score_threshold=0.1) =====
CUDA_VISIBLE_DEVICES=1 python main.py \
  --output_dir ./logs/tda_exp/phase0_stats_thr01 \
  -c ./config/GroundingDINO_SwinT_OGC_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swint_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --wandb_project tda-tuning \
  --wandb_run_name phase0_stats_thr0.1 \
  --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False use_test_set=True \
  tda_collect_stats=True \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.1 \
  tda_obj_neg_entropy_lower=0.0 \
  tda_obj_neg_entropy_upper=1.0 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.1 \
  tda_rln_neg_entropy_lower=0.0 \
  tda_rln_neg_entropy_upper=1.0


# ===== GPU 2: Phase 0 stats (score_threshold=0.3) =====
CUDA_VISIBLE_DEVICES=2 python main.py \
  --output_dir ./logs/tda_exp/phase0_stats_thr03 \
  -c ./config/GroundingDINO_SwinT_OGC_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swint_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --wandb_project tda-tuning \
  --wandb_run_name phase0_stats_thr0.3 \
  --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False use_test_set=True \
  tda_collect_stats=True \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.3 \
  tda_obj_neg_entropy_lower=0.0 \
  tda_obj_neg_entropy_upper=1.0 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.3 \
  tda_rln_neg_entropy_lower=0.0 \
  tda_rln_neg_entropy_upper=1.0


# ===== GPU 3: Phase 0 stats (score_threshold=0.5) =====
CUDA_VISIBLE_DEVICES=3 python main.py \
  --output_dir ./logs/tda_exp/phase0_stats_thr05 \
  -c ./config/GroundingDINO_SwinT_OGC_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swint_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --wandb_project tda-tuning \
  --wandb_run_name phase0_stats_thr0.5 \
  --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False use_test_set=True \
  tda_collect_stats=True \
  tda_obj_enabled=True \
  tda_obj_pos_alpha=0.0 \
  tda_obj_neg_alpha=0.0 \
  tda_obj_score_threshold=0.5 \
  tda_obj_neg_entropy_lower=0.0 \
  tda_obj_neg_entropy_upper=1.0 \
  tda_rln_enabled=True \
  tda_rln_pos_alpha=0.0 \
  tda_rln_neg_alpha=0.0 \
  tda_rln_score_threshold=0.5 \
  tda_rln_neg_entropy_lower=0.0 \
  tda_rln_neg_entropy_upper=1.0
