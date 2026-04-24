# 1. TDA 없이 기존 성능 확인 (baseline)
bash scripts/DINO_eval.sh vg ./config/GroundingDINO_SwinB_full.py ./data ./logs/ovsgtr_vg_swinb_full_eval_baseline ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth
bash scripts/DINO_eval.sh vg ./config/GroundingDINO_SwinT_OGC_full.py ./data ./logs/ovsgtr_vg_swint_full_eval_baseline ./logs/ovsgtr_vg_swint_full/checkpoint_best_regular.pth

bash scripts/DINO_eval.sh vg ./config/GroundingDINO_SwinB_full.py ./data ./logs/ovsgtr_vg_swinb_full_eval_baseline ./logs/ovsgtr_vg_swinb_full/checkpoint0005.pth

# 2. TDA 적용하여 테스트
CUDA_VISIBLE_DEVICES=1 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_full_eval_tda \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_obj_enabled=True

# 3. TDA 하이퍼파라미터 튜닝
CUDA_VISIBLE_DEVICES=0 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_full_eval_tda_v2 \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_obj_enabled=True \
  tda_pos_alpha=2.0 tda_pos_beta=3.0 tda_score_threshold=0.4

# ------------------------------------------------------------------------
  # alpha=0.001
  CUDA_VISIBLE_DEVICES=0 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_full_eval_tda_v2 \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_obj_enabled=True \
  tda_pos_alpha=0.0005 tda_neg_alpha=0.0

  CUDA_VISIBLE_DEVICES=1 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_full_eval_tda_v2 \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_obj_enabled=True \
  tda_pos_alpha=0.001 tda_neg_alpha=0.0

  CUDA_VISIBLE_DEVICES=2 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_full_eval_tda_v2 \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_obj_enabled=True \
  tda_pos_alpha=0.0 tda_neg_alpha=0.0001

  CUDA_VISIBLE_DEVICES=3 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_full_eval_tda_v2 \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_obj_enabled=True \
  tda_pos_alpha=0.005 tda_neg_alpha=0.0


  #  Relation TDA ----------------------
  CUDA_VISIBLE_DEVICES=0 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_eval_tda_rln \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_rln_enabled=True \
  tda_rln_pos_alpha=0.001 tda_rln_neg_alpha=0.0001 \
  tda_rln_score_threshold=0.5

  CUDA_VISIBLE_DEVICES=1 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_eval_tda_rln \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_rln_enabled=True \
  tda_rln_pos_alpha=0.001 tda_rln_neg_alpha=0.0001 \
  tda_rln_neg_mask_lower=0.1 tda_rln_score_threshold=0.5

  CUDA_VISIBLE_DEVICES=2 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_eval_tda_rln \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_rln_enabled=True \
  tda_rln_pos_alpha=0.001 tda_rln_neg_alpha=0.0001 \
  tda_rln_score_threshold=0.7

  CUDA_VISIBLE_DEVICES=3 python main.py \
  --output_dir ./logs/ovsgtr_vg_swinb_eval_tda_rln \
  -c ./config/GroundingDINO_SwinB_full.py \
  --data_path ./data \
  --eval \
  --resume ./logs/ovsgtr_vg_swinb_full/checkpoint_best_regular.pth \
  --dataset_file vg \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  use_test_set=True tda_rln_enabled=True \
  tda_rln_pos_alpha=0.001 tda_rln_neg_alpha=0.0001 \
  tda_rln_neg_mask_lower=0.1 tda_rln_score_threshold=0.7


  # TDA Phase 0 + Phase 1
  bash scripts/tda_phase0_phase1.sh