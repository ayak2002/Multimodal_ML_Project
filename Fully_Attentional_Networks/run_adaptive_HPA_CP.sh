#!/bin/bash

# Set the Python path to include the project directories
export PYTHONPATH=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy_aya:$PYTHONPATH

conda activate /projectnb/cs598/projects/Modalities_Robustness/dichavit

# Define a timestamp for folder names
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Set Weights & Biases directory
export WANDB_DIR=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy_aya/wandb
mkdir -p $WANDB_DIR

# Use your wandb account with the correct entity
export WANDB_PROJECT=adaptive_dichavit_test
export WANDB_ENTITY=ayak-boston-university



python /projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy_aya/diverse_channel_vit/main.py -m \
  --config-dir /projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/configs \
  --config-name chammi_cfg \
  model=dichavit_adaptive \
  ++model.enable_sample=True \
  ++model.pretrained_model_name=small \
  tag=adaptive_chammi_fan_demo \
  dataset=adaptive_morphem70k_v2_12channels \
  hardware=dp \
  ++optimizer.params.lr=0.00004 \
  ++model.temperature=0.07 \
  ++train.num_epochs=60 \
  ++train.save_model=last \
  ++model.new_channel_inits=[zero] \
  ++train.batch_size=64 \
  ++eval.batch_size=256 \
  ++train.debug=False \
  ++eval.every_n_epochs=6 \
  ++eval.skip_eval_first_epoch=False \
  ++eval.eval_subset_channels=True \
  ++eval.skip_feature_extraction=True \
  ++train.tps_prob=0.2 \
  ++model.orthogonal_channel_emb_init=True \
  ++train.extra_loss_lambda=1 \
  ++model.proxy_loss_lambda=0.1 \
  ++model.ortho_loss_v1_lambda=1 \
  ++model.gamma_s=0.5 \
  ++model.gamma_d=2 \
  ++model.reverse_pos_pairs=True \
  ++train.seed=2025 \
  ++logging.wandb.run_name=Adaptive-DiChaViT-FAN-HPA-CP \
  ++model.hcs_sampling=lowest_cosine_prob \
  ++model.fan_enabled=True \
  ++train.training_chunks="HPA_CP" \
  ++train.checkpoints=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy_aya/checkpoints/HPA_CP_adaptive \
  ++eval.dest_dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy_aya/results/HPA_CP_adaptive_${TIMESTAMP}/results \
  ++eval.feature_dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy_aya/features/HPA_CP_adaptive_${TIMESTAMP}/features \
  ++hydra.run.dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy_aya/hydra_output/${TIMESTAMP} \
  ++hydra.sweep.dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy_aya/hydra_output/multirun/${TIMESTAMP} \
  ++hardware.num_workers=4 \
  ++logging.wandb.project=adaptive_dichavit_FAN_test_HPA_CP \
  ++logging.wandb.entity=ayak-boston-university
