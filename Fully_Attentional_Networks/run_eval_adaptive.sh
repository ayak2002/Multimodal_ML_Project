#!/bin/bash
export PYTHONPATH=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy:$PYTHONPATH

# Define a timestamp for folder names
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Set Weights & Biases directory to our multimodal_tests directory
export WANDB_DIR=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/wandb
export WANDB_API_KEY=76ba312836aa6fce37a4623d64c9519f2528078d

#export WANDB_MODE=disabled
mkdir -p $WANDB_DIR

# Use your wandb account with the correct entity
export WANDB_PROJECT=dichavit_test_avantika
export WANDB_ENTITY=avanc-boston-university

python diverse_channel_vit/main.py -m \
  --config-dir /projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/configs \
  --config-name chammi_cfg \
  model=dichavit_adaptive \
  dataset=adaptive_morphem70k_v2_12channels \
  hardware=dp \
  ++train.num_epochs=0 \
  scheduler=none \
  ++train.resume_train=True \
  ++train.resume_model=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/checkpoints/morphem70k/2025-Mar-27-14-31-39--seed2025/model_last.pt \
  ++eval.skip_feature_extraction=True \
  ++eval.eval_subset_channels=True \
  ++eval.feature_dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/features/adaptive_channel_strategy_2025-04-03_00-52-21/features \
  ++eval.dest_dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/results/adaptive_channel_strategy_2025-04-03_00-52-21/results \
  ++model.new_channel_inits=[zero] \
  ++logging.wandb.run_name=C-dichavit \
   ++logging.wandb.project=re_eval_attnlogging \
  ++logging.wandb.entity=avanc-boston-university