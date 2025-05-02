#!/bin/bash

# Set the Python path to include the project directories
#export PYTHONPATH=$PYTHONPATH:/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy:/projectnb/cs598/projects/Modalities_Robustness/diverse_channel_vit
#export PYTHONPATH=$PYTHONPATH:/projectnb/cs598/projects/Modalities_Robustness:/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy:/projectnb/cs598/projects/Modalities_Robustness/diverse_channel_vit
export PYTHONPATH=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy:$PYTHONPATH
#export PYTHONPATH=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy:/projectnb/cs598/projects/Modalities_Robustness/diverse_channel_vit:$PYTHONPATH
# Navigate to the diverse_channel_vit directory
#cd /projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy

# Define a timestamp for folder names
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Set Weights & Biases directory to our multimodal_tests directory
export WANDB_DIR=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/wandb
export WANDB_API_KEY=76ba312836aa6fce37a4623d64c9519f2528078d

#export WANDB_MODE=disabled
mkdir -p $WANDB_DIR

# Use your wandb account with the correct entity
export WANDB_PROJECT=dichavit_test
export WANDB_ENTITY=avanc-boston-university
#++dataset.file_name=/projectnb/cs598/projects/Modalities_Robustness/diverse_channel_vit/metadata/morphem70k_v2.csv \
#python main.py -m --config-path /projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy --config-name chammi_cfg \
python diverse_channel_vit/main.py -m \
  --config-dir /projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/configs \
  --config-name chammi_cfg \
  model=dichavit_adaptive \
  ++model.enable_sample=True \
  ++model.pretrained_model_name=small \
  tag=chammi_demo \
  dataset=adaptive_morphem70k_v2_12channels \
  hardware=dp \
  ++optimizer.params.lr=0.00004 \
  ++model.temperature=0.07 \
  ++train.num_epochs=1 \
  scheduler=none \
  ++train.resume_train=True \
  ++train.resume_model=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/checkpoints/morphem70k/2025-Mar-27-14-31-39--seed2025/model_last.pt \
  ++train.save_model=last \
  ++model.new_channel_inits=[zero] \
  ++train.batch_size=64 \
  ++eval.batch_size=256 \
  ++train.debug=False \
  ++eval.every_n_epochs=6 \
  ++eval.skip_eval_first_epoch=True \
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
  ++logging.wandb.run_name=C-dichavit \
  ++model.hcs_sampling=lowest_cosine_prob \
  ++train.checkpoints=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/checkpoints \
  ++eval.dest_dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/results/adaptive_channel_strategy_${TIMESTAMP}/results \
  ++eval.feature_dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/features/adaptive_channel_strategy_${TIMESTAMP}/features \
  ++hydra.run.dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/hydra_output/${TIMESTAMP} \
  ++hydra.sweep.dir=/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/hydra_output/multirun/${TIMESTAMP} \
  ++hardware.num_workers=4 \
  ++logging.wandb.project=dichavit_test_avantika \
  ++logging.wandb.entity=avanc-boston-university