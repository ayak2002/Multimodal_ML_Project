#!/bin/bash

# Set the Python path to include the project directories
export PYTHONPATH=$PYTHONPATH:/projectnb/cs598/students/ayak:/projectnb/cs598/students/ayak/diverse_channel_vit

# Navigate to the diverse_channel_vit directory
cd /projectnb/cs598/students/ayak/diverse_channel_vit

# Define a timestamp for folder names
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Set Weights & Biases directory to our multimodal_tests directory
export WANDB_DIR=/projectnb/cs598/students/ayak/multimodal_tests/wandb
export WANDB_API_KEY=7a5ef7f1e907d577d3ccf8e33d46b70d5e1cc4f7

#export WANDB_MODE=disabled
mkdir -p $WANDB_DIR

# Use your wandb account with the correct entity
export WANDB_PROJECT=dichavit_test
export WANDB_ENTITY=ayak-boston-university

python main.py -m -cn chammi_cfg \
  model=dichavit \
  ++model.enable_sample=True \
  ++model.pretrained_model_name=small \
  tag=chammi_demo \
  dataset=morphem70k_v2_12channels \
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
  ++eval.skip_eval_first_epoch=True \
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
  ++train.training_chunks="Allen" \
  ++train.checkpoints=/projectnb/cs598/students/ayak/multimodal_tests/checkpoints/Allen \
  ++eval.dest_dir=/projectnb/cs598/students/ayak/multimodal_tests/results/Allen_tests_${TIMESTAMP}/results \
  ++eval.feature_dir=/projectnb/cs598/students/ayak/multimodal_tests/features/Allen_tests_${TIMESTAMP}/features \
  ++hydra.run.dir=/projectnb/cs598/students/ayak/multimodal_tests/hydra_output/${TIMESTAMP} \
  ++hydra.sweep.dir=/projectnb/cs598/students/ayak/multimodal_tests/hydra_output/multirun/${TIMESTAMP} \
  ++hardware.num_workers=4 \
  ++logging.wandb.project=dichavit_test \
  ++logging.wandb.entity=ayak-boston-university
