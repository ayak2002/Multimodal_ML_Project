#!/bin/bash

# Set the Python path to include the project directories
export PYTHONPATH=$PYTHONPATH:/projectnb/cs598/projects/Modalities_Robustness:/projectnb/cs598/projects/Modalities_Robustness/diverse_channel_vit

# Navigate to the diverse_channel_vit directory
cd /projectnb/cs598/projects/Modalities_Robustness/diverse_channel_vit

# Define a timestamp for folder names
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Set Weights & Biases directory to our multimodal_tests directory
export WANDB_DIR=/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/wandb
mkdir -p $WANDB_DIR

# Disable Weights & Biases for now
export WANDB_MODE=disabled

# Run the experiment with the novel channel test configuration
# Use the original config path but override specific values
python main.py -m --config-name=chammi_cfg \
  model=dichavit \
  dataset=morphem70k_v2_12channels \
  data_chunk=morphem70k \
  scheduler=cosine \
  optimizer=adamw \
  hardware=dp \
  eval=default \
  attn_pooling=none \
  logging=no \
  ++train.resume_train=False \
  ++train.training_chunks=Allen_HPA \
  ++train.num_epochs=60 \
  ++train.batch_size=64 \
  ++train.save_model=last \
  ++train.checkpoints=/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/checkpoints \
  ++model.new_channel_inits=[avg_2,replicate,zero] \
  ++model.in_channel_names=[er,golgi,membrane,microtubules,mito,nucleus,protein,rna] \
  ++eval.batch_size=64 \
  ++eval.dest_dir=/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/results/novel_channel_${TIMESTAMP}/results \
  ++eval.feature_dir=/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/features/novel_channel_${TIMESTAMP}/features \
  ++hydra.run.dir=/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/hydra_output/${TIMESTAMP} \
  ++hydra.sweep.dir=/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/hydra_output/multirun/${TIMESTAMP} \
  ++hardware.num_workers=4
