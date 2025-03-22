# Novel Channel Testing Framework

This directory contains a framework for testing machine learning models on multi-channel imaging data with novel channel configurations.

## Overview

The novel channel testing framework allows you to:

1. Train models on a subset of available datasets (e.g., Allen and HPA)
2. Test on a different dataset with novel channels (e.g., CP)
3. Evaluate different initialization strategies for novel channels

## Available Datasets and Channels

1. **Allen Dataset** (3 channels):
   - nucleus (index 5)
   - membrane (index 2)
   - protein (index 6)

2. **HPA Dataset** (4 channels):
   - microtubules (index 3)
   - protein (index 6)
   - nucleus (index 5)
   - er (index 0)

3. **CP Dataset** (5 channels):
   - nucleus (index 5)
   - er (index 0)
   - rna (index 7) - **Novel channel**
   - golgi (index 1) - **Novel channel**
   - mito (index 4) - **Novel channel**

## Novel Channel Initialization Strategies

The framework supports multiple initialization strategies for novel channels:

- `avg_2`: Average of 2 existing channels
- `replicate`: Copy parameters from similar channels
- `zero`: Initialize with zeros
- And more...

## Running Experiments

To run an experiment:

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

Or submit as a job:

```bash
qsub -P cs598 run_experiment.sh
```

## Configuration

The experiment configuration is defined in `configs/novel_channel_test.yaml`. You can modify this file to:

- Change training datasets
- Adjust model parameters
- Try different initialization strategies
- Configure evaluation metrics

## Results

Results will be saved to the specified directories in the configuration file:

- Model checkpoints: `/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/checkpoints`
- Evaluation results: `/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/results`
- Feature outputs: `/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/features`

The results will also be logged to Weights & Biases if configured.
