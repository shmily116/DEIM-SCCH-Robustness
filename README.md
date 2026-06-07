# DEIM-SCCH-Robustness
# DEIM-SCCH Supplementary Robustness Evaluation

This repository provides supplementary robustness evaluation code and results for the paper:

**DEIM-SCCH: An Efficient Lightweight Detector for Real-Time Small-Object Perception**

Due to space limitations, the detailed corruption-specific robustness evaluation is not included in the main manuscript. This repository provides the corruption generation code, evaluation scripts, and supplementary robustness results for reproducibility.

## Contents

- `corruption_tools/`: scripts for generating corrupted test images
- `evaluation/`: scripts for evaluating models under corrupted inputs
- `results/`: supplementary robustness results
- `configs/`: model and evaluation configuration files
- `docs/`: detailed evaluation protocol and table reproduction instructions

## Corruption Types

The robustness evaluation includes the following corruption types:

- Brightness shift
- Gaussian noise
- Contrast shift
- Defocus blur
- Motion blur
- Rain and snow
- Lens droplets
- Wave reflection glare
- Fog

Each corruption is evaluated under five severity levels.

## Results

The supplementary results include:

1. Overall clean/corrupted comparison
2. Corruption-specific results
3. Severity-level results
4. Module-wise robustness ablation

The result files are available in the `results/` directory.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
