# Unsupervised Damage Taxonomy Construction for Composite Materials Using Acoustic Emission Analysis

## Overview

This repository contains the implementation accompanying our research on automated damage taxonomy construction for composite materials using Acoustic Emission (AE) signals.

The proposed workflow applies unsupervised machine learning to identify distinct damage states from AE feature data and models damage evolution during fatigue progression.

---

## Features

- AE data preprocessing
- Feature extraction and normalization
- PCA visualization
- K-Means clustering
- Damage taxonomy construction
- Hierarchical damage-state mapping
- Damage evolution analysis
- Cluster transition matrix
- Bootstrap stability analysis
- Silhouette analysis
- Model validation

---

## Dataset

The experiments use the publicly available Acoustic Emission dataset from Mendeley Data.

DOI:

https://doi.org/10.17632/svz439j8hb.1

---

## Repository Structure

```
01_data_loading.py
```

Loads the AE dataset.

```
ae02_preprocessing.py
```

Data cleaning and normalization.

```
03_feature_physics.py
```

Physical interpretation of AE descriptors.

```
ae04_dimensionality_reduction.py
```

PCA visualization.

```
ae05_clustering.py
```

K-Means clustering.

```
06_taxonomy_analysis.py
```

Damage taxonomy construction.

```
ae07_robustness_analysis.py
```

Cluster robustness evaluation.

```
ae08_hierarchy_construction.py
```

Hierarchical damage-state mapping.

```
ae09_damage_trajectories.py
```

Damage evolution modelling.

```
ae10_visualization.py
```

Generation of publication figures.

```
ae11_stability_figure.py
```

Bootstrap stability analysis.

---

## Validation

The proposed taxonomy is validated using

- Silhouette Score
- Bootstrap Jaccard Stability
- Cluster Transition Matrix
- Hierarchical Damage Mapping

---

## Requirements

Python 3.11+

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

Run the complete workflow

```bash
python main.py
```

or execute the individual modules independently.

---

## Outputs

Running the pipeline generates

- PCA embedding
- Cluster feature profiles
- Damage evolution plots
- Transition matrices
- Validation figures
- Cluster statistics

All figures are saved inside the `Figures/` directory.

---

## Citation

If you use this repository, please cite our accompanying paper.

---

## Author

Mythili Paturi

Robotics and Artificial Intelligence

Amrita Vishwa Vidyapeetham

Bengaluru, India
