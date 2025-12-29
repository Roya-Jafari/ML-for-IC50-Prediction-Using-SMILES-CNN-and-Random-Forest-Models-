# ML-for-IC50-Prediction-Using-SMILES-(CNN-and-Random-Forest-Models)-
Machine learning models (Random Forest and Neural Networks) for predicting pIC50 from SMILES using RDKit descriptors.

# Predicting pIC50 from SMILES using Machine Learning

This project explores molecular activity prediction using machine learning
models trained on RDKit molecular descriptors.

## Overview
- Dataset: ~200k IC50 measurements from ChEMBL
- Task: Predict pIC50 values from SMILES
- Models:
  - Random Forest (scikit-learn)
  - Neural Network (PyTorch)

## Pipeline
1. Data cleaning and IC50 → pIC50 conversion
2. SMILES validation and RDKit molecule generation
3. Descriptor and fingerprint calculation
4. Model training and evaluation
5. Comparison of classical ML vs deep learning

## Key Takeaways
- Random Forest models generalize more robustly on noisy bioactivity data
- Neural networks achieve higher training performance but require
  careful regularization to avoid overfitting
- Large-scale cheminformatics datasets present numerical stability challenges

## Technologies
- Python
- RDKit
- scikit-learn
- PyTorch
- NumPy / pandas

## Disclaimer
This repository is intended as a research and learning project.
Raw ChEMBL data is not included.
