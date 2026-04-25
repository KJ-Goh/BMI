# BMI Coursework: Monkey Decoders

This project implements a causal brain-machine interface (BMI) decoder for predicting two-dimensional hand trajectories from motor cortical spike trains.

## Project Overview

The goal of this project is to decode hand position from neural spike activity recorded during an eight-direction reaching task. The proposed decoder uses a task-structured routing strategy: movement direction is first estimated using a PCA-LDA classifier, and the predicted direction is then used to select a direction-specific principal component regression (PCR) model for trajectory decoding.

The final decoder performs causal position estimation every 20 ms from 320 ms onward. In the main benchmark, performance is evaluated over the fixed 320–560 ms window using RMSE.

## Method Summary

The decoding pipeline consists of:

1. **Spike preprocessing**
   - Neural spike trains are converted into binned spike-count features.
   - 20 ms bins are used for trajectory regression.
   - 80 ms bins are used for direction classification.

2. **Direction classification**
   - PCA-LDA classifiers estimate the intended reach direction at predefined checkpoints.
   - The most recent direction prediction is used to route the decoder.

3. **Trajectory decoding**
   - A separate PCR model is trained for each movement direction.
   - The selected direction-specific PCR expert predicts 2D hand position.

4. **Evaluation**
   - The main benchmark uses 50 Monte Carlo 80/20 train-test splits.
   - Performance is measured by RMSE over 320–560 ms.
   - Additional ablation studies examine temporal settings, robustness, and data efficiency.

## Main Result

The proposed PCA-LDA + direction-specific PCR decoder achieved strong performance compared with pooled baselines. In the main benchmark, it reached an RMSE of approximately **7.59 cm**, outperforming global PCR and remaining close to the oracle routed-PCR reference.

A separate coursework real-time test script was also used for qualitative continuous trajectory visualization from 320 ms to the end of each trial.

