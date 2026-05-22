# CIFAR-10 Spiking Neural Network (CSNN) Pipeline

This document outlines the architecture, training process, and hyperparameter configuration for the CIFAR-10 image classification model using `snnTorch`.

## Overview
The goal of this pipeline is to effectively classify the CIFAR-10 dataset (32x32 RGB images, 10 classes) using a Convolutional Spiking Neural Network (CSNN). Spiking Neural Networks mimic biological neurons by communicating via discrete spikes over time, which requires specialized architectures and training methodologies (like Backpropagation Through Time).

## 1. Model Architecture

The SNN model (`CIFAR10SpikingNet`) is designed as a deep VGG-style Convolutional Network. Key features include the use of Leaky Integrate-and-Fire (LIF) neurons and 2D Batch Normalization to stabilize membrane potentials and ensure consistent spiking activity across layers.

**Architecture Flow:**
1. **Block 1**: `Conv2d(3 -> 64)` -> `BatchNorm2d` -> `MaxPool2d(2)` -> `snn.Leaky`
2. **Block 2**: `Conv2d(64 -> 128)` -> `BatchNorm2d` -> `MaxPool2d(2)` -> `snn.Leaky`
3. **Block 3**: `Conv2d(128 -> 256)` -> `BatchNorm2d` -> `MaxPool2d(2)` -> `snn.Leaky`
4. **FC Block 1**: `Flatten` -> `Linear(256 * 4 * 4 -> 1024)` -> `snn.Leaky`
5. **FC Block 2**: `Linear(1024 -> 10)` -> `snn.Leaky(output=True)`

### LIF Neuron Settings
- **Surrogate Gradient**: `surrogate.fast_sigmoid()` is used to approximate the derivative of the step function (spike emission), enabling standard backpropagation.
- **Beta (Decay Rate)**: `0.9`

## 2. Dataset & Augmentation

The `cifar10_dataset.py` handles data loading and augmentation using `torchvision`. 

**Training Augmentations:**
- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `ToTensor()`
- `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`

**Test Transformations:**
- `ToTensor()`
- `Normalize`

## 3. Training Loop Configurations

Training is executed via `train_cifar10.py` and is heavily optimized for SNNs.

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_epochs` | 50 | Total passes over the dataset. |
| `num_steps` | 25 | The number of time steps (T) simulated in the SNN for each input. |
| `batch_size` | 128 | Number of samples per batch. |
| `learning_rate` | 1e-3 | Initial learning rate. |

### Optimizer & Loss
- **Optimizer**: `Adam`
- **Scheduler**: `CosineAnnealingLR` (Decays the learning rate following a cosine curve over `num_epochs`).
- **Loss Function**: `SF.ce_rate_loss()` (Cross Entropy Rate Loss). This computes the cross entropy loss based on the rate of spikes over time emitted by the final layer.

## 4. How to Run

1. **Activate Virtual Environment**
   ```bash
   source .venv/bin/activate
   ```
2. **Start Training**
   ```bash
   python train/train_cifar10.py
   ```

The script automatically detects the optimal hardware accelerator (CUDA, MPS for Mac, or CPU). It prints the training and test accuracy per epoch, logs the learning rate, and saves the best model weights to `model_weights/cifar10/best_cifar10_snn.pth`.
