# ProtoPNet for Medical Image Classification

This repository contains the implementation of ProtoPNet (Prototype Network) adapted for medical image classification tasks. The model learns to classify images by comparing them to learned prototypes that are visualizable and interpretable. The approach makes model decisions transparent, which is particularly valuable in medical imaging applications.

## Features

- **Prototype-based Interpretability:** The model learns a set of prototypes, each representing different parts of the training images. Classification is based on the similarity between image patches and these prototypes.
- **Medical Image Classification:** This adaptation is focused on classifying medical images, providing both accuracy and interpretability in the results.
- **Support for Push Operations:** The prototypes are refined using a push phase to ensure that each prototype represents meaningful and distinguishable parts of the input data.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- PIL
- matplotlib
- OpenCV
- Wandb (for logging)
- ProtoPNet library (included)

You can install the required dependencies using:

```bash
git clone https://github.com/cfchen-duke/ProtoPNet
pip install -r requirements.txt
```