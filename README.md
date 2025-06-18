# Musical Key CNN

This repository provides a full pipeline for musical key classification based on convolutional neural networks, inspired by the paper [\[1\]](#literature). It contains scripts for preprocessing, training, evaluation, and prediction of key labels on new music tracks, using the GiantSteps and GiantSteps-MTG datasets and the [Camelot Wheel](https://mixedinkey.com/camelot-wheel/) for key mapping.

---

## Table of Contents

- [Description](#description)
- [Setup and Installation](#setup-and-installation)
- [Key Prediction for Your Own Songs](#key-prediction-for-your-own-songs)
- [Dataset Preparation](#dataset-preparation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Literature](#literature)

---

## Description

This repository implements a CNN model for musical key detection. It provides scripts to:
- Preprocess datasets (extract CQT spectrograms, prepare annotations, augment with pitch shifts)
- Train the model from scratch
- Evaluate model performance with [MIREX key evaluation metrics](https://www.music-ir.org/mirex/wiki/2025:Audio_Key_Detection)
- Predict keys for custom .mp3 files

---

## Setup and Installation

We recommend using a Python virtual environment for reproducibility and package management:

```sh
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

`pip install -r requirements.txt`

**Important**

For PyTorch and torchaudio, follow the instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
Choose the correct command for your CUDA/cuDNN or CPU environment.

## Key Prediction for Your Own Songs

You can analyze any individual .mp3 or a folder of .mp3 tracks using the provided model or your own trained model:

```sh
python predict_keys.py -f path/to/your_song.mp3
python predict_keys.py -f path/to/your/music_folder/
```

The script prints a summary table with:
- Filename
- Classification index (0-23)
- Index according to the Camelot Wheel (e.g., "8A" or "3B")
- The corresponding key

You can set the model checkpoint path with `-m path/to/your_model.pt` and the computation device with `--device cuda` or `--device cpu`.

## Dataset Preparation

For training and evaluation, you need the following datasets:

- [GiantSteps MTG Key Dataset](https://github.com/GiantSteps/giantsteps-mtg-key-dataset) (Training)
- [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) (Evaluation)

Directory structure:
Place or symlink the datasets under the `Dataset/` folder:

```sh
Dataset/
    giantsteps-key-dataset/
    giantsteps-mtg-key-dataset/
```

## Preprocessing

Before training or evaluation, preprocess the datasets to generate CQT spectrograms for all tracks and pitch-shifted variants.

`python preprocess_data.py`

All resulting .pkl spectrograms are stored in subfolders of `Dataset/`.

## Training

To train a new key classification model on the MTG dataset, run:

`python train.py`

You can modify hyperparameters or training parameters by editing `train.py`.

## Evaluation

To evaluate a trained model (e.g., calculate MIREX scores on GiantSteps):

`python eval.py`

The output includes overall accuracy and weighted [MIREX scores](https://www.music-ir.org/mirex/wiki/2025:Audio_Key_Detection).

The following table contains the percentage ratios and the weighted Mirex scores:

| Method | Weighted | Correct | Fifth | Relative | Parallel | Other |
| ------ | -------- | ------- | ----- |--------- | -------- | ----- |
| `keynet.pt`| 73.51 | 66.72 | 8.11 | 6.79 | 3.48 | 14.90 |
| Mixed In Key 8.3 | 75.70 | 69.37 | 8.11 | 5.13 | 3.64 | 13.74 |
| RekordBox 7.12 | 65.53 | 56.79 | 11.92 | 5.96 | 4.97 | 20.36 |

## Literature

Please cite and refer to the original publication for scientific use and further reading:

- \[1\] F. Korzeniowski and G. Widmer. "Genre-Agnostic Key Classification With Convolutional Neural Networks".
In: *Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR)* (2018) [arXiv](https://arxiv.org/abs/1808.05340)

- \[2\] F. Korzeniowski and G. Widmer. "End-to-End Musical Key Estimation Using a Convolutional Neural Network". In: *Proceedings of the 25th European Signal Processing Conference* (2017) [arXiv](https://arxiv.org/abs/1706.02921)
