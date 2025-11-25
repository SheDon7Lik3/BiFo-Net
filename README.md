# Training Guide

## 1. Prerequisites
- Recommended environment: Google Colab with an L4 GPU runtime.
- Run the following commands to satisfy all system and Python dependencies:
  ```bash
  !apt-get update && apt-get install -y sox libsox-fmt-all
  !sudo apt install aria2
  !pip install sox
  !pip install pytorch_lightning
  !pip install torchinfo
  !pip install av
  !pip install onnx
  !pip install onnxscript
  !pip install GPUtil
  !pip install audb
  !pip install torchcodec
  ```

## 2. Dataset Preparation

### 2.1 Directory Layout
Set the `dataset_dir` constants at the top of `dataset/dcase25.py` and `dataset/cochlsence.py` to the absolute paths of your local datasets. For example:

```python
# dataset/dcase25.py
dataset_dir = "/mnt/datasets/TAU-urban-acoustic-scenes-2022-mobile-development"

# dataset/cochlsence.py
dataset_dir = "/mnt/datasets/CochlScene_1s_middle_simple"
```

### 2.2 Optional DIR Augmentation Assets
- Collect impulse responses (`*.wav`) into a folder (default `/content/ir_files`).
- Pass `--ir_path /absolute/path/to/irs` when invoking training to enable device impulse response convolution.

## 3. BiFoNet Training & Testing (TAU/DCASE25)

```bash

# Train BiFoNet on the TAU
python train_tau.py \
  --model_type BiFo_Net \
  --seed 1999

# Evaluate on the TAU
python test_tau.py \
  --model_type BiFo_Net \
  --checkpoint /mnt/exp/checkpoints/bifonet_tau25_seed1999/best.ckpt
```

## 4. BiFoNet Training & Testing (CochlScene benchmark)


```bash

# Train BiFoNet on CochlScene
python train_cochlsence.py \
  --model_type BiFo_Net \
  --seed 1999

# Evaluate on the CochlScene
python test_cochlsence.py \
  --model_type BiFo_Net \
  --checkpoint /mnt/exp/checkpoints/bifonet_cochl_seed1999/best.ckpt
```
