# Training Guide

End-to-end instructions for reproducing CochlScene pre-training and TAU/DCASE25 fine-tuning runs.

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
Both dataset loaders expect absolute paths defined at the top of `dataset/dcase25.py` and `dataset/cochlsence.py`.

```python
dataset_dir = "/content/TAU-urban-acoustic-scenes-2022-mobile-development"
dataset_dir = "/content/CochlScene_1s_middle_simple"
```

Update these to your local storage (e.g., `/mnt/datasets/TAU2022`):
- Edit the constants directly, or
- Keep defaults and create symlinks: `ln -s /mnt/datasets/TAU2022 /content/TAU-urban-acoustic-scenes-2022-mobile-development`.

Each directory must match the official release structure:
- **TAU**: `meta.csv`, `split_setup/*.csv`, `audio/train/device_x/...`
- **CochlScene**: `train.csv`, `val.csv`, `test.csv`, `metadata.csv`, and `audio/*.wav`

### 2.2 Optional DIR Augmentation Assets
- Collect impulse responses (`*.wav`) into a folder (default `/content/ir_files`).
- Pass `--ir_path /absolute/path/to/irs` when invoking training to enable device impulse response convolution.

## 3. Common Configuration Flags
| Flag | Meaning |
|------|---------|
| `--model_type` | One of `CSA_Net`, `cnn_attn_plus`, `BiFo_Net`, `cnn_cross_attn_plus`, `cnn_gru`. |
| `--base_channels` | Set to `36` for `*_plus` backbones, `24` otherwise. |
| `--mixstyle_p`, `--mixstyle_alpha` | Enable/shape MixStyle augmentation in the Mel domain. |
| `--roll_sec` | Random waveform roll seconds (time-shift). |
| `--dir_p`, `--ir_path` | Probability and IR directory for DIR augmentation (applies to device `a`). |
| `--precision` | Lightning precision string (`32`, `16-mixed`, `bf16-mixed`). |
| `--check_val_every_n_epoch` | Validation frequency in epochs. |
| `--lr`, `--warmup_steps` | Adam + cosine schedule hyperparameters. |
| `--batch_size`, `--n_epochs` | Training budget. Reduce batch size if you hit OOM. |

All other arguments are declared near the bottom of the respective training scripts—run `python train_*.py --help` for the full list.

## 4. CochlScene Pre-Training

`train_cochlsence.py` uses the full 13-class dataset with train/val/test splits.

1. **Set checkpoint root** (defaults to `/content/drive/...`). Override with `--checkpoint_dir` to any writable path. The script auto-creates a run subfolder (`{model_type}_{seed}`) unless `--exp_subdir` is provided.
2. **Resume policy**:
   - Existing `last.ckpt` in the run folder is auto-loaded.
   - Use `--resume /abs/path/to/ckpt` to target a specific checkpoint.
   - Add `--restart` to ignore on-disk checkpoints.
3. **Launch training** (example):
   ```bash
   cd /Users/metro/Downloads/ProJ-3
   python train_cochlsence.py \
     --model_type cnn_gru \
     --base_channels 24 \
     --batch_size 192 \
     --lr 0.0015 \
     --checkpoint_dir /mnt/exp/checkpoints \
     --exp_subdir cochl_cnn_gru_seed1999 \
     --ir_path /mnt/exp/irs
   ```
4. **Monitoring**:
   - Key metrics (`val/acc`, `val/macro_avg_acc`) stream to WandB and stdout.
   - Checkpoints: best model (by `val/acc`) plus `last.ckpt` for resuming.
5. **Testing**: After training, the script automatically evaluates the best checkpoint on the official test split and logs device/label breakdowns.

## 5. TAU / DCASE25 Fine-Tuning

`train_tau.py` assumes the official 25 % subset (competition rule). Ensure `--subset 25` remains unchanged.

Example command:
```bash
cd /Users/metro/Downloads/ProJ-3
python train_tau.py \
  --model_type CSA_Net \
  --base_channels 24 \
  --batch_size 256 \
  --lr 0.002 \
  --warmup_steps 50 \
  --roll_sec 0.25 \
  --dir_p 0.3 \
  --ir_path /mnt/exp/irs
```

Key differences vs. CochlScene:
- Uses TAU-provided split CSVs; if missing, they are downloaded to `split_setup/`.
- Only training + *test* loader (validation happens via the same test set for logging).
- Checkpoints are written to `/content/checkpoints` (adjust path inside the script if desired).
- Device and label names correspond to the 10 DCASE classes.

## 6. Evaluation Scripts
- `test_cochlsence.py` and `test_tau.py` mirror the training data flow but load an already-trained checkpoint. Use them to re-score models without re-running training (see each script’s `--help` for options).

## 7. Troubleshooting & Tips
- **OOM errors**: lower `--batch_size`, switch to `--precision 16-mixed`, or disable MixStyle (`--mixstyle_p 0`).
- **Dataset path errors**: double-check the absolute paths in dataset modules; the loaders raise explicit `FileNotFoundError` when CSV/audio missing.
- **IR augmentation speed**: preloading many IR files can be slow. Trim the list by editing `load_dirs(..., cut_dirs_offset=...)` or storing only the IRs you need.
- **Reproducibility**: both scripts set `pl.seed_everything(args.seed, workers=True)`; change `--seed` to spawn new runs.
- **WandB opt-out**: set `WANDB_MODE=offline` or `WANDB_DISABLED=true` before launching if you cannot log remotely. Metrics will still print locally.

You should now be able to launch pre-training on CochlScene, fine-tune on TAU/DCASE25, and reproduce the reported metrics end to end.
