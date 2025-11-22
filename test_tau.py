import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
import warnings

from dataset.dcase25 import get_test_set
from helpers.init import worker_init_fn
from helpers import complexity

warnings.filterwarnings("ignore", category=UserWarning)


def get_model_by_name(model_name, **kwargs):
    """
    Dynamically load the model implementation by name.

    Args:
        model_name: model identifier
        **kwargs: keyword arguments forwarded to the constructor

    Returns:
        Instantiated model
    """
    model_mapping = {
        'CSA_Net': 'models.CSA_Net',
        'cnn_attn_plus': 'models.cnn_attn_plus',
        'BiFo_Net': 'models.BiFo_Net',
        'cnn_cross_attn_plus': 'models.cnn_cross_attn_plus',
        'cnn_gru': 'models.cnn_gru',
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model type: {model_name}. Supported models: {list(model_mapping.keys())}")
    
    # Dynamically import the model module
    module_name = model_mapping[model_name]
    module = __import__(module_name, fromlist=['get_ntu_model'])
    get_ntu_model = getattr(module, 'get_ntu_model')
    
    return get_ntu_model(**kwargs)


class PLModule(pl.LightningModule):
    """
    PyTorch Lightning Module for testing on TAU Urban Acoustic Scenes
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # -------- Preprocessing Pipeline --------
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=config.orig_sample_rate,
                new_freq=config.sample_rate
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.window_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max
            )
        )

        # -------- Model Selection --------
        self.model = get_model_by_name(
            config.model_type,
            n_classes=config.n_classes,
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channels_multiplier=config.channels_multiplier,
            expansion_rate=config.expansion_rate,
            divisor=config.divisor
        )

        # -------- Device/Label Definitions --------
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]
        self.device_groups = {
            'a': "real", 'b': "real", 'c': "real",
            's1': "seen", 's2': "seen", 's3': "seen",
            's4': "unseen", 's5': "unseen", 's6': "unseen"
        }

        # Container for test outputs
        self.test_step_outputs = []

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch

        # For memory constraints, switch model to half-precision
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()

        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")  

        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        # Per-device stats
        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)

        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]
            results[f"devcnt.{d}"] += 1

        # Per-label stats
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)

        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        self.test_step_outputs.append({k: v.cpu() for k, v in results.items()})

    def on_test_epoch_end(self):
        # Flatten the outputs
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack each list of tensors
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
        logs = {"acc": acc, "loss": avg_loss}

        # Device-level stats
        for d in self.device_ids:
            dev_loss = outputs[f"devloss.{d}"].sum()
            dev_cnt = outputs[f"devcnt.{d}"].sum()
            dev_correct = outputs[f"devn_correct.{d}"].sum()
            logs[f"loss.{d}"] = dev_loss / dev_cnt
            logs[f"acc.{d}"] = dev_correct / dev_cnt
            logs[f"cnt.{d}"] = dev_cnt

            # Device groups
            grp = self.device_groups[d]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + dev_correct
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt
            logs[f"lloss.{grp}"] = logs.get(f"lloss.{grp}", 0.) + dev_loss

        # Group-level stats
        for grp in set(self.device_groups.values()):
            logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
            logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]

        # Label-level stats
        for lbl in self.label_ids:
            lbl_loss = outputs[f"lblloss.{lbl}"].sum()
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
            logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
            logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"cnt.{lbl}"] = lbl_cnt

        # Macro-average accuracy over all labels
        logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))

        # Print results
        print("\n" + "="*80)
        print("TAU Urban Acoustic Scenes Test Results")
        print("="*80)
        print(f"\nOverall Metrics:")
        print(f"  - Overall Accuracy: {logs['acc']:.4f} ({logs['acc']*100:.2f}%)")
        print(f"  - Macro Avg Accuracy: {logs['macro_avg_acc']:.4f} ({logs['macro_avg_acc']*100:.2f}%)")
        print(f"  - Average Loss: {logs['loss']:.4f}")
        
        print(f"\nDevice Group Accuracy:")
        print(f"  - Real Devices: {logs['acc.real']:.4f} ({logs['acc.real']*100:.2f}%)")
        print(f"  - Seen Devices: {logs['acc.seen']:.4f} ({logs['acc.seen']*100:.2f}%)")
        print(f"  - Unseen Devices: {logs['acc.unseen']:.4f} ({logs['acc.unseen']*100:.2f}%)")
        
        print(f"\nPer-Device Accuracy:")
        for d in self.device_ids:
            print(f"  - Device {d}: {logs[f'acc.{d}']:.4f} ({logs[f'acc.{d}']*100:.2f}%) | Samples: {int(logs[f'cnt.{d}'])}")
        
        print(f"\nPer-Label Accuracy:")
        for lbl in self.label_ids:
            print(f"  - {lbl:20s}: {logs[f'acc.{lbl}']:.4f} ({logs[f'acc.{lbl}']*100:.2f}%) | Samples: {int(logs[f'cnt.{lbl}'])}")
        
        print("="*80 + "\n")
        
        self.test_step_outputs.clear()


def test(config):
    """
    Test the model on TAU Urban Acoustic Scenes test set
    
    Args:
        config: Namespace containing all test configurations
    """
    print(f"\nStarting TAU Urban Acoustic Scenes Test")
    print(f"  - Model Type: {config.model_type}")
    print(f"  - Checkpoint: {config.checkpoint}")
    print(f"  - Batch Size: {config.batch_size}")
    print(f"  - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Load test dataset
    test_ds = get_test_set(device=None)
    test_dl = DataLoader(
        dataset=test_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True
    )
    
    print(f"  - Test Samples: {len(test_ds)}")
    
    # Load model from checkpoint
    print(f"\nLoading model from checkpoint...")
    pl_module = PLModule.load_from_checkpoint(
        config.checkpoint,
        config=config,
        strict=False  # Allow loading with some parameter mismatches
    )
    
    # Compute and display model complexity
    print(f"\nComputing model complexity...")
    # Move model to CPU for complexity computation
    pl_module.cpu()
    pl_module.mel.cpu()
    sample = next(iter(test_dl))[0][0].unsqueeze(0).cpu()  # Single sample on CPU
    shape = pl_module.mel_forward(sample).size()
    
    # Convert model to half precision (FP16) for realistic parameter count
    pl_module.model.half()
    macs, params_bytes = complexity.get_torch_macs_memory(pl_module.model, input_size=shape)
    
    print(f"\nModel Complexity:")
    macs_m = macs / 1_000_000
    params_kb = params_bytes / 1024
    print(f"  - MACs: {macs_m:.2f} M")
    print(f"  - Parameters (KB): {params_kb:.2f} KB")
    
    # Create trainer for testing
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False
    )
    
    # Run test
    print(f"\nRunning inference on test set...")
    trainer.test(pl_module, test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TAU Urban Acoustic Scenes Test Script')

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the model checkpoint (.ckpt file)")
    parser.add_argument("--model_type", type=str, required=True,
                      choices=['CSA_Net', 'cnn_attn_plus', 'BiFo_Net', 
                               'cnn_cross_attn_plus', 'cnn_gru'],
                       help="Model architecture to evaluate")

    # General arguments
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of dataloader worker processes")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--orig_sample_rate", type=int, default=44100,
                       help="Original audio sample rate")

    # Model hyperparameters
    parser.add_argument("--n_classes", type=int, default=10,
                       help="Number of classes (TAU has 10 scene classes)")
    parser.add_argument("--in_channels", type=int, default=1,
                       help="Number of input channels")
    parser.add_argument("--base_channels", type=int, default=24,
                       help="Base channel count")
    parser.add_argument("--channels_multiplier", type=float, default=1.5,
                       help="Channels multiplier")
    parser.add_argument("--expansion_rate", type=float, default=2,
                       help="Expansion rate")
    parser.add_argument('--divisor', type=float, default=5,
                       help="Divisor")

    # Spectrogram parameters
    parser.add_argument("--sample_rate", type=int, default=44100,
                       help="Target sample rate")
    parser.add_argument("--window_length", type=int, default=8192,
                       help="Window length")
    parser.add_argument("--hop_length", type=int, default=1364,
                       help="Hop length")
    parser.add_argument("--n_fft", type=int, default=8192,
                       help="Number of FFT points")
    parser.add_argument("--n_mels", type=int, default=256,
                       help="Number of Mel bands")
    parser.add_argument("--f_min", type=int, default=1,
                       help="Minimum frequency")
    parser.add_argument("--f_max", type=int, default=None,
                       help="Maximum frequency")
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Run test
    test(args)

