
import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import transformers
import wandb

from dataset.cochlsence import get_cochlscene_training_set, get_cochlscene_val_set, get_cochlscene_test_set
from helpers.init import worker_init_fn
from helpers.utils import mixstyle
from helpers import complexity
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["WANDB_SILENT"] = "true"


def get_model_by_name(model_name, **kwargs):
    """
    Dynamically load the requested model class by name.
    
    Args:
        model_name: name of the model architecture
        **kwargs: keyword arguments forwarded to the model constructor
    
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
    PyTorch Lightning Module for training the DCASE'25 baseline model (shared across all devices).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # -------- Preprocessing Pipeline --------
        # resample & generate mel spectrogram
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

        # SpecAug
        self.mel_augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True),
            torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)
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
        # CochlScene label names (13 classes)
        self.label_ids = [
            'Bus', 'Cafe', 'Car', 'CrowdedIndoor', 'Elevator', 
            'Kitchen', 'Park', 'ResidentialArea', 'Restaurant', 
            'Restroom', 'Street', 'Subway', 'SubwayStation'
        ]
        # Grouping devices into real/seen/unseen categories
        self.device_groups = {
            'a': "real", 'b': "real", 'c': "real",
            's1': "seen", 's2': "seen", 's3': "seen",
            's4': "unseen", 's5': "unseen", 's6': "unseen"
        }

        # Containers to store step outputs (PyTorch Lightning 2.x pattern)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_save_checkpoint(self, checkpoint):
        """
        Save RNG state selectively based on checkpoint purpose:
        - last.ckpt: store full RNG state for resuming
        - best model: keep checkpoint clean without RNG state (fine-tuning)
        """
        import random
        import numpy as np
        import torch
        
        # Determine whether we are saving last.ckpt
        is_last_checkpoint = self._is_saving_last_checkpoint()
        
        if is_last_checkpoint:
            # last.ckpt: store full RNG state for resuming
            checkpoint['python_rng_state'] = random.getstate()
            checkpoint['numpy_rng_state'] = np.random.get_state() 
            checkpoint['torch_rng_state'] = torch.get_rng_state()
            
            # Save CUDA RNG state if available
            if torch.cuda.is_available():
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
                # Save RNG state for all GPUs when applicable
                if torch.cuda.device_count() > 1:
                    checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
            
            # Store PyTorch version for compatibility checks
            checkpoint['torch_version'] = torch.__version__
            checkpoint['has_rng_state'] = True
            print("Saved full RNG state to last.ckpt (resume-ready)")
        else:
            # best model: keep checkpoint clean without RNG state
            print("Saved clean checkpoint (fine-tune friendly, no RNG state)")

    def _is_saving_last_checkpoint(self):
        """
        Heuristically detect whether the current checkpoint corresponds to last.ckpt.
        Uses a simplified call-stack inspection strategy.
        """
        import inspect
        
        try:
            # Inspect function names and local variables in the call stack
            frame = inspect.currentframe()
            while frame:
                # Check function name
                code_name = frame.f_code.co_name.lower()
                if 'last' in code_name:
                    return True
                
                # Check local variables for file paths
                frame_locals = frame.f_locals
                for var_name, var_value in frame_locals.items():
                    var_str = str(var_value).lower()
                    if 'last.ckpt' in var_str or 'last_model_path' in var_str:
                        return True
                    # Evaluate variable names as well
                    if 'last' in var_name.lower() and ('path' in var_name.lower() or 'file' in var_name.lower()):
                        return True
                
                frame = frame.f_back
        except Exception:
            pass
        
        # Fallback: inspect trainer state
        try:
            if hasattr(self.trainer, 'state') and self.trainer.state:
                # If training loop is stopping or finishing, last checkpoint is likely
                state_str = str(self.trainer.state).lower()
                if 'stopping' in state_str or 'finished' in state_str:
                    return True
        except Exception:
            pass
        
        # Default assumption: we are saving the best model
        return False

    def on_load_checkpoint(self, checkpoint):
        """
        Restore RNG state when available:
        - With RNG: resume the training run
        - Without RNG: skip restoration (fine-tuning)
        """
        import random
        import numpy as np
        import torch
        
        # Check whether the checkpoint contains RNG state
        if not checkpoint.get('has_rng_state', False):
            print("Checkpoint does not include RNG state (fine-tune mode)")
            return
        
        # Warn about PyTorch version mismatch
        if 'torch_version' in checkpoint:
            saved_version = checkpoint['torch_version']
            current_version = torch.__version__
            if saved_version != current_version:
                print(f"PyTorch version mismatch: saved={saved_version}, current={current_version}")
        
        # Restore Python RNG
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
            print("Restored Python RNG state")
        
        # Restore NumPy RNG
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
            print("Restored NumPy RNG state")
            
        # Restore PyTorch CPU RNG
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
            print("Restored PyTorch CPU RNG state")
            
        # Restore CUDA RNG
        if torch.cuda.is_available():
            if 'cuda_rng_state' in checkpoint:
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
                print("Restored CUDA RNG state")
            if 'cuda_rng_state_all' in checkpoint and torch.cuda.device_count() > 1:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
                print("Restored multi-GPU CUDA RNG state")
        
        print("RNG state restoration complete; resuming training seamlessly")

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
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

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """

        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.config.lr,
        #     weight_decay=self.config.weight_decay
        # )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            # standard cosine schedule is 1, 0.4's decay is only 40% of the original.
            num_cycles=0.4,
        )
        # the scheduler is updated every step, and the frequency is 1.
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, _, labels, _, _ = train_batch
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, labels)

        # Log learning rate and epoch
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("epoch", self.current_epoch)

        # Log training loss
        self.log("train/loss", loss.detach().cpu())

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        y_hat = self.forward(x)

        # Compute loss per sample
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy 
        # argmax
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        # Initialize per-device stats
        # loss, count, correct
        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)

        # Accumulate device-wise stats
        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devcnt.{d}"] += 1
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]

        # Initialize per-label stats
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)

        # Accumulate label-wise stats
        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch. Aggregates per-device/per-label stats and logs them.
        """
        # Flatten the outputs into a dict of lists
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack each list of tensors
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        # Compute overall metrics
        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()

        logs = {"acc": acc, "loss": avg_loss}

        # Per-device stats
        for d in self.device_ids:
            dev_loss = outputs[f"devloss.{d}"].sum()
            dev_cnt = outputs[f"devcnt.{d}"].sum()
            dev_correct = outputs[f"devn_correct.{d}"].sum()
            logs[f"loss.{d}"] = dev_loss / dev_cnt
            logs[f"acc.{d}"] = dev_correct / dev_cnt
            logs[f"cnt.{d}"] = dev_cnt

            # Group stats -> real, seen, unseen
            group_name = self.device_groups[d]
            logs[f"acc.{group_name}"] = logs.get(f"acc.{group_name}", 0.) + dev_correct
            logs[f"count.{group_name}"] = logs.get(f"count.{group_name}", 0.) + dev_cnt
            logs[f"lloss.{group_name}"] = logs.get(f"lloss.{group_name}", 0.) + dev_loss

        # Reduce group stats
        for grp in set(self.device_groups.values()):
            logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
            logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]

        # Per-label stats
        for lbl in self.label_ids:
            lbl_loss = outputs[f"lblloss.{lbl}"].sum()
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()

            logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
            logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"cnt.{lbl}"] = lbl_cnt.float()

        # Compute macro-average accuracy over all labels
        logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))

        # Log key metrics to the progress bar first
        self.log("val/acc", logs["acc"], prog_bar=True, logger=True)
        self.log("val/loss", logs["loss"], prog_bar=True, logger=True)
        self.log("val/macro_avg_acc", logs["macro_avg_acc"], prog_bar=True, logger=True)
        
        # Remove already-logged key metrics to avoid duplicates
        detailed_logs = {k: v for k, v in logs.items() if k not in ['acc', 'loss', 'macro_avg_acc']}
        # Log remaining metrics with a 'val/' prefix
        self.log_dict({f"val/{k}": v for k, v in detailed_logs.items()})
        
        self.validation_step_outputs.clear()
        print('\n')

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

        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()


def train(config):
    """
        Main training loop using PyTorch Lightning.

        Args:
            config: Namespace or dictionary containing all experiment hyperparameters
                    (subset size, number of epochs, LR, etc.).
        """
    # Build experiment sub-directory path
    if config.exp_subdir:
        # Use user-provided sub-directory name
        exp_dir = os.path.join(config.checkpoint_dir, config.exp_subdir)
    else:
        # Auto-generate name: {model_type}_{seed}
        exp_dir = os.path.join(config.checkpoint_dir, f"{config.model_type}_{config.seed}")
    
    # Update config checkpoint_dir to the resolved experiment directory
    config.checkpoint_dir = exp_dir
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    
    # Configure W&B logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'25 Task 1.",
        tags=["DCASE25"],
        config=config,  # logs all hyperparameters
        name=config.experiment_name
    )

    # Dataloader for training - uses the CochlScene training split
    roll_samples = config.orig_sample_rate * config.roll_sec
    
    # Construct CochlScene training dataset with configurable DIR augmentation
    train_ds = get_cochlscene_training_set(device=None, roll=roll_samples,
                                         apply_dir=config.ir_path is not None,  # enable DIR if an IR path is provided
                                         dir_p=config.dir_p,
                                         ir_path=config.ir_path if config.ir_path else "/content/ir_files")
    train_dl = DataLoader(
        dataset=train_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory = True,
        shuffle=True,
    )

    # Dataloader for validation - CochlScene validation split
    val_ds = get_cochlscene_val_set(device=None)
    val_dl = DataLoader(
        dataset=val_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory = True
    )
    
    # Dataloader for final testing - CochlScene test split
    test_ds = get_cochlscene_test_set(device=None)
    test_dl = DataLoader(
        dataset=test_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory = True
    )

    # Create PyTorch Lightning module
    pl_module = PLModule(config)

    # Compute model complexity (MACs, parameters) and log to W&B
    sample = next(iter(val_dl))[0][0].unsqueeze(0)  # Single sample from validation set
    shape = pl_module.mel_forward(sample).size()
    macs, params_bytes = complexity.get_torch_macs_memory(pl_module.model, input_size=shape)
    wandb_logger.experiment.config["MACs"] = macs
    wandb_logger.experiment.config["Parameters_Bytes"] = round(params_bytes / 2000, 2)

    filename = f"BM-{{epoch}}-{{val/acc:.2f}}-{config.lr:.3f}-{config.warmup_steps:.0f}"

    # Use the configurable checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,              
        filename=filename, 
        save_top_k=1,
        save_last=True,  # keep the latest checkpoint for resuming
        monitor="val/acc",
        mode="max",
        enable_version_counter=False  # disable version counter to overwrite last.ckpt
    )
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val/acc", mode="max")

    # create the pytorch lightening trainer
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         gradient_clip_val = 0.5,
                         gradient_clip_algorithm="norm",                                                  
                         num_sanity_val_steps=0,
                         logger=wandb_logger,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1,
                         precision=config.precision,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback],
                         deterministic=True)
                         #callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)])

    # Resume logic: detect or use user-specified checkpoint
    ckpt_path = None
    if config.restart:
        print("Restart requested: ignoring existing checkpoints")
    elif config.resume:
        ckpt_path = config.resume
        print(f"Resuming from user-specified checkpoint: {ckpt_path}")
    else:
        # Automatically detect last.ckpt
        last_ckpt = os.path.join(config.checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
            print(f"Found checkpoint, resuming run: {ckpt_path}")

    # Fit (train + validation) with resume support
    trainer.fit(pl_module, train_dl, val_dl, ckpt_path=ckpt_path)

    # Final test on test set
    ckp_path = Path(checkpoint_callback.best_model_path)
    trainer.test(ckpt_path=ckp_path, dataloaders=test_dl)

    # Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 25 argument parser')

    # General arguments
    parser.add_argument("--project_name", type=str, default="CNN_ATTN_PRETRAIN_COCHLE") # fixed
    parser.add_argument("--experiment_name", type=str, default="Pretrain_CochlScene_v1") # fixed
    parser.add_argument("--num_workers", type=int, default=8) # fixed
    parser.add_argument("--precision", type=str, default="32") # fixed
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1) # fixed
    parser.add_argument("--orig_sample_rate", type=int, default=44100) # fixed

    # CochlScene pretraining uses the full dataset, no subset flag

    # Model selection - additional argument
    parser.add_argument("--model_type", type=str, default="CSA_Net",
                      choices=['CSA_Net', 'cnn_attn_plus', 'BiFo_Net', 
                               'cnn_cross_attn_plus', 'cnn_gru'],
                       help="Select which model architecture to train")

    # Model hyperparameters
    parser.add_argument("--n_classes", type=int, default=13) # fixed
    parser.add_argument("--in_channels", type=int, default=1) # fixed
    parser.add_argument("--base_channels", type=int, default=24,
                       help="Base channel size; set to 36 for *_plus models") # fixed
    parser.add_argument("--channels_multiplier", type=float, default=1.5) # fixed
    parser.add_argument("--expansion_rate", type=float, default=2) # fixed
    parser.add_argument('--divisor', type=float, default=5) # fixed

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=150) # fixed
    parser.add_argument("--batch_size", type=int, default=256) # fixed
    # mixed sample probability
    parser.add_argument("--mixstyle_p", type=float, default=0.4)
    # mixed sample bias, 0 close to sample1, 1 close to sample2
    parser.add_argument("--mixstyle_alpha", type=float, default=0.3) 
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    # random audio starting point
    parser.add_argument("--roll_sec", type=float, default=0.0) # fixed
    # device difference, 0 close to the same device, 1 close to different device
    parser.add_argument('--dir_p', type=float, default=0.4)  #  mic impulse response
    parser.add_argument('--ir_path', type=str, default="/content/ir_files")  # DIR augmentation: impulse response directory

    # Checkpoint and resume arguments
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/train_cochlsence_checkpoint", 
                       help="Root directory for checkpoints (subfolders created per experiment)")
    parser.add_argument("--exp_subdir", type=str, default=None,
                       help="Optional experiment sub-directory name; defaults to {model_type}_{seed}")
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to a checkpoint for manual resume")
    parser.add_argument("--restart", action="store_true", 
                       help="Force a fresh run, ignoring existing checkpoints")

    # Learning rate schedule
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--warmup_steps", type=int, default=20)

    # Spectrogram parameters
    parser.add_argument("--sample_rate", type=int, default=44100) # fixed
    # single sample length
    parser.add_argument("--window_length", type=int, default=8192) # fixed
    # sample_rate / hop_length = frames
    parser.add_argument("--hop_length", type=int, default=1364) # fixed
    # window_length decides the range of frequency resolution, and n_fft determines the actual frequency resolution.
    parser.add_argument("--n_fft", type=int, default=8192) # fixed
    parser.add_argument("--n_mels", type=int, default=256) # fixed
    parser.add_argument("--freqm", type=int, default=48) # fixed
    parser.add_argument("--timem", type=int, default=0) # fixed
    # frequency range
    parser.add_argument("--f_min", type=int, default=1) # fixed
    parser.add_argument("--f_max", type=int, default=None) # fixed
    parser.add_argument("--seed", type=int, default=1999,
                       help="Random seed (e.g., 1999, 1337, 99, 1024, 2025)")
    
    args = parser.parse_args()

    # Use the user-specified seed
    pl.seed_everything(args.seed, workers=True)

    train(args)
