import os
import pandas as pd
import torch
import torchaudio
import numpy as np
import random
from pathlib import Path
from scipy.signal import convolve
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List
from audio_processor import get_processor_default


# CochlScene dataset path
dataset_dir = "/content/CochlScene_1s_middle_simple"
# Dataset path validation happens at runtime to avoid import failures

# CochlScene 13-class label set
COCHLSCENE_LABELS = [
    'Bus', 'Cafe', 'Car', 'CrowdedIndoor', 'Elevator', 
    'Kitchen', 'Park', 'ResidentialArea', 'Restaurant', 
    'Restroom', 'Street', 'Subway', 'SubwayStation'
]

# Dataset configuration
dataset_config = {
    "dataset_name": "cochlscene",
    "dataset_dir": dataset_dir,
    "train_csv": os.path.join(dataset_dir, "train.csv"),
    "val_csv": os.path.join(dataset_dir, "val.csv"),
    "test_csv": os.path.join(dataset_dir, "test.csv"),
    "metadata_csv": os.path.join(dataset_dir, "metadata.csv"),
}


class CochlSceneDataset(Dataset):
    """
    CochlScene Dataset: loads audio samples and metadata.
    """

    def __init__(self, meta_csv: str):
        """
        Initialize the CochlScene dataset.

        Args:
            meta_csv (str): Path to the metadata CSV file.
        """
        # Validate dataset directory
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"CochlScene dataset not found at {dataset_dir}. Please check the path.")
        
        # Validate CSV file
        if not os.path.exists(meta_csv):
            raise FileNotFoundError(f"Metadata CSV file not found at {meta_csv}")
            
        df = pd.read_csv(meta_csv, sep="\t")
        
        # Extract filenames, devices, and identifiers
        self.files = df["filename"].values
        self.devices = df["source_label"].values
        
        # Encode identifier (city) information
        self.cities = LabelEncoder().fit_transform(df["identifier"])
        
        # Encode scene labels with a fixed 13-class order
        label_encoder = LabelEncoder()
        label_encoder.fit(COCHLSCENE_LABELS)  # ensure consistent label order
        self.labels = torch.tensor(label_encoder.transform(df["scene_label"]), dtype=torch.long)
        
        # Keep the encoder for downstream use
        self.label_encoder = label_encoder

    def __getitem__(self, index: int):
        """Load an audio sample and its metadata."""
        audio_path = os.path.join(dataset_dir, self.files[index])
        waveform, _ = torchaudio.load(audio_path)
        
        # Audio clips are 1 second long; enforce a 44100-sample length
        target_length = 44100
        if waveform.shape[-1] != target_length:
            if waveform.shape[-1] < target_length:
                # Pad shorter clips with zeros
                waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[-1]))
            else:
                # Trim longer clips to the first 44100 samples
                waveform = waveform[..., :target_length]
                
        # Return waveform, filename, label, device, and city
        return waveform, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self) -> int:
        return len(self.files)


class SubsetDataset(Dataset):
    """
    Dataset wrapper that exposes a subset of samples by index.
    """

    def __init__(self, dataset: Dataset, indices: List[int]):
        # Underlying dataset stores waveform, filename, label, device, and city
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]] # return the subset of the dataset

    def __len__(self) -> int:
        return len(self.indices)


class TimeShiftDataset(Dataset):
    """
    Dataset wrapper that applies random time shifting.
    """

    def __init__(self, dataset: Dataset, shift_range: int, axis: int = 1):
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index: int):
        waveform, file, label, device, city = self.dataset[index]
        shift = np.random.randint(-self.shift_range, self.shift_range + 1)
        return waveform.roll(shift, self.axis), file, label, device, city

    def __len__(self) -> int:
        return len(self.dataset)


def load_dirs(dirs_path, sr, cut_dirs_offset=None):
    """Load and preprocess impulse-response files."""
    all_paths = [path for path in Path(os.path.expanduser(dirs_path)).rglob('*.wav')]
    all_paths = sorted(all_paths)

    if cut_dirs_offset is not None:
        all_paths = all_paths[cut_dirs_offset:cut_dirs_offset + 10]
    all_paths_name = [str(p).rsplit("/", 1)[-1] for p in all_paths]

    print("Using these impulse-response files:")
    for i in range(len(all_paths_name)):
        print(i, ": ", all_paths_name[i])

    audio_processor = get_processor_default(sr=44100,resample_only=True)
    return [audio_processor(str(p)) for p in all_paths]


class DIRAugmentDataset(Dataset):
    """
    Augment audio with a Device Impulse Response (DIR).
    """
    def __init__(self, ds, dirs, prob):
        self.ds = ds
        self.dirs = dirs  # preprocessed IR list
        self.prob = prob

    def __getitem__(self, index):
        x, file, label, device, city = self.ds[index]
        # Device is provided directly via the CSV metadata

        if device == 'a':
            if torch.rand(1) < self.prob:
                dir_idx = int(np.random.randint(0, len(self.dirs)))
                dir = self.dirs[dir_idx]
                # Ensure convolution inputs are numpy arrays
                x_np = x.numpy() if torch.is_tensor(x) else x
                dir_np = dir.numpy() if torch.is_tensor(dir) else dir
                
                # Perform the convolution
                x_convolved = convolve(x_np, dir_np, 'full')[:, :x_np.shape[1]]
                
                # Convert back to a tensor
                x_convolved = torch.from_numpy(x_convolved).float()
                return x_convolved, file, label, device, city

        return x, file, label, device, city

    def __len__(self):
        return len(self.ds)


def add_dir_augment_ds(ds, sr, apply=False, prob=0.4, ir_path="/content/ir_files"):
    """Optionally wrap a dataset with DIR augmentation."""
    if not apply:
        return ds
    return DIRAugmentDataset(ds, load_dirs(ir_path, sr=sr), prob)


# --- CochlScene Dataset Loading Functions --- #

def get_cochlscene_dataset_split(meta_csv: str, device: Optional[str] = None) -> Dataset:
    """
    Load the CochlScene dataset based on a metadata CSV, optionally filtering by device.
    
    Args:
        meta_csv (str): Path to the metadata CSV file.
        device (Optional[str]): Optional device filter.
    
    Returns:
        Dataset: Filtered or full CochlScene dataset.
    """
    dataset = CochlSceneDataset(meta_csv)
    
    if device:
        # If a device is specified, build a filtered subset
        meta = pd.read_csv(meta_csv, sep="\t")
        subset_indices = meta.query("source_label == @device").index.tolist()
        return SubsetDataset(dataset, subset_indices)
    
    return dataset


def get_cochlscene_training_set(device: Optional[str] = None, roll: int = 0, 
                               apply_dir: bool = False, dir_p: float = 0.4, 
                               ir_path: str = "/content/ir_files") -> Dataset:
    """
    Return the CochlScene training dataset with optional augmentations.
    
    Args:
        device (Optional[str]): Optional device filter.
        roll (int): Time-shift range.
        apply_dir (bool): Whether to apply DIR augmentation.
        dir_p (float): Probability of applying DIR augmentation.
        ir_path (str): Path to impulse-response files.
    
    Returns:
        Dataset: Prepared CochlScene training dataset.
    """
    dataset = get_cochlscene_dataset_split(dataset_config["train_csv"], device)
    
    # Apply time-shift augmentation
    if roll:
        dataset = TimeShiftDataset(dataset, shift_range=roll)
    
    # Apply DIR augmentation
    if dir_p > 0:
        dataset = add_dir_augment_ds(dataset, sr=44100, apply=apply_dir, prob=dir_p, ir_path=ir_path)
    
    return dataset


def get_cochlscene_val_set(device: Optional[str] = None) -> Dataset:
    """
    Return the CochlScene validation dataset.
    
    Args:
        device (Optional[str]): Optional device filter.
    
    Returns:
        Dataset: Validation subset.
    """
    return get_cochlscene_dataset_split(dataset_config["val_csv"], device)


def get_cochlscene_test_set(device: Optional[str] = None) -> Dataset:
    """
    Return the CochlScene test dataset.
    
    Args:
        device (Optional[str]): Optional device filter.
    
    Returns:
        Dataset: Test subset.
    """
    return get_cochlscene_dataset_split(dataset_config["test_csv"], device)


# --- Compatibility wrappers (used for TAU22) --- #
def get_training_set(device: Optional[str] = None, roll: int = 0, 
                    apply_dir: bool = False, dir_p: float = 0.4, ir_path: str = "/content/ir_files") -> Dataset:
    """
    Compatibility function: return the CochlScene training dataset (for pretraining).
    CochlScene uses the entire training set and does not support split parameters.
    """
    return get_cochlscene_training_set(device=device, roll=roll, apply_dir=apply_dir, 
                                     dir_p=dir_p, ir_path=ir_path)


def get_test_set(device: Optional[str] = None) -> Dataset:
    """
    Compatibility function: return the CochlScene validation dataset (for pretraining validation).
    """
    return get_cochlscene_val_set(device=device)
