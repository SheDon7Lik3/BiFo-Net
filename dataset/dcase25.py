import os
import pandas as pd
import torch
import torchaudio
import numpy as np
import random
from pathlib import Path
from scipy.signal import convolve
from torch.utils.data import Dataset
from torch.hub import download_url_to_file
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List
from audio_processor import get_processor_default


# Dev dataset path
dataset_dir = "/content/TAU-urban-acoustic-scenes-2022-mobile-development"
assert dataset_dir, "Specify 'TAU Urban Acoustic Scenes 2022 Mobile' dataset location in 'dataset_dir'. Download from: https://zenodo.org/record/6337421"

# Dataset configuration
dataset_config = {
    "dataset_name": "tau25",
    "meta_csv": os.path.join(dataset_dir, "meta.csv"),
    "split_path": "split_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "eval_dir": None,  # Evaluation set release on 1st of June
    "eval_fold_csv": None
}


class DCASE25Dataset(Dataset):
    """
    DCASE'25 Dataset: Loads metadata and provides access to audio samples.
    """

    def __init__(self, meta_csv: str):
        """
        Initializes the dataset.

        Args:
            meta_csv (str): Path to the dataset metadata CSV file.
        """
        df = pd.read_csv(meta_csv, sep="\t")
        # Get the filename column 2 Numpy
        self.files = df["filename"].values
        self.devices = df["source_label"].values
        # Get city column & give it a number label
        self.cities = LabelEncoder().fit_transform(df["identifier"].apply(lambda loc: loc.split("-")[0]))
        # This function does the same as the last line
        self.labels = torch.tensor(LabelEncoder().fit_transform(df["scene_label"]), dtype=torch.long)

    def __getitem__(self, index: int):
        """Loads an audio sample and corresponding metadata."""
        audio_path = os.path.join(dataset_dir, self.files[index])
        waveform, _ = torchaudio.load(audio_path)
        
        # Normalize audio length to 44100 samples
        if waveform.shape[-1] == 44032:  # length used by the generator
            waveform = torch.nn.functional.pad(waveform, (0, 68))  # pad to 44100
        elif waveform.shape[-1] != 44100:  # catch any other lengths
            if waveform.shape[-1] < 44100:
                waveform = torch.nn.functional.pad(waveform, (0, 44100 - waveform.shape[-1]))
            else:
                waveform = waveform[..., :44100]
                
        # Return the waveform, filename, label, device, city 
        return waveform, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self) -> int:
        return len(self.files)


class SubsetDataset(Dataset):
    """
    A dataset that selects a subset of samples based on given indices.
    """

    def __init__(self, dataset: Dataset, indices: List[int]):
        # origin dataset, its content are the waveform, filename, label, device, city
        self.dataset = dataset # origin dataset
        self.indices = indices # indices of the subset

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]] # return the subset of the dataset

    def __len__(self) -> int:
        return len(self.indices)


class TimeShiftDataset(Dataset):
    """
    A dataset implementing time shifting of waveforms.
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
        fsplit = file.rsplit("-", 1)
        device = fsplit[1][:-4]  # derive device ID from the filename

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


# --- Dataset Loading Functions --- #

def download_split_file(split_name: str):
    """Downloads dataset split files if not available."""
    os.makedirs(dataset_config["split_path"], exist_ok=True)
    split_file = os.path.join(dataset_config["split_path"], split_name)
    if not os.path.isfile(split_file):
        print(f"Downloading {split_name}...")
        download_url_to_file(dataset_config["split_url"] + split_name, split_file)
    return split_file


def get_dataset_split(meta_csv: str, split_csv: str, device: Optional[str] = None) -> Dataset:
    """
    Filters the dataset based on the given split file and optionally by device.
    """
    meta = pd.read_csv(meta_csv, sep="\t")
    # Get the filename of the subset
    split_files = pd.read_csv(split_csv, sep="\t")["filename"].values
    # Get the indices of the subset from the origin dataset
    subset_indices = meta[meta["filename"].isin(split_files)].index.tolist()
    if device:
        # select the indices of the specific device
        subset_indices = meta.loc[subset_indices, :].query("source_label == @device").index.tolist()
    return SubsetDataset(DCASE25Dataset(meta_csv), subset_indices)


def get_training_set(split: int = 100, device: Optional[str] = None, roll: int = 0, 
                    apply_dir: bool = False, dir_p: float = 0.4, ir_path: str = "/content/ir_files") -> Dataset:
    """
    Returns the training dataset for a specified data split percentage.

    Args:
        split (int): Percentage of the dataset to use [5, 10, 25, 50, 100].
        device (Optional[str]): Specific device to filter on.
        roll (int): Time shift range.
        apply_dir (bool): Whether to apply DIR augmentation.
        dir_p (float): Probability of applying DIR augmentation.
        ir_path (str): Path to impulse response files.

    Process: Get split info -> DCASE25Dataset -> SubsetDataset -> TimeShiftDataset -> DIRAugmentDataset
    """
    assert str(split) in ("5", "10", "25", "50", "100"), "split must be in {5, 10, 25, 50, 100}"
    # download the split file
    subset_file = download_split_file(f"split{split}.csv")
    # subset_file = "/content/ProJ-3/Tan_SNTLNTU_task1/csv/split25.csv"
    dataset = get_dataset_split(dataset_config["meta_csv"], subset_file, device)
    
    # Apply time-shift augmentation
    if roll:
        dataset = TimeShiftDataset(dataset, shift_range=roll)
    
    # Apply DIR augmentation
    dataset = add_dir_augment_ds(dataset, sr=44100, apply=apply_dir, prob=dir_p, ir_path=ir_path)
    
    return dataset


def get_test_set(device: Optional[str] = None) -> Dataset:
    """
    Returns the test dataset.
    Process: Get split info -> DCASE25Dataset -> SubsetDataset
    """
    test_split_file = download_split_file(dataset_config["test_split_csv"])
    return get_dataset_split(dataset_config["meta_csv"], test_split_file, device)
