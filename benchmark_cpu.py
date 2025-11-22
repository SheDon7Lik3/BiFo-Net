import os
import argparse
import torch
import torchaudio
import time
import platform
import psutil
import numpy as np
from pathlib import Path
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)


def get_model_by_name(model_name, **kwargs):
    """
    Dynamically load the model implementation by name.
    
    Args:
        model_name: model type identifier
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


def get_cpu_info():
    """Collect CPU information."""
    try:
        cpu_info = {
            'model': platform.processor(),
            'cores': psutil.cpu_count(logical=False),  # physical cores
            'logical_cores': psutil.cpu_count(logical=True),  # logical cores
        }
        # Try to retrieve more detailed CPU info per OS
        if platform.system() == 'Darwin':  # macOS
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cpu_info['model'] = result.stdout.strip()
            except:
                pass
        elif platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_info['model'] = line.split(':')[1].strip()
                            break
            except:
                pass
    except Exception as e:
        cpu_info = {'model': 'Unknown', 'cores': 'Unknown', 'logical_cores': 'Unknown'}
    
    return cpu_info


def load_audio_sample(audio_path, target_length=44100):
    """
    Load a single audio sample.
    
    Args:
        audio_path: path to the audio file
        target_length: desired length (samples), default 44100
    
    Returns:
        waveform: torch.Tensor, shape [1, target_length]
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
    
    # Adjust length
    if waveform.shape[-1] < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[-1]))
    elif waveform.shape[-1] > target_length:
        waveform = waveform[..., :target_length]
    
    return waveform


def load_model_weights(checkpoint_path, model, strict=False):
    """
    Load only the model weights from a checkpoint file.
    
    Args:
        checkpoint_path: checkpoint file path
        model: instantiated model
        strict: whether to enforce exact key matching
    
    Returns:
        model: model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Lightning checkpoints may include a 'model.' prefix in state_dict keys
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Extract model weights (remove 'model.' prefix)
        model_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_key = k[6:]  # drop 'model.' prefix
                model_state_dict[new_key] = v
        
        # If prefixed weights were found, use them
        if model_state_dict:
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=strict)
            if missing_keys and strict:
                print(f"Warning: missing weight keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Warning: unexpected weight keys: {len(unexpected_keys)}")
        else:
            # No prefixed weights, attempt direct loading (different format)
            print("Warning: no 'model.' prefixed weights found, attempting to load state_dict directly...")
            model.load_state_dict(state_dict, strict=strict)
    else:
        # Non-Lightning format, load directly
        model.load_state_dict(checkpoint, strict=strict)
    
    return model


def mel_forward(mel_transform, x):
    """
    Preprocess audio into a log-mel spectrogram.
    
    Args:
        mel_transform: torch.nn.Sequential with resample and mel transforms
        x: raw waveform [1, length]
    
    Returns:
        log-mel spectrogram
    """
    x = mel_transform(x)
    x = (x + 1e-5).log()
    return x


def benchmark_inference(model, mel_transform, sample, num_runs=100, warmup_runs=10):
    """
    Run the inference performance benchmark.
    
    Args:
        model: model instance
        mel_transform: preprocessing pipeline
        sample: input audio sample
        num_runs: number of benchmark iterations
        warmup_runs: number of warmup iterations
    
    Returns:
        dict: timing statistics
    """
    # Ensure the model runs on CPU in eval mode
    model = model.cpu()
    model.eval()
    
    # Convert to FP16 (consistent with test_tau.py)
    model = model.half()
    
    # Warmup phase
    print(f"Starting warmup phase ({warmup_runs} runs)...")
    with torch.inference_mode():
        for _ in range(warmup_runs):
            x_mel = mel_forward(mel_transform, sample)
            x_mel = x_mel.unsqueeze(1)  # add channel dim: [1, n_mels, frames] -> [1, 1, n_mels, frames]
            x_mel = x_mel.half()
            _ = model(x_mel)
    
    # Benchmark runs
    print(f"Starting benchmark phase ({num_runs} runs)...")
    preprocess_times = []
    inference_times = []
    total_times = []
    
    with torch.inference_mode():
        for i in range(num_runs):
            # Total time
            start_total = time.perf_counter()
            
            # Preprocessing time
            start_preprocess = time.perf_counter()
            x_mel = mel_forward(mel_transform, sample)
            x_mel = x_mel.unsqueeze(1)  # add channel dim
            x_mel = x_mel.half()
            preprocess_time = (time.perf_counter() - start_preprocess) * 1000  # ms
            
            # Inference time
            start_inference = time.perf_counter()
            output = model(x_mel)
            inference_time = (time.perf_counter() - start_inference) * 1000  # ms
            
            total_time = (time.perf_counter() - start_total) * 1000  # ms
            
            preprocess_times.append(preprocess_time)
            inference_times.append(inference_time)
            total_times.append(total_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs...")
    
    # Compute summary statistics
    def compute_stats(times):
        times = np.array(times)
        return {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
        }
    
    results = {
        'preprocess': compute_stats(preprocess_times),
        'inference': compute_stats(inference_times),
        'total': compute_stats(total_times),
    }
    
    return results


def print_report(config, cpu_info, results):
    """Print the benchmark report."""
    print("\n" + "="*80)
    print("CPU Inference Benchmark Report")
    print("="*80)
    
    print("\nTest Environment:")
    print(f"  - CPU Model: {cpu_info['model']}")
    print(f"  - Physical Cores: {cpu_info['cores']}")
    print(f"  - Logical Cores: {cpu_info['logical_cores']}")
    print(f"  - Python Version: {sys.version.split()[0]}")
    print(f"  - PyTorch Version: {torch.__version__}")
    
    print("\nModel Details:")
    print(f"  - Model Type: {config.model_type}")
    print(f"  - Checkpoint: {Path(config.checkpoint).name}")
    print("  - Precision: FP16 (Half)")
    
    print("\nPerformance Metrics:")
    print("\n  Single-Sample Latency:")
    print(f"    - Total: {results['total']['mean']:.2f} ms (mean) +/- {results['total']['std']:.2f} ms (std)")
    print(f"    - Preprocess: {results['preprocess']['mean']:.2f} ms (mean) +/- {results['preprocess']['std']:.2f} ms (std)")
    print(f"    - Inference: {results['inference']['mean']:.2f} ms (mean) +/- {results['inference']['std']:.2f} ms (std)")
    
    print("\n  Latency Distribution:")
    print(f"    - Fastest: {results['total']['min']:.2f} ms")
    print(f"    - Slowest: {results['total']['max']:.2f} ms")
    print(f"    - Median: {results['total']['median']:.2f} ms")
    
    total_mean = results['total']['mean']
    
    print("\nConclusion:")
    if total_mean < 50:
        status = "PASS"
        conclusion = "Model can be deployed efficiently on CPU (latency < 50 ms)."
    elif total_mean < 100:
        status = "PASS"
        conclusion = "Model can be deployed on CPU (latency < 100 ms)."
    elif total_mean < 200:
        status = "WARN"
        conclusion = "Model is deployable but may need optimization (latency 100-200 ms)."
    else:
        status = "FAIL"
        conclusion = "Model is too slow for CPU deployment (latency > 200 ms)."
    
    print(f"  [{status}] {conclusion}")
    
    print("="*80 + "\n")


def benchmark(config):
    """
    Main entry point for the CPU inference benchmark.
    
    Args:
        config: Namespace with all configuration parameters
    """
    print("\nStarting CPU inference benchmark")
    print(f"  - Model Type: {config.model_type}")
    print(f"  - Checkpoint: {config.checkpoint}")
    print(f"  - Audio Sample: {config.sample_path}")
    print(f"  - Benchmark Runs: {config.num_runs}")
    print(f"  - Warmup Runs: {config.warmup_runs}")
    
    # Gather CPU information
    cpu_info = get_cpu_info()
    
    # Load audio sample
    print("\nLoading audio sample...")
    sample = load_audio_sample(config.sample_path, target_length=44100)
    print(f"  - Sample Shape: {sample.shape}")
    print(f"  - Sample Duration: {sample.shape[-1] / 44100:.2f} seconds")
    
    # Build preprocessing pipeline
    print("\nBuilding preprocessing pipeline...")
    mel_transform = torch.nn.Sequential(
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
    mel_transform.eval()
    
    # Initialize model
    print("\nBuilding model...")
    model = get_model_by_name(
        config.model_type,
        n_classes=config.n_classes,
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        channels_multiplier=config.channels_multiplier,
        expansion_rate=config.expansion_rate,
        divisor=config.divisor
    )
    
    # Load model weights
    print("\nLoading model weights...")
    model = load_model_weights(config.checkpoint, model, strict=False)
    print("  - Model weights loaded successfully")
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = benchmark_inference(
        model, mel_transform, sample,
        num_runs=config.num_runs,
        warmup_runs=config.warmup_runs
    )
    
    # Print report
    print_report(config, cpu_info, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPU inference benchmark script')
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the model checkpoint (.ckpt file)")
    parser.add_argument("--model_type", type=str, required=True,
                      choices=['CSA_Net', 'cnn_attn_plus', 'BiFo_Net', 
                               'cnn_cross_attn_plus', 'cnn_gru'],
                       help="Model architecture to benchmark")
    parser.add_argument("--sample_path", type=str, required=True,
                       help="Path to the audio sample file")
    
    # Benchmark parameters
    parser.add_argument("--num_runs", type=int, default=1000,
                       help="Number of benchmark runs (default: 1000)")
    parser.add_argument("--warmup_runs", type=int, default=100,
                       help="Number of warmup runs (default: 100)")
    
    # Audio parameters
    parser.add_argument("--orig_sample_rate", type=int, default=44100,
                       help="Original audio sample rate (default: 44100)")
    
    # Model hyperparameters
    parser.add_argument("--n_classes", type=int, default=10,
                       help="Number of output classes (default: 10)")
    parser.add_argument("--in_channels", type=int, default=1,
                       help="Number of input channels (default: 1)")
    parser.add_argument("--base_channels", type=int, default=24,
                       help="Base channel width (default: 24)")
    parser.add_argument("--channels_multiplier", type=float, default=1.5,
                       help="Channel multiplier (default: 1.5)")
    parser.add_argument("--expansion_rate", type=float, default=2,
                       help="Expansion rate (default: 2)")
    parser.add_argument('--divisor', type=float, default=5,
                       help="Divisor for width rounding (default: 5)")
    
    # Spectrogram parameters
    parser.add_argument("--sample_rate", type=int, default=44100,
                       help="Target sample rate (default: 44100)")
    parser.add_argument("--window_length", type=int, default=8192,
                       help="Window length (default: 8192)")
    parser.add_argument("--hop_length", type=int, default=1364,
                       help="Hop length (default: 1364)")
    parser.add_argument("--n_fft", type=int, default=8192,
                       help="FFT size (default: 8192)")
    parser.add_argument("--n_mels", type=int, default=256,
                       help="Number of mel bins (default: 256)")
    parser.add_argument("--f_min", type=int, default=1,
                       help="Minimum frequency (default: 1)")
    parser.add_argument("--f_max", type=int, default=None,
                       help="Maximum frequency (default: None)")
    
    args = parser.parse_args()
    
    # Validate file paths
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if not os.path.exists(args.sample_path):
        raise FileNotFoundError(f"Audio file not found: {args.sample_path}")
    
    # Run benchmark
    benchmark(args)

