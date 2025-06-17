import argparse
from pathlib import Path
import torch
import torchaudio
import librosa
import numpy as np

from dataset import CAMELOT_MAPPING
from eval import load_model

def parse_args():
    """
    Parses command-line arguments.
    Returns:
        args: Parsed arguments.
    """
    default_model_path = Path('checkpoints') / 'keynet.pt'
    parser = argparse.ArgumentParser(description="Predict Camelot key for single or multiple mp3 files.")
    parser.add_argument('-f', '--path', type=str, required=True,
                        help="Path to an .mp3 file or folder containing .mp3 files.")
    parser.add_argument('-m', '--model_path', type=str, default=str(default_model_path),
                        help="Path to the trained model checkpoint (.pt).")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use: 'cpu' or 'cuda'. If not given, uses CUDA if available.")
    return parser.parse_args()

def get_mp3_list(path):
    """
    Returns a list of mp3 files from a folder or a single file.
    Args:
        path (str or Path): Path to .mp3 file or directory.

    Returns:
        List[Path]: List of mp3 file Paths.

    Raises:
        ValueError: If file is not .mp3 or folder contains none.
    """
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".mp3":
            raise ValueError(f"File {path} is not a .mp3 file.")
        return [path]
    elif path.is_dir():
        files = list(path.glob("*.mp3"))
        if not files:
            raise ValueError(f"No .mp3 files found in {path}")
        return files
    else:
        raise FileNotFoundError(f"{path} is not a valid file or folder.")

def preprocess_mp3(mp3_path, sample_rate=44100, n_bins=105, hop_length=8820):
    """
    Loads an mp3, converts to mono, resamples, and extracts a log-magnitude CQT spectrogram.
    Then slices result as in MTG preprocessed dataset (removes last frequency bin and converts to torch tensor).

    Args:
        mp3_path (Path): Path to .mp3 file.
        sample_rate (int): Target sampling rate for audio.
        n_bins (int): Number of CQT bins.
        hop_length (int): Hop length for CQT.

    Returns:
        torch.Tensor: Shape (1, freq_bins, time_frames), ready for model input.
    """
    waveform, sr = torchaudio.load(mp3_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0).numpy().astype(np.float32)

    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=24, fmin=65)
    spec = np.abs(cqt)
    spec = np.log1p(spec)

    # Remove last frequency bin
    chunk = spec[:, 0:-2]
    spec_tensor = torch.tensor(chunk, dtype=torch.float32)
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)  # Shape: (1, freq, time)
    return spec_tensor

def camelot_output(pred_camelot):
    """
    Formats the Camelot prediction:
    - Indexing as in DJ software: ID (1-12) + Mode (A=minor, B=major)
    - minor: 1-12A, major: 1-12B
    Args:
        pred_camelot (int): 0-23, neural network output

    Returns:
        (str, str): camelot_str (e.g. "6A"), key_text (from CAMELOT_MAPPING, possibly two synonyms)
    """
    idx = (pred_camelot % 12) + 1        # 1-based index for wheel
    mode = "A" if pred_camelot < 12 else "B"
    camelot_str = f"{idx}{mode}"

    # fetch key string(s) for this camelot index
    names = [k for k, v in CAMELOT_MAPPING.items() if v == pred_camelot]
    if names:
        key_text = "/".join(sorted(set(names)))
    else:
        key_text = "Unknown"
    return camelot_str, key_text

def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)

    mp3_files = get_mp3_list(args.path)

    print("="*70)
    print("{:^70}".format("Key Prediction Results"))
    print("="*70)
    print(f"{'File':<28} | {'ID':^5} | {'Camelot':^8} | {'Key':^20}")
    print("-"*70)

    for mp3_path in mp3_files:
        try:
            spec_tensor = preprocess_mp3(mp3_path)
            # Torch shape: (1, freq, time); batchify and to device
            spec_tensor = spec_tensor.to(device)
            spec_tensor = spec_tensor.unsqueeze(0) if spec_tensor.ndim == 3 else spec_tensor  # Add batch dimension if needed

            with torch.no_grad():
                outputs = model(spec_tensor)
                pred = int(torch.argmax(outputs, dim=1).cpu().item())

            camelot_str, key_text = camelot_output(pred)

            print(f"{mp3_path.name:<28} | {pred:^5} | {camelot_str:^8} | {key_text:^20}")
        except Exception as e:
            print(f"Error processing {mp3_path.name}: {e}")
    
    print("="*70)
    print(f"Total files processed: {len(mp3_files)}")
    print("="*70)

if __name__ == "__main__":
    main()