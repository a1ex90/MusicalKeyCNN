from pathlib import Path
from dataset import CAMELOT_MAPPING
from tqdm import tqdm
import torchaudio
import librosa
import numpy as np
import pickle

def preprocess_data(dataset_dir, output_dir, pitch_range = (-4, 7)):
    """
    Preprocesses the MTG/GiantSteps Key Dataset for key classification, as in
    Korzeniowski & Widmer (2018).

    The function:
      - Loads all high-confidence audio files and corresponding key labels,
      - For each audio file, generates multiple pitch-shifted versions (data augmentation),
      - Computes the Constant-Q Transform (CQT) log-magnitude spectrogram with Librosa,
      - Stores results as .pkl files for efficient training use.

    Key differences to the original paper:
      - Uses librosa's `cqt` (Constant-Q Transform) extractor instead of approximation
        from Korzeniowski & Widmer (2017)
      - Parameterization of CQT (n_bins=105, bins_per_octave=24, fmin=65 Hz)

    Args:
        dataset_dir (Path): Path to original MTG dataset with 'audio' and 'annotations' folders.
        output_dir (Path): Target directory for preprocessed .pkl spectrogram files.
        pitch_range (): Augmentation range: semitone shifts
    """
    output_dir.mkdir(exist_ok=True)
    audio_dir = Path(dataset_dir) / 'audio'
    annotations_path = Path(dataset_dir) / 'annotations' / 'annotations.txt'
    sample_rate = 44100
    n_bins = 105             # Number of CQT bins: covers range with high resolution
    hop_length = 8820        # Large hop (approx. 0.2 sec at 44100 Hz, ~5 FPS) as in paper for global context
    data = []

    # 1. Gather all high-confidence audio files and Camelot-encoded labels
    with open(annotations_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                file_num, key_str, confidence = parts[0], parts[1], int(parts[2])
                if key_str in CAMELOT_MAPPING and confidence == 2:
                    camelot_idx = CAMELOT_MAPPING[key_str]
                    filename = f'{file_num}.LOFI.mp3'
                    if (dataset_dir / 'audio' / filename).exists():
                        data.append((filename, camelot_idx))

    # 2. Iterate over files and create spectral representations for all pitch shifts
    for file, _ in tqdm(data):
        filepath = audio_dir / file
        waveform, sr = torchaudio.load(filepath)
        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to target sample rate if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0).numpy()   # Convert tensor to numpy array for librosa

        # For each pitch shift in the augmentation window
        for n_steps in range(pitch_range[0], pitch_range[1] + 1):
            out_file = output_dir / f'{file[:-4]}_{n_steps}.pkl'
            if out_file.exists():
                continue
            # Apply pitch shift except when n_steps==0 (original)
            if n_steps != 0:
                shifted_waveform = librosa.effects.pitch_shift(waveform.astype(np.float32), sr=sample_rate, n_steps=n_steps)
            else:
                shifted_waveform = waveform
            # Compute CQT (log-frequency spectrogram), following the paper's input representation
            cqt = librosa.cqt(
                shifted_waveform,
                sr=sample_rate,
                hop_length=hop_length,
                n_bins=n_bins,
                bins_per_octave=24,
                fmin=65,                       # Lowest frequency bin (Hz)
            )
            spec = np.abs(cqt)                 # Only magnitude is used
            spec = np.log1p(spec)              # Log-magnitude for dynamic range compression, as in the paper
            # Save as pickled numpy array for later efficient loading during training
            with open(out_file, 'wb') as f:
                pickle.dump(spec, f)

def create_annotations_txt(dataset_dir):
    """
    Creates an 'annotations.txt' file for the GiantSteps dataset in the same
    format as the MTG (GiantSteps-MTG) dataset, enabling unified preprocessing 
    and data loading.

    For GiantSteps, original labels are in individual .key files (per track).
    Unlike MTG, there is no explicit confidence indicator.
    To mimic MTG format, all entries are assigned high confidence (2).

    The generated file will have one header line and then tab-separated lines:
      ID    MANUAL KEY    C
    where C = 2 for all entries (high confidence).

    Args:
        dataset_dir (str or Path): Path to GiantSteps dataset root (must contain
                                   'annotations/giantsteps/' with .key files).
    """
    dataset_dir = Path(dataset_dir)
    # Find all .key files in the standard GiantSteps annotation folder
    files = (dataset_dir / 'annotations' / 'giantsteps').glob("*.key")

    data = []

    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            # Only process files with exactly two lines (ID + key information)
            if len(lines) != 2:
                continue
            l = lines[1]
            parts = l.strip().split(" ")
            # The second line is expected to contain key information as four space-separated fields
            if len(parts) != 4:
                print(parts)
                continue
            key_str = f'{parts[2]} {parts[3]}'
            # The file number (ID) is extracted from the file name by stripping "_LOFI" and extension
            file_num = file.stem[:-5]
            data.append([file_num, key_str])

    # Write the unified annotations.txt file, matching the MTG format for workflow compatibility
    with open(dataset_dir / 'annotations' / 'annotations.txt', 'w') as writer:
        writer.writelines(['ID\tMANUAL KEY\tC\n'])
        for d in data:
            # Each line: file_num, key_str, confidence=2 ('high')
            writer.writelines([f'{d[0]}\t{d[1]}\t2\n'])

if __name__ == '__main__':
    # --- This part is needed for training ---
    # Set input/output folders
    dataset_dir = Path('Dataset') / 'giantsteps-mtg-key-dataset'
    output_dir = Path('Dataset') / 'mtg-preprocessed-audio'
    preprocess_data(dataset_dir, output_dir)
    # --- This part is needed for evaluation ---
    # Set input/output folders
    dataset_dir = Path('Dataset') / 'giantsteps-key-dataset'
    output_dir = Path('Dataset') / 'giantsteps-preprocessed-audio'
    create_annotations_txt(dataset_dir)
    preprocess_data(dataset_dir, output_dir, pitch_range=(0,0))