# Evaluation Results: GiantSteps Dataset – Commercial Key Detection
This folder contains the musical key classification results for the GiantSteps dataset using two commercial software products:

- Mixed In Key (Version 10.3)
- rekordbox (Version 7.12)

Both tools were used to estimate the keys for all provided tracks. The output is given in two .txt files, one for each software, with identical formatting.

## File Contents
Each .txt file contains one entry per song of the GiantSteps dataset (in the same order for easy comparison).
The information per entry is tab-separated and structured as follows:

- Song ID: The original identifier for the track (i.e., file number).
- Class ID: The key class index, corresponding to the format used by the neural network in this repository (i.e., 0-23).
- Camelot notation: The predicted key label in the Camelot Wheel notation (e.g., 8A, 4B).
- Key name: The human-readable key (e.g., A minor, C# major).

Example snippet:
```
4846004	9	10A	B minor
5740146	1	2A	D# minor
...
```

## Files

- `mixedinkey_10.3.txt`
Results as estimated by Mixed In Key (version 10.3).

- `rekordbox_7.12.txt`
Results as estimated by rekordbox (version 7.12).

## Usage
You can use these files for:

- Direct comparison with the neural network predictions from this repository
- Competitive evaluation or benchmarking

## Citation
If you use these software outputs as baselines or in publications, please cite the respective commercial tools according to their vendor guidelines.