from pathlib import Path
from tqdm import tqdm
import torch
from dataset import KeyDataset
from torch.utils.data import DataLoader
from model import KeyNet

def load_model(model_path, device, num_classes=24, in_channels=1, Nf=20):
    """
    Loads a pretrained KeyNet model from disk.

    Args:
        model_path (str or Path): Path to the saved model weights.
        device (torch.device): Target device (CPU or CUDA).
        num_classes (int): Number of key classes (default: 24).
        in_channels (int): Input channels (default: 1).
        Nf (int): Number of feature maps in first convolution.

    Returns:
        KeyNet: Loaded model, ready for evaluation.
    """
    model = KeyNet(num_classes=num_classes, in_channels=in_channels, Nf=Nf).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def mirex_category(pred_idx, gt_idx):
    """
    Computes the key evaluation category for a single prediction.

    Args:
        pred_idx (int): Predicted Camelot key index.
        gt_idx (int): Ground truth Camelot key index.

    Returns:
        str: One of 'correct', 'fifth', 'relative', 'parallel', 'rest'.
    """
    if pred_idx == gt_idx:
        return "correct"

    # Determine mode: first 12 = minor, last 12 = major
    t_pred, t_gt = pred_idx, gt_idx
    mode_match = False
    if gt_idx > 11 and pred_idx > 11:
        mode_match = True
        t_pred -= 12
        t_gt -= 12
    elif gt_idx < 12 and pred_idx < 12:
        mode_match = True

    diff = min(t_pred, t_gt) - max(t_pred, t_gt)

    if mode_match and (diff == -1 or diff == -11):
        return "fifth"
    if diff == -12:
        return "relative"
    if diff == -15:
        return "parallel"
    return "rest"

def evaluate_mirex(model, dataloader, device):
    """
    Runs the model on the full dataset and computes MIREX score.

    Args:
        model (nn.Module): Trained model for evaluation.
        dataloader (DataLoader): DataLoader for GiantSteps.
        device (torch.device): Device for model/inference.

    Returns:
        dict: Aggregated ratios for each category and weighted mirex score.
    """
    counts = {"correct": 0, "fifth": 0, "relative": 0, "parallel": 0, "rest": 0}
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['spec'].to(device)
            labels = batch['gt_id'].to(device)

            outputs = model(inputs)
            predicted_camelot = torch.argmax(outputs, dim=1).item()
            gt_camelot = labels.item()

            cat = mirex_category(predicted_camelot, gt_camelot)
            counts[cat] += 1
            total += 1

    # Calculate normalized scores
    scores = {k: v/total for k, v in counts.items()}
    weighted = (
        scores["correct"] +
        0.5 * scores["fifth"] +
        0.3 * scores["relative"] +
        0.2 * scores["parallel"]
    )
    scores["weighted"] = weighted
    scores["total"] = total
    return scores

def print_mirex_report(scores):
    """
    Neatly prints the percentage ratios and the weighted Mirex score.
    Args:
        scores (dict): Dictionary of MIREX evaluation scores.
    """
    print("="*40)
    print("{:^40}".format("MIREX Key Evaluation Results"))
    print("="*40)
    print(f"{'Category':<12} | {'Score':>7}")
    print("-"*40)
    print(f"{'Correct':<12} | {scores['correct']*100:.2f}")
    print(f"{'Fifth':<12} | {scores['fifth']*100:.2f}")
    print(f"{'Relative':<12} | {scores['relative']*100:.2f}")
    print(f"{'Parallel':<12} | {scores['parallel']*100:.2f}")
    print(f"{'Other':<12} | {scores['rest']*100:.2f}")
    print("-"*40)
    print(f"{'Weighted':<12} | {scores['weighted']*100:.2f}")
    print("="*40)
    print(f"Samples evaluated: {scores['total']}")
    print("="*40)

def main():
    root_dir = Path('Dataset/giantsteps-key-dataset')
    preprocessed_dir = Path('Dataset/giantsteps-preprocessed-audio')
    model_file_path = Path('checkpoints') / 'keynet.pt'  # Change as needed

    dataset = KeyDataset(root_dir, preprocessed_dir, chunk_samples=float('inf'), pitch_range=(0,0))
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_file_path, DEVICE)

    scores = evaluate_mirex(model, val_loader, DEVICE)
    print_mirex_report(scores)

if __name__ == "__main__":
    main()