"""ML4SCI Strong Lensing Task A — ResNet-18 transfer-learning version.

Step 2 of the practice loop: same data and same metrics as run_lensing_baseline.py,
but the approach that actually passes ML4SCI evaluation tests. Compare
resnet_results.json against baseline_results.json to see what each change bought.

Differences from the baseline:
  * ResNet-18 with ImageNet weights instead of a 3-layer tanh CNN
  * conv1 rebuilt for 1 input channel, seeding it by summing the pretrained RGB
    filters — keeps the pretrained edge detectors instead of throwing them away
  * 64x64 -> 128x128 upsample, since ResNet's stride-2 stem plus maxpool leaves
    almost no spatial extent at 64x64
  * light augmentation (flips + 90-degree rotations), which are label-preserving
    for lensing images since substructure has no canonical orientation
  * AdamW + cosine schedule, and the checkpoint is chosen by best validation
    macro-AUC rather than whatever the last epoch happened to be
"""
import os, json, time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as tdata
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH = 128
EPOCHS = 15
LR = 3e-4
IMG = 128


def npy_loader(path):
    # float64 on disk; cast before the tensor reaches MPS (no float64 support there)
    return torch.from_numpy(np.load(path)).float()


class Augment(nn.Module):
    """Label-preserving augmentation for lensing images: substructure morphology
    has no preferred orientation, so flips and 90-degree rotations are free data."""
    def forward(self, x):
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, dims=[-2, -1])
        return x


def build_model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    old = m.conv1.weight.data                      # (64, 3, 7, 7)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight.data = old.sum(dim=1, keepdim=True)   # collapse RGB -> grayscale
    m.fc = nn.Linear(m.fc.in_features, 3)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for xb, yb in loader:
        xb = torch.nn.functional.interpolate(xb.to(device), size=IMG, mode="bilinear",
                                             align_corners=False)
        scores.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        labels.append(yb.numpy())
    y_score = np.concatenate(scores)
    y_true = np.concatenate(labels)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    per_class = {i: auc(*roc_curve(y_bin[:, i], y_score[:, i])[:2]) for i in range(3)}
    return y_score, y_true, y_bin, per_class, float(np.mean(list(per_class.values())))


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_data = torchvision.datasets.DatasetFolder(
        root=os.path.join(ROOT, "dataset/train"), loader=npy_loader, extensions=".npy")
    val_data = torchvision.datasets.DatasetFolder(
        root=os.path.join(ROOT, "dataset/val"), loader=npy_loader, extensions=".npy")
    print("classes:", train_data.class_to_idx, "| train:", len(train_data), "val:", len(val_data))

    train_loader = tdata.DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = tdata.DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=4)

    model = build_model().to(device)
    aug = Augment()
    criteria = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_state, history = 0.0, None, []
    t0 = time.time()
    pbar = tqdm(range(1, EPOCHS + 1))
    for epoch in pbar:
        model.train()
        tot_loss = tot_acc = 0.0
        for xb, yb in train_loader:
            xb = aug(xb)
            xb = torch.nn.functional.interpolate(xb.to(device), size=IMG, mode="bilinear",
                                                 align_corners=False)
            yb = yb.to(device, dtype=torch.long)
            optimizer.zero_grad()
            out = model(xb)
            loss = criteria(out, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            tot_acc += (out.argmax(1) == yb).float().mean().item()
        sched.step()
        tr_loss = tot_loss / len(train_loader)
        tr_acc = tot_acc / len(train_loader)
        _, _, _, _, macro = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": tr_loss, "train_acc": tr_acc, "val_macro_auc": macro})
        if macro > best_auc:
            best_auc = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pbar.set_postfix({"loss": round(tr_loss, 4), "acc": round(tr_acc, 4),
                          "valAUC": round(macro, 4)})
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    y_score, y_true, y_bin, per_class, macro_auc = evaluate(model, val_loader, device)
    micro_auc = auc(*roc_curve(y_bin.ravel(), y_score.ravel())[:2])
    y_pred = y_score.argmax(1)

    res = {
        "model": "resnet18-imagenet, 1ch stem, 128px, flip+rot90 aug, AdamW+cosine",
        "device": str(device), "epochs": EPOCHS, "train_seconds": round(train_time, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "auc_per_class": {str(k): float(v) for k, v in per_class.items()},
        "auc_micro": float(micro_auc), "auc_macro": float(macro_auc),
        "history": history,
    }
    print("\n===== RESNET RESULTS =====")
    for k in ("accuracy", "precision", "recall", "f1", "auc_micro", "auc_macro"):
        print(f"{k:12} {res[k]:.4f}")
    print("per-class AUC:", {k: round(v, 4) for k, v in per_class.items()})
    print(f"train time: {train_time:.0f}s")

    json.dump(res, open(os.path.join(ROOT, "resnet_results.json"), "w"), indent=1)
    torch.save(best_state, os.path.join(ROOT, "resnet18_best.pt"))

    base_path = os.path.join(ROOT, "baseline_results.json")
    if os.path.exists(base_path):
        b = json.load(open(base_path))
        print("\n===== BASELINE vs RESNET =====")
        print(f"{'metric':<12}{'baseline':>10}{'resnet':>10}{'delta':>10}")
        for k in ("accuracy", "f1", "auc_macro"):
            print(f"{k:<12}{b[k]:>10.4f}{res[k]:>10.4f}{res[k]-b[k]:>+10.4f}")


if __name__ == "__main__":
    main()
