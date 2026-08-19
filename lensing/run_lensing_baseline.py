"""ML4SCI DeepLearnHackathon — Strong Lensing Task A (multi-class classification).

Faithful reproduction of StrongLensingChallenge_Classification.ipynb, adapted to run
locally on Apple Silicon (MPS) instead of Colab/CUDA. Baseline model, unchanged
architecture and hyperparameters — the point is a reference number to improve on.
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
BATCH = 100
EPOCHS = 20


def npy_loader(path):
    # .npy files are float64; MPS has no float64 support, so cast on the CPU side
    # before the tensor is ever moved to the device.
    return torch.from_numpy(np.load(path)).float()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=3, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 64)
        self.linear2 = nn.Linear(64, 3)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.avgpool(self.tanh(self.conv1(x)))
        x = self.avgpool(self.tanh(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.tanh(self.linear1(x))
        return self.linear2(x)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_data = torchvision.datasets.DatasetFolder(
        root=os.path.join(ROOT, "dataset/train"), loader=npy_loader, extensions=".npy")
    val_data = torchvision.datasets.DatasetFolder(
        root=os.path.join(ROOT, "dataset/val"), loader=npy_loader, extensions=".npy")
    print("classes:", train_data.class_to_idx)
    print(f"train: {len(train_data)}  val: {len(val_data)}")

    train_loader = tdata.DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = tdata.DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=4)

    model = CNN().to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    n_batches = len(train_loader)

    t0 = time.time()
    history = []
    pbar = tqdm(range(1, EPOCHS + 1))
    for epoch in pbar:
        model.train()
        train_loss = train_acc = 0.0
        for x_tr, y_tr in train_loader:
            xb = x_tr.to(device)
            yb = y_tr.to(device, dtype=torch.long)
            optimizer.zero_grad()
            out = model(xb)
            _, preds = torch.max(out.data, 1)
            correct = (preds == yb).float().sum()
            loss = criteria(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += correct.item() / xb.shape[0]
        train_loss /= n_batches
        train_acc /= n_batches
        history.append({"epoch": epoch, "loss": train_loss, "acc": train_acc})
        pbar.set_postfix({"loss": round(train_loss, 4), "acc": round(train_acc, 4)})
    train_time = time.time() - t0

    # ---- validation ----
    model.eval()
    y_score, y_test = [], []
    for x_ts, y_ts in val_loader:
        xb = x_ts.to(device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(xb), dim=1)
        y_score.append(probs.cpu().numpy())
        y_test.append(y_ts.numpy())
    y_score = np.concatenate(y_score, axis=0)
    y_true_labels = np.concatenate(y_test, axis=0)
    y_val = label_binarize(y_true_labels, classes=[0, 1, 2])

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr_micro, tpr_micro, _ = roc_curve(y_val.ravel(), y_score.ravel())
    micro_auc = auc(fpr_micro, tpr_micro)
    macro_auc = float(np.mean([roc_auc[i] for i in range(3)]))

    y_pred = np.argmax(y_score, axis=1)
    res = {
        "device": str(device),
        "epochs": EPOCHS,
        "train_seconds": round(train_time, 1),
        "accuracy": float(accuracy_score(y_true_labels, y_pred)),
        "precision": float(precision_score(y_true_labels, y_pred, average="weighted")),
        "recall": float(recall_score(y_true_labels, y_pred, average="weighted")),
        "f1": float(f1_score(y_true_labels, y_pred, average="weighted")),
        "auc_per_class": {k: float(v) for k, v in roc_auc.items()},
        "auc_micro": float(micro_auc),
        "auc_macro": macro_auc,
        "history": history,
    }
    print("\n===== RESULTS =====")
    for k in ("accuracy", "precision", "recall", "f1", "auc_micro", "auc_macro"):
        print(f"{k:12} {res[k]:.4f}")
    print("per-class AUC:", {k: round(v, 4) for k, v in roc_auc.items()})
    print(f"train time: {train_time:.0f}s")

    json.dump(res, open(os.path.join(ROOT, "baseline_results.json"), "w"), indent=1)
    torch.save(model.state_dict(), os.path.join(ROOT, "baseline_cnn.pt"))
    print("saved baseline_results.json + baseline_cnn.pt")


if __name__ == "__main__":
    main()
