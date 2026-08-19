"""Sanity-check the suspiciously perfect ResNet result.

99.96% accuracy / AUC 1.0 is either a genuinely easy dataset or a bug. This checks:
  1. confusion matrix + exact error count
  2. full-precision AUC (not rounded to 4dp)
  3. prediction confidence distribution
  4. nearest-neighbour distance from val samples to train samples, to catch
     near-duplicates that md5 hashing cannot see (simulated data re-rendered
     with a different noise seed would hash differently but be effectively the
     same image)
"""
import numpy as np, torch, torchvision, torch.nn as nn, torch.utils.data as tdata, glob, os
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))


def npy_loader(path):
    return torch.from_numpy(np.load(path)).float()


def build_model():
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 3)
    return m


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = build_model().to(dev)
    m.load_state_dict(torch.load(os.path.join(ROOT, "resnet18_best.pt"), map_location=dev))
    m.eval()

    vd = torchvision.datasets.DatasetFolder(root=os.path.join(ROOT, "dataset/val"),
                                            loader=npy_loader, extensions=".npy")
    vl = tdata.DataLoader(vd, batch_size=128, shuffle=False, num_workers=0)
    S, Y = [], []
    with torch.no_grad():
        for xb, yb in vl:
            xb = torch.nn.functional.interpolate(xb.to(dev), size=128, mode="bilinear",
                                                 align_corners=False)
            S.append(torch.softmax(m(xb), 1).cpu().numpy())
            Y.append(yb.numpy())
    S = np.concatenate(S); Y = np.concatenate(Y); P = S.argmax(1)

    print("classes:", vd.class_to_idx)
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(Y, P))
    print("errors:", int((P != Y).sum()), "of", len(Y))
    print("macro AUC (full precision):", repr(roc_auc_score(Y, S, multi_class="ovr", average="macro")))
    print("min prob assigned to the true class:", float(S[np.arange(len(Y)), Y].min()))
    print("mean top-1 confidence:", float(S.max(1).mean()))

    # ---- near-duplicate check ----
    print("\nnearest-neighbour check (200 val samples vs 3000 train samples)")
    rng = np.random.default_rng(0)
    tr_files = np.array(sorted(glob.glob(os.path.join(ROOT, "dataset/train/*/*.npy"))))
    va_files = np.array(sorted(glob.glob(os.path.join(ROOT, "dataset/val/*/*.npy"))))
    tr_s = rng.choice(tr_files, 3000, replace=False)
    va_s = rng.choice(va_files, 200, replace=False)
    TR = np.stack([np.load(f).ravel() for f in tr_s]).astype(np.float32)
    VA = np.stack([np.load(f).ravel() for f in va_s]).astype(np.float32)
    # cosine distance
    TRn = TR / (np.linalg.norm(TR, axis=1, keepdims=True) + 1e-9)
    VAn = VA / (np.linalg.norm(VA, axis=1, keepdims=True) + 1e-9)
    sim = VAn @ TRn.T
    best = sim.max(1)
    print(f"  cosine similarity to nearest train sample: max={best.max():.4f} "
          f"mean={best.mean():.4f} min={best.min():.4f}")
    print(f"  val samples with a >0.99 similar train sample: {(best > 0.99).sum()}/200")

    # ---- class separability without any learning ----
    print("\nsanity: mean pixel intensity by class (val)")
    for cname, ci in vd.class_to_idx.items():
        fs = [f for f in va_files if f"/{cname}/" in f][:400]
        arr = np.stack([np.load(f) for f in fs])
        print(f"  {cname:6} mean={arr.mean():.5f} std={arr.std():.5f} "
              f"nonzero_frac={(arr > 0.01).mean():.4f}")


if __name__ == "__main__":
    main()
