"""Control for the near-duplicate test.

Raw-pixel cosine similarity is high between ANY two images in this dataset (all are
simulated lenses with a bright central arc), so "val is 0.99 similar to train" proves
nothing on its own. The discriminating question is whether val->train nearest-neighbour
similarity is HIGHER than train->train nearest-neighbour similarity. If the two
distributions match, there is no val-specific leakage.
"""
import numpy as np, glob, os

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(files):
    X = np.stack([np.load(f).ravel() for f in files]).astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def main():
    rng = np.random.default_rng(0)
    tr_files = np.array(sorted(glob.glob(os.path.join(ROOT, "dataset/train/*/*.npy"))))
    va_files = np.array(sorted(glob.glob(os.path.join(ROOT, "dataset/val/*/*.npy"))))

    ref = load(rng.choice(tr_files, 3000, replace=False))          # reference bank
    val_q = load(rng.choice(va_files, 300, replace=False))         # val queries
    tr_q_files = rng.choice(tr_files, 300, replace=False)
    tr_q = load(tr_q_files)

    v_best = (val_q @ ref.T).max(1)

    # train queries: mask out self-matches (a query may be in the bank)
    sim_t = tr_q @ ref.T
    sim_t[sim_t > 0.999999] = -1          # drop identical self-match
    t_best = sim_t.max(1)

    print("nearest-neighbour cosine similarity to a 3000-sample TRAIN bank")
    print(f"  val   queries: mean={v_best.mean():.5f}  p99={np.percentile(v_best,99):.5f}  max={v_best.max():.5f}")
    print(f"  train queries: mean={t_best.mean():.5f}  p99={np.percentile(t_best,99):.5f}  max={t_best.max():.5f}")
    print(f"  difference in means: {v_best.mean()-t_best.mean():+.6f}")
    print()
    print("  If val and train queries score the same, there is no val-specific leakage —")
    print("  the high absolute numbers are just global homogeneity of simulated lens images.")

    # how separable are the classes without any learning at all?
    print("\nbaseline check: 1-NN classifier on raw pixels (300 val queries)")
    tr_sample = rng.choice(tr_files, 3000, replace=False)
    bank = load(tr_sample)
    bank_lab = np.array([f.split("/")[-2] for f in tr_sample])
    q_files = rng.choice(va_files, 300, replace=False)
    q = load(q_files)
    q_lab = np.array([f.split("/")[-2] for f in q_files])
    pred = bank_lab[(q @ bank.T).argmax(1)]
    print(f"  raw-pixel 1-NN accuracy: {(pred==q_lab).mean():.4f}  (chance = 0.3333)")
    print("  A low number here means the classes are NOT trivially separable,")
    print("  so the ResNet's score comes from learned features, not a giveaway artefact.")


if __name__ == "__main__":
    main()
