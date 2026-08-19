"""SYMBA Task 2 — vanilla encoder-decoder Transformer, amplitude -> squared amplitude.

Scaffolding built against the 54-pair public dev sample. THE NUMBERS THIS PRODUCES ARE
NOT MEANINGFUL: 43 training examples cannot teach a transformer symbolic algebra. The
purpose is that the whole path — tokenise, batch, train, greedy-decode, score by exact
sequence match — is known-good before the real dataset arrives with the evaluation test.

Metric is exact sequence match, not token accuracy: a squared amplitude that is 95%
right is wrong. Token-level accuracy flatters these models badly.

Usage:
    SYMBA_DATA_DIR=/path/to/real/data python train_transformer.py --epochs 200
"""
import os, math, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocess import build, PAD, BOS, EOS

ROOT = os.path.dirname(os.path.abspath(__file__))


class PairDS(Dataset):
    def __init__(self, pairs, sv, tv):
        self.data = [(torch.tensor(sv.encode(s)), torch.tensor(tv.encode(t)))
                     for s, t in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch, pad_s, pad_t):
    xs, ys = zip(*batch)
    mx = max(len(x) for x in xs)
    my = max(len(y) for y in ys)
    X = torch.full((len(xs), mx), pad_s, dtype=torch.long)
    Y = torch.full((len(ys), my), pad_t, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :len(x)] = x
        Y[i, :len(y)] = y
    return X, Y


class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Seq2Seq(nn.Module):
    def __init__(self, n_src, n_tgt, d=256, heads=4, layers=3, ff=512, drop=0.1):
        super().__init__()
        self.d = d
        self.se = nn.Embedding(n_src, d)
        self.te = nn.Embedding(n_tgt, d)
        self.pe = PosEnc(d)
        self.tr = nn.Transformer(d_model=d, nhead=heads, num_encoder_layers=layers,
                                 num_decoder_layers=layers, dim_feedforward=ff,
                                 dropout=drop, batch_first=True)
        self.out = nn.Linear(d, n_tgt)
        # MPS has no `aten::_nested_tensor_from_mask_left_aligned`, which the encoder's
        # nested-tensor fast path calls whenever a src padding mask is supplied. Disable
        # the fast path rather than setting PYTORCH_ENABLE_MPS_FALLBACK=1, which would
        # silently move the op to CPU. Same maths either way.
        self.tr.encoder.use_nested_tensor = False   # attribute is use_nested_tensor, not enable_

    def forward(self, src, tgt_in, src_pad_mask, tgt_pad_mask):
        s = self.pe(self.se(src) * math.sqrt(self.d))
        t = self.pe(self.te(tgt_in) * math.sqrt(self.d))
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(src.device)
        h = self.tr(s, t, tgt_mask=causal,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_pad_mask)
        return self.out(h)


@torch.no_grad()
def greedy(model, src, sv, tv, device, max_len=64):
    model.eval()
    src = src.to(device)
    pad_s = sv.stoi[PAD]
    ys = torch.full((src.size(0), 1), tv.stoi[BOS], dtype=torch.long, device=device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits = model(src, ys, src == pad_s, torch.zeros_like(ys, dtype=torch.bool))
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        done |= nxt.squeeze(1) == tv.stoi[EOS]
        if done.all():
            break
    return ys


def exact_match(model, loader, sv, tv, device):
    hit = tot = 0
    for X, Y in loader:
        pred = greedy(model, X, sv, tv, device)
        for p, y in zip(pred, Y):
            if tv.decode(p.tolist()) == tv.decode(y.tolist()):
                hit += 1
            tot += 1
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--by-target", action="store_true",
                    help="target-disjoint split: no target seen in training (honest number)")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    (tr, va, te), sv, tv = build(by_target=args.by_target)
    print(f"split={'target-disjoint' if args.by_target else 'random'}")
    print(f"device={device}  train={len(tr)} val={len(va)} test={len(te)}  "
          f"src_vocab={len(sv)} tgt_vocab={len(tv)}")
    if len(tr) < 500:
        print("!! WARNING: this is the dev sample. Results below are scaffolding checks,\n"
              "!! not performance. Point SYMBA_DATA_DIR at the real dataset to train.")

    pad_s, pad_t = sv.stoi[PAD], tv.stoi[PAD]
    coll = lambda b: collate(b, pad_s, pad_t)
    dl_tr = DataLoader(PairDS(tr, sv, tv), batch_size=args.batch, shuffle=True, collate_fn=coll)
    dl_te = DataLoader(PairDS(te, sv, tv), batch_size=args.batch, collate_fn=coll)
    dl_va = DataLoader(PairDS(va, sv, tv), batch_size=args.batch, collate_fn=coll)

    model = Seq2Seq(len(sv), len(tv)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=pad_t)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for X, Y in dl_tr:
            X, Y = X.to(device), Y.to(device)
            tin, tout = Y[:, :-1], Y[:, 1:]
            logits = model(X, tin, X == pad_s, tin == pad_t)
            loss = crit(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if ep % 25 == 0 or ep == args.epochs:
            va_em = exact_match(model, dl_va, sv, tv, device)
            print(f"  epoch {ep:4d}  loss={tot/len(dl_tr):.4f}  val_exact_match={va_em:.3f}")

    te_em = exact_match(model, dl_te, sv, tv, device)
    print(f"\ntest exact-match: {te_em:.3f} ({len(te)} examples)  [{time.time()-t0:.0f}s]")

    X, Y = next(iter(dl_te))
    pred = greedy(model, X, sv, tv, device)
    print("\nsample predictions:")
    for i in range(min(3, len(X))):
        print(f"  target: {' '.join(tv.decode(Y[i].tolist()))}")
        print(f"  pred:   {' '.join(tv.decode(pred[i].tolist()))}")
    torch.save(model.state_dict(), os.path.join(ROOT, "symba_transformer.pt"))


if __name__ == "__main__":
    main()
