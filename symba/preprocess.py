"""SYMBA Task 1.2 — preprocessing amplitude -> squared-amplitude data.

Built against the public dev sample (54 unique pairs). The real paired dataset ships
with the evaluation test, so this is written so that only DATA_DIR changes in March.

The one non-obvious step is dummy-index normalisation. A Feynman amplitude contains
two kinds of index:

  * physical  - s_12, p_1, m_e, m_mu: these carry meaning and MUST be preserved
  * dummy     - alpha_0, i_2, ...: summed-over indices whose particular numbering is
                arbitrary. The same physics can be written with alpha_0/alpha_1 or
                alpha_5/alpha_9, so leaving them raw makes the model memorise
                irrelevant labels and inflates the vocabulary.

Normalising dummies to first-appearance order (alpha_0, alpha_1, ... in the order
encountered) canonicalises equivalent expressions without touching the physics.
"""
import os, re, json, random, argparse
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SYMBA_DATA_DIR", os.path.join(ROOT, "data"))

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

# physics symbols that must survive tokenisation intact
SQ_RE = re.compile(r"[A-Za-z]+_[A-Za-z0-9]+|\*\*|[A-Za-z]+|\d+|[()+\-*/]")

# dummy index families: prefix -> regex over a single amplitude token
DUMMY = re.compile(r"^(alpha|beta|gamma|mu|nu|i|j|k)_(\d+)$")


def tok_amp(line):
    return [t for t in line.strip().split(",") if t]


def tok_sq(line):
    return SQ_RE.findall(line.strip())


def normalise_dummies(tokens):
    """Rename dummy indices to first-appearance order, per family. Physical indices
    (s_12, p_1, m_e ...) are untouched because they are not in the DUMMY families."""
    mapping, counters = {}, collections.Counter()
    out = []
    for t in tokens:
        m = DUMMY.match(t)
        if m:
            fam = m.group(1)
            if t not in mapping:
                mapping[t] = f"{fam}_{counters[fam]}"
                counters[fam] += 1
            out.append(mapping[t])
        else:
            out.append(t)
    return out


class Vocab:
    def __init__(self, counter, min_freq=1):
        self.itos = list(SPECIALS) + [t for t, c in sorted(counter.items()) if c >= min_freq]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, toks, add_bos_eos=True):
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks]
        return [self.stoi[BOS]] + ids + [self.stoi[EOS]] if add_bos_eos else ids

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos[i]
            if t == EOS:
                break
            if t not in (PAD, BOS):
                out.append(t)
        return out

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        json.dump(self.itos, open(path, "w"))

    @staticmethod
    def load(path):
        v = Vocab(collections.Counter())
        v.itos = json.load(open(path))
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


def load_pairs(prefix="QED", normalise=True):
    amp_f = os.path.join(DATA_DIR, "amp_1to2.txt")
    sq_f = os.path.join(DATA_DIR, "sqamp_1to2.txt")
    amps = [l.strip() for l in open(amp_f) if l.strip()]
    sqs = [l.strip() for l in open(sq_f) if l.strip()]
    assert len(amps) == len(sqs), "amplitude/squared-amplitude files disagree in length"

    seen, pairs = set(), []
    for a, s in zip(amps, sqs):
        if (a, s) in seen:          # raw files repeat every pair ~8x
            continue
        seen.add((a, s))
        at = tok_amp(a)
        if normalise:
            at = normalise_dummies(at)
        pairs.append((at, tok_sq(s)))
    return pairs


def split(pairs, seed=0, ratios=(0.8, 0.1, 0.1), by_target=False):
    """Random 80-10-10, or target-disjoint.

    Several distinct amplitudes reduce to the SAME squared amplitude (54 sources -> 36
    targets in the dev sample), so a random split leaks: 33-60% of held-out targets also
    appear in train, and exact-match is inflated accordingly. `by_target=True` groups by
    target first, so no target is seen during training. Report both — the random split is
    the standard comparison, the grouped one is the honest generalisation number.
    """
    if not by_target:
        idx = list(range(len(pairs)))
        random.Random(seed).shuffle(idx)
        n = len(pairs)
        n_tr, n_va = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        take = lambda sl: [pairs[i] for i in sl]
        return take(idx[:n_tr]), take(idx[n_tr:n_va]), take(idx[n_va:])

    groups = collections.defaultdict(list)
    for s, t in pairs:
        groups[tuple(t)].append((s, t))
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_tr, n_va = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    flat = lambda ks: [p for k in ks for p in groups[k]]
    return flat(keys[:n_tr]), flat(keys[n_tr:n_va]), flat(keys[n_va:])


def build(normalise=True, by_target=False):
    pairs = load_pairs(normalise=normalise)
    tr, va, te = split(pairs, by_target=by_target)
    src_c = collections.Counter(t for p, _ in pairs for t in p)
    tgt_c = collections.Counter(t for _, q in pairs for t in q)
    return (tr, va, te), Vocab(src_c), Vocab(tgt_c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-normalise", action="store_true")
    a = ap.parse_args()

    raw = load_pairs(normalise=False)
    norm = load_pairs(normalise=True)
    rv = collections.Counter(t for p, _ in raw for t in p)
    nv = collections.Counter(t for p, _ in norm for t in p)
    print(f"unique pairs: {len(raw)}")
    print(f"source vocab  raw={len(rv)}  normalised={len(nv)}  "
          f"(reduction {100*(1-len(nv)/len(rv)):.0f}%)")
    print(f"distinct source sequences  raw={len({tuple(p) for p,_ in raw})}  "
          f"normalised={len({tuple(p) for p,_ in norm})}")

    (tr, va, te), sv, tv = build(not a.no_normalise)
    print(f"\nsplit: train={len(tr)} val={len(va)} test={len(te)}")
    print(f"src vocab={len(sv)}  tgt vocab={len(tv)}")
    print(f"\nexample source: {' '.join(tr[0][0][:14])} ...")
    print(f"example target: {' '.join(tr[0][1])}")
    sv.save(os.path.join(ROOT, "vocab_src.json"))
    tv.save(os.path.join(ROOT, "vocab_tgt.json"))
    print("\nwrote vocab_src.json + vocab_tgt.json")
