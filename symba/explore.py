"""SYMBA Task 1.2 groundwork: characterise the QED amplitude -> squared-amplitude data.

Before writing a tokeniser or a model, establish what is actually in the data:
how many DISTINCT pairs exist (the raw files repeat lines heavily), how long the
sequences are, and how large the vocabulary is. Those three numbers decide whether
a transformer is even the right tool here.
"""
import re, os, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")

amps = [l.strip() for l in open(os.path.join(D, "amp_1to2.txt")) if l.strip()]
sqs = [l.strip() for l in open(os.path.join(D, "sqamp_1to2.txt")) if l.strip()]
assert len(amps) == len(sqs), (len(amps), len(sqs))
print(f"paired lines: {len(amps)}")

pairs = list(zip(amps, sqs))
uniq_pairs = set(pairs)
print(f"unique (amp, sq) pairs: {len(uniq_pairs)}")
print(f"unique amplitudes:      {len(set(amps))}")
print(f"unique squared amps:    {len(set(sqs))}")

# is the mapping a function? (same amplitude -> different targets would be a problem)
m = collections.defaultdict(set)
for a, s in pairs:
    m[a].add(s)
ambiguous = {a: v for a, v in m.items() if len(v) > 1}
print(f"amplitudes mapping to >1 distinct target: {len(ambiguous)}")

# ---- tokenisation ----
def tok_amp(s):
    return s.split(",")

# keep physics symbols whole: m_mu, s_12, e, **, numbers, operators, parens
SQ_RE = re.compile(r"[A-Za-z]+_[A-Za-z0-9]+|\*\*|[A-Za-z]+|\d+|[()+\-*/]")

def tok_sq(s):
    return SQ_RE.findall(s)

al = [len(tok_amp(a)) for a in amps]
sl = [len(tok_sq(s)) for s in sqs]
def stats(x):
    x = sorted(x)
    return f"min={x[0]} med={x[len(x)//2]} max={x[-1]} mean={sum(x)/len(x):.1f}"
print(f"\namplitude token length:  {stats(al)}")
print(f"sq-amp   token length:  {stats(sl)}")

va = collections.Counter(t for a in amps for t in tok_amp(a))
vs = collections.Counter(t for s in sqs for t in tok_sq(s))
print(f"\namplitude vocab: {len(va)}  |  sq-amp vocab: {len(vs)}")
print("amp top-15:", [t for t, _ in va.most_common(15)])
print("sq  top-15:", [t for t, _ in vs.most_common(15)])

print("\nsample distinct targets:")
for s in list(dict.fromkeys(sqs))[:6]:
    print("  ", s)

# round-trip check: does the tokeniser lose information?
bad = [s for s in set(sqs) if "".join(tok_sq(s)) != s.replace(" ", "")]
print(f"\ntokeniser round-trip failures: {len(bad)} of {len(set(sqs))}")
for b in bad[:3]:
    print("   FAIL:", b, "->", "".join(tok_sq(b)))
