"""Probe which PyTorch operators SOFIE's parser can actually handle.

Claim under test: SOFIE's PyTorch parser maps only 6 ONNX ops
(Gemm, Conv, Relu, Selu, Sigmoid, Transpose) while the SOFIE backend implements 56.
So any model using pooling, batch norm, residual adds, softmax, flatten, etc. cannot be
converted from PyTorch even though SOFIE could execute it via the ONNX path.

This builds one minimal model per operator and reports which parse and which fail,
so the gap is measured rather than asserted.

REQUIRES A ROOT BUILT WITH SOFIE. `root-config --features` must list `tmva-sofie`.
The conda-forge ROOT package does NOT enable it (verified on 6.40.02: features list
tmva, tmva-cpu, tmva-cudnn, tmva-pymva only, and TMVA::Experimental::SOFIE exposes
PyKeras but no PyTorch). On such a build this probe cannot run and reports so rather
than appearing to measure something.

    conda run -n rootsofie python probe_pytorch_parser.py
"""
import os, sys, traceback, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

OUT = os.path.dirname(os.path.abspath(__file__))


class Gemm(nn.Module):                      # supported (baseline)
    def __init__(s):
        super().__init__(); s.f = nn.Linear(8, 4)
    def forward(s, x): return s.f(x)


class Relu(nn.Module):                      # supported (baseline)
    def __init__(s):
        super().__init__(); s.f = nn.Linear(8, 4); s.a = nn.ReLU()
    def forward(s, x): return s.a(s.f(x))


class MaxPool(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.Conv2d(1, 2, 3, padding=1); s.p = nn.MaxPool2d(2)
    def forward(s, x): return s.p(s.c(x))


class AvgPool(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.Conv2d(1, 2, 3, padding=1); s.p = nn.AvgPool2d(2)
    def forward(s, x): return s.p(s.c(x))


class BatchNorm(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.Conv2d(1, 2, 3, padding=1); s.b = nn.BatchNorm2d(2)
    def forward(s, x): return s.b(s.c(x))


class Residual(nn.Module):                  # needs onnx::Add
    def __init__(s):
        super().__init__(); s.f = nn.Linear(8, 8)
    def forward(s, x): return x + s.f(x)


class Softmax(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Linear(8, 4); s.s = nn.Softmax(dim=1)
    def forward(s, x): return s.s(s.f(x))


class Tanh(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Linear(8, 4); s.a = nn.Tanh()
    def forward(s, x): return s.a(s.f(x))


class LeakyRelu(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Linear(8, 4); s.a = nn.LeakyReLU(0.1)
    def forward(s, x): return s.a(s.f(x))


class Flatten(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.Conv2d(1, 2, 3, padding=1); s.fl = nn.Flatten()
    def forward(s, x): return s.fl(s.c(x))


class Concat(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(8, 4); s.b = nn.Linear(8, 4)
    def forward(s, x): return torch.cat([s.a(x), s.b(x)], dim=1)


CASES = [
    ("Gemm        (baseline, expected OK)", Gemm,       [1, 8]),
    ("Relu        (baseline, expected OK)", Relu,       [1, 8]),
    ("MaxPool     -> ROperator_Pool",       MaxPool,    [1, 1, 8, 8]),
    ("AveragePool -> ROperator_Pool",       AvgPool,    [1, 1, 8, 8]),
    ("BatchNorm   -> ROperator_BatchNormalization", BatchNorm, [1, 1, 8, 8]),
    ("Add/residual-> ROperator_BasicBinary", Residual,  [1, 8]),
    ("Softmax     -> ROperator_Softmax",    Softmax,    [1, 8]),
    ("Tanh        -> ROperator_Tanh",       Tanh,       [1, 8]),
    ("LeakyRelu   -> ROperator_LeakyRelu",  LeakyRelu,  [1, 8]),
    ("Flatten     -> ROperator_Reshape",    Flatten,    [1, 1, 8, 8]),
    ("Concat      -> ROperator_Concat",     Concat,     [1, 8]),
]


def require_sofie(ROOT, attr):
    """Fail loudly and specifically when the ROOT build has no SOFIE parser."""
    import subprocess
    feats = subprocess.run(["root-config", "--features"], capture_output=True,
                           text=True).stdout.split()
    S = getattr(ROOT.TMVA.Experimental, "SOFIE", None)
    if S is None or not hasattr(S, attr):
        raise SystemExit(
            f"This ROOT build cannot run the probe: SOFIE.{attr} is unavailable.\n"
            f"  tmva-sofie in root-config --features: {'tmva-sofie' in feats}\n"
            f"  features: {' '.join(f for f in feats if f.startswith('tmva'))}\n"
            "Rebuild ROOT with -Dtmva-sofie=ON (needs protobuf) to measure anything.")
    return S


def main():
    import ROOT
    Parse = require_sofie(ROOT, "PyTorch").PyTorch.Parse
    print(f"ROOT {ROOT.gROOT.GetVersion()}  torch {torch.__version__}\n")

    ok, fail = [], []
    for label, cls, shape in CASES:
        path = os.path.join(OUT, f"m_{cls.__name__}.pt")
        m = cls().eval()
        torch.jit.save(torch.jit.trace(m, torch.randn(*shape)), path)
        try:
            Parse(path, [shape])
            ok.append(label)
            print(f"  PASS  {label}")
        except Exception as e:
            msg = str(e).strip().splitlines()
            msg = msg[-1][:110] if msg else repr(e)
            fail.append((label, msg))
            print(f"  FAIL  {label}\n          {msg}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    print(f"\n{len(ok)} parsed, {len(fail)} failed, of {len(CASES)} operators tested")
    if fail:
        print("\nOperators SOFIE implements for ONNX but cannot reach from PyTorch:")
        for label, _ in fail:
            print(f"  - {label.split('->')[-1].strip() or label}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
