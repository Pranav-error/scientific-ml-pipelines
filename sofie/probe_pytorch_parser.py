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
import os, sys, subprocess, traceback, warnings
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
    """BatchNorm with NO preceding conv, deliberately.

    `Conv2d -> BatchNorm2d` in eval mode does not test BatchNormalization at all: the
    export folds the normalisation into the convolution's weights, so the ONNX graph is
    just ['Conv'] and the case silently measures Conv instead. Standing alone there is
    nothing to fold into, and the BatchNormalization node survives.
    """
    def __init__(s):
        super().__init__(); s.b = nn.BatchNorm2d(1)
    def forward(s, x): return s.b(x)


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


def require_sofie_cli():
    """Fail loudly when the ROOT on PATH has no SOFIE, before anything else runs."""
    feats = subprocess.run(["root-config", "--features"], capture_output=True,
                           text=True).stdout.split()
    if "tmva-sofie" not in feats:
        raise SystemExit(
            "This ROOT build cannot run the probe: tmva-sofie is not enabled.\n"
            f"  features: {' '.join(f for f in feats if f.startswith('tmva'))}\n"
            "Rebuild ROOT with -Dtmva-sofie=ON -Dtmva-pymva=ON to measure anything.")


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


MACRO = r"""
// Parse each traced model with SOFIE's PyTorch parser and print one PASS/FAIL line.
//
// This runs as a ROOT macro rather than through PyROOT on purpose. The parser embeds a
// Python interpreter and runs `import torch` inside it; from a python process that import
// fails ("Failed to run python code: import torch") while the identical model parses fine
// from a .C macro. Driving it here measures the parser instead of the embedding.
#include "TMVA/RModelParser_PyTorch.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

void probe(const char *manifest) {
   std::ifstream in(manifest);
   std::string line;
   while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::istringstream ss(line);
      std::string path, label;
      std::getline(ss, path, '|');
      std::string dims;
      std::getline(ss, dims, '|');
      std::getline(ss, label);
      std::vector<std::vector<size_t>> shapes(1);
      std::istringstream ds(dims);
      long d;
      while (ds >> d) shapes[0].push_back((size_t)d);
      try {
         auto model = TMVA::Experimental::SOFIE::PyTorch::Parse(path, shapes);
         std::cout << "RESULT\tPASS\t" << label << "\t" << std::endl;
      } catch (const std::exception &e) {
         std::string m(e.what());
         auto nl = m.find('\n');
         if (nl != std::string::npos) m = m.substr(0, nl);
         std::cout << "RESULT\tFAIL\t" << label << "\t" << m.substr(0, 110) << std::endl;
      } catch (...) {
         std::cout << "RESULT\tFAIL\t" << label << "\tunknown C++ exception" << std::endl;
      }
   }
}
"""


def main():
    import tempfile
    import shutil

    require_sofie_cli()
    print(f"torch {torch.__version__}\n")

    tmp = tempfile.mkdtemp(prefix="sofieprobe_")
    try:
        manifest = os.path.join(tmp, "manifest.txt")
        with open(manifest, "w") as fh:
            for label, cls, shape in CASES:
                path = os.path.join(tmp, f"m_{cls.__name__}.pt")
                m = cls().eval()
                torch.jit.save(torch.jit.trace(m, torch.randn(*shape)), path)
                fh.write(f"{path}|{' '.join(str(d) for d in shape)}|{label}\n")

        macro = os.path.join(tmp, "probe.C")
        with open(macro, "w") as fh:
            fh.write(MACRO)

        proc = subprocess.run(["root", "-l", "-b", "-q",
                               f'{macro}("{manifest}")'],
                              capture_output=True, text=True)
        rows = [l.split("\t") for l in proc.stdout.splitlines()
                if l.startswith("RESULT\t")]
        if not rows:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            raise SystemExit("the macro produced no results — see output above")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ok, fail = [], []
    for _, verdict, label, msg in rows:
        if verdict == "PASS":
            ok.append(label)
            print(f"  PASS  {label}")
        else:
            fail.append((label, msg))
            print(f"  FAIL  {label}\n          {msg}")

    print(f"\n{len(ok)} parsed, {len(fail)} failed, of {len(rows)} operators tested")

    if len([l for l in ok if "baseline" in l]) < 2:
        print("\nCONTROLS FAILED — this run measures the harness, not the parser.\n"
              "Gemm and Relu are documented as supported; if they do not parse, no other\n"
              "row here means anything. Do not report these numbers.")
        return
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
