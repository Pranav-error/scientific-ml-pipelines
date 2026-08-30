"""Probe which operators survive the PyTorch -> ONNX -> SOFIE route.

`probe_pytorch_parser.py` measures SOFIE's *PyTorch* parser, which maps only 6 ONNX
operators. That parser is being removed (root-project/root#22734), which makes the
question it answers a historical one and makes this the live question instead: once
PyTorch users are pushed onto the ONNX path, does that path actually carry the operators
SOFIE's backend implements?

Two failure modes are separated, because they are different bugs with different owners:

  PARSE  the ONNX graph cannot be turned into an RModel at all
  CODEGEN the graph parses but emitting C++ fails

The same operator cases as the PyTorch probe are reused, so the two routes are directly
comparable and any difference is attributable to the route rather than to the models.

    conda run -n rootsofie python probe_onnx_parser.py
"""
import os, sys, traceback, warnings
warnings.filterwarnings("ignore")

import torch

from probe_pytorch_parser import CASES, require_sofie

OUT = os.path.dirname(os.path.abspath(__file__))


def describe(e):
    """Last line of an exception message, or its type when the message is empty.

    SOFIE's cppyy bindings sometimes raise with no message at all, and an empty
    string has no last line -- indexing one aborted the probe mid-survey and threw
    away every result collected up to that point.
    """
    lines = str(e).strip().splitlines()
    return lines[-1][:110] if lines else type(e).__name__


def cleanup(name):
    for ext in (".onnx", ".hxx", ".dat"):
        f = os.path.join(OUT, f"m_{name}{ext}")
        if os.path.exists(f):
            os.remove(f)


def export(cls, shape, path, opset):
    m = cls().eval()
    torch.onnx.export(m, (torch.randn(*shape),), path,
                      input_names=["input"], output_names=["output"],
                      opset_version=opset, dynamo=False)


def main():
    import ROOT
    SOFIE = require_sofie(ROOT, "RModelParser_ONNX")
    opset = int(os.environ.get("OPSET", "13"))
    print(f"ROOT {ROOT.gROOT.GetVersion()}  torch {torch.__version__}  opset {opset}\n")

    results = []
    for label, cls, shape in CASES:
        name = cls.__name__
        path = os.path.join(OUT, f"m_{name}.onnx")
        try:
            export(cls, shape, path, opset)
        except Exception as e:
            results.append((label, "EXPORT", describe(e)))
            print(f"  EXPORT-FAIL  {label}")
            cleanup(name)
            continue
        try:
            parser = SOFIE.RModelParser_ONNX()
            model = parser.Parse(path)
        except Exception as e:
            results.append((label, "PARSE", describe(e)))
            print(f"  PARSE-FAIL   {label}")
            cleanup(name)
            continue
        try:
            model.Generate()
            results.append((label, "OK", ""))
            print(f"  PASS         {label}")
        except Exception as e:
            results.append((label, "CODEGEN", describe(e)))
            print(f"  CODEGEN-FAIL {label}")
        finally:
            cleanup(name)

    ok = [r for r in results if r[1] == "OK"]
    print(f"\n{len(ok)} of {len(CASES)} operators survive PyTorch -> ONNX -> SOFIE")
    bad = [r for r in results if r[1] != "OK"]
    if bad:
        print("\nFailures, by stage:")
        for label, stage, msg in bad:
            print(f"  {stage:<7} {label}")
            if msg:
                print(f"          {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
