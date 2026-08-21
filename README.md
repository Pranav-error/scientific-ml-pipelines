# scientific-ml-pipelines

Reproducible pipelines for scientific machine-learning benchmarks, written while working
through open datasets from [ML4SCI](https://ml4sci.org/) and
[Stanford AIMI](https://aimi.stanford.edu/).

The theme across all of it: **a result you cannot verify is not a result.** Each pipeline
carries the check that would have caught it being wrong, not just the number it produced.

---

## Contents

### `lensing/` — dark-matter substructure classification
Three-class classification of simulated strong-lensing images (no substructure / CDM / axion),
30,000 train and 7,500 validation images at 64x64.

| model | accuracy | macro-AUC |
|---|---|---|
| `run_lensing_baseline.py` — 3-layer CNN, tanh, 20 epochs | 0.8708 | 0.9601 |
| `run_lensing_resnet.py` — ImageNet ResNet-18, 128px, flip/rot90 aug | 0.9996 | 0.99999995 |

The ResNet passed the baseline's 20-epoch score **after a single epoch**. Notable choices:
the 1-channel stem is seeded by summing the pretrained RGB filters rather than being
re-initialised (keeping the pretrained edge detectors), and augmentation is limited to flips
and 90-degree rotations because substructure morphology has no canonical orientation — a
domain argument, not a default.

**A 0.9996 deserves suspicion, so it gets tested.** `verify_resnet.py` and
`verify_leakage_control.py` check for train/validation leakage. The naive test — cosine
similarity between validation and training images — is *useless* here: every pair scores above
0.986, because every image is a simulated lens with a bright central arc. The discriminating
control is comparing validation-vs-train nearest-neighbour similarity against
train-vs-train: 0.99618 vs 0.99670. Validation scores *lower*, so there is no leakage. A
raw-pixel 1-NN classifier reaches 0.717 against a 0.333 chance baseline, so the signal is real
but the ResNet's gain is learned rather than an artefact.

### `predict/` — coronary calcium scoring and radiomics
Pipeline over the COCA dataset (gated cardiac CT with expert calcium annotations).

- `agatston.py` — Agatston coronary calcium scoring. The annotations are OsiriX ROI exports in
  Apple **plist** format carrying per-lesion area and peak Hounsfield units, which is
  everything the score needs — so the whole scoring task runs on ~1.5 MB of annotations
  instead of 21 GB of DICOMs.
- `radiomics_pipeline.py` — PyRadiomics feature extraction over the annotated lesions,
  rasterising each ROI polygon onto its slice.
- `explore_coca.py`, `map_coca.py`, `fetch_coca.py` — dataset exploration and selective
  download; the DICOMs are pulled per patient rather than wholesale.

**The verification that matters:** mapping an annotation onto the right CT slice rests on the
assumption that the annotation's `ImageIndex` is a 0-based index into the series sorted by
z-position. A mismatch there produces perfectly plausible features that are silently wrong. So
`radiomics_pipeline.py --verify-only` compares the pixels each mask selects against the peak
HU the annotation already recorded: **100% exact agreement across every lesion**, which turns
the assumption into a checked fact.

The scoring reproduces a published reference analysis exactly on the same patient subset
(min 1.6, max 3010.2, risk categories 14/9/7) — but only with no minimum lesion area, whereas
the classical Agatston definition requires ≥1 mm² to reject noise. That discrepancy is
surfaced rather than silently matched.

> No data is included here. COCA is distributed under a data use agreement; request access
> from Stanford AIMI directly.

### `renaissance/` — OCR on 17th-century Spanish print
Tesseract and TrOCR over the RenAIssance (HumanAI) extracts: two Padilla volumes, 32
half-pages each, scored as character error rate against expert transcriptions.

| engine | nobleza | noble |
|---|---|---|
| `tesseract-page` — off-the-shelf baseline | 0.1292 | 0.1446 |
| `tesseract-line`, projection segmentation | 0.3535 | — |
| `tesseract-line`, layout boxes + body filter | 0.1668 | 0.1430 |
| `trocr`, layout boxes + body filter | 0.1786 | 0.1499 |
| **`tesseract-page-cropped`** | **0.1136** | **0.1050** |

CER is folded for u/v and long-s: the transcription notes record that u and v are used
interchangeably and that long-s is transcribed as `s` though OCR reads it as `f`, so scores
are reported raw and folded to separate misreading from period orthography.

**The win is not a better recogniser.** Most of the avoidable error was never recognition
error at all — running heads, folio numbers and the marginal note column get read off the
page and scored against a transcription that contains none of them. Cropping to the body
block and keeping `--psm 6` beats the baseline by 12% and 27% relative. Line-level
recognition, the obvious approach, *never* beat the page baseline and was not likely to:
per-line `--psm 7` discards the page-level language context `--psm 6` exploits. It earned
its place as the diagnostic that located the real cost, not as the fix.

**Nothing here is claimed off a mean.** Page difficulty varies enough to swamp a real
effect, so `significance.py` pairs engines page by page and bootstraps the difference. That
mattered: the line pipeline scored 0.1430 against a 0.1446 baseline on `noble` and looked
like the first win, but the paired CI was [-0.0394, +0.0377] over 15/25 pages —
indistinguishable. The cropped engine clears the bar honestly, at +0.0156 CI
[+0.0077, +0.0243] and +0.0397 CI [+0.0303, +0.0494], winning 18/25 and 24/25 pages.

**Two documented dead ends.** Sweeping `--psm 3/4/6` over the cropped page moves CER by
under 0.005 and the best mode disagrees between the books — opposite signs with CIs over
zero, which is what noise looks like. And masking every non-body region instead of cropping
to one rectangle, which should in principle remove marginalia reaching inside the body box,
is indistinguishable from cropping for noticeably more machinery. Neither is adopted.

TrOCR stays short of the baseline throughout. That is domain mismatch, not segmentation —
the checkpoint is trained on modern printed English — and fine-tuning it here was judged
not worth the cost, since the transcriptions are page-level and would require inducing
line-level pairs from an engine the model is meant to beat.

### `symba/` — symbolic regression of squared amplitudes
Sequence-to-sequence transformer mapping Feynman amplitudes (prefix-notation token trees) to
squared amplitudes (SymPy expressions).

- `preprocess.py` — regex tokenisers that keep physics symbols intact (`m_e`, `s_12`), plus
  **dummy-index normalisation**: summed-over indices like `alpha_0`/`i_2` carry no meaning, so
  they are renamed to first-appearance order, canonicalising expressions that differ only in
  arbitrary labelling. Physical indices are left untouched.
- `train_transformer.py` — encoder-decoder transformer, teacher forcing, greedy decoding.

**Two deliberate choices.** The metric is exact sequence match, not token accuracy — a squared
amplitude that is 95% right is wrong, and token accuracy flatters these models badly. And
because several distinct amplitudes reduce to the *same* squared amplitude, a random split
leaks: on the public sample, 33-60% of held-out targets also appear in training. `split()`
therefore offers a target-disjoint mode, so the honest generalisation number is available
alongside the conventional one.

### `sofie/` — operator coverage probes for ROOT's SOFIE
Two probes that build one minimal model per operator and report which SOFIE can convert:
`probe_pytorch_parser.py` for the direct PyTorch parser, `probe_onnx_parser.py` for the
PyTorch -> ONNX -> SOFIE route, separating export, parse and C++ codegen failures. Measured
on a ROOT 6.40.02 built from source with `-Dtmva-sofie=ON`, torch 2.13, opset 13.

| route | result |
|---|---|
| PyTorch parser | **2 / 11** — Gemm and Relu only |
| via ONNX | **11 / 11** — parse and codegen |

The PyTorch column matches ROOT's source exactly: that parser maps six operators — Gemm,
Conv, Relu, Selu, Sigmoid, Transpose — and only Gemm and Relu stand alone in these cases.
Each failure reports `Parsing PyTorch node onnx::X is not yet supported`. The gap is real,
and reporting it upstream as
[root-project/root#23108](https://github.com/root-project/root/issues/23108) got the
definitive answer: the PyTorch parser is orphaned and
[#22734](https://github.com/root-project/root/pull/22734) removes it, because PyTorch's own
ONNX export is now the preferred path. On `master` it is already gone. The ONNX column says
that removal costs nothing — every operator the outgoing parser could not reach survives
the route replacing it.

**Getting an honest 11/11 took three corrections, and the wrong answers were the
believable ones.** The BatchNorm case started as `Conv2d -> BatchNorm2d`, which in eval
mode does not test BatchNormalization at all: the export folds the normalisation into the
convolution's weights and the graph is literally `['Conv']`. It passed, and that pass
briefly read as evidence the parser supported an operator its own source does not list.
Standing BatchNorm alone fixed that but introduced a second artifact — a freshly
initialised BatchNorm has `weight == running_var` and `bias == running_mean`, so the
exporter deduplicates them behind `Identity` nodes, and SOFIE requires those four to be
initialized tensors ([#16282](https://github.com/root-project/root/pull/16282)). Codegen
failed on a graph no trained model produces. Giving the parameters distinct values is what
finally tests the operator a real model would export.

**Both probes treat Gemm and Relu as controls and refuse to report if they fail.** That
guard fired twice — the conda-forge ROOT package does not enable SOFIE at all
(`root-config --features` lists no `tmva-sofie`), and driven through PyROOT the parser
fails on every case because its embedded interpreter cannot `import torch`, so the probe
runs as a ROOT macro. Either would otherwise have read as "0 of 11 operators supported".

Build note: ROOT 6.40 cannot configure with `-Dtmva-sofie=ON` while `tmva-pymva` is off,
which is what `-Dgminimal=ON` gives you. `SearchInstalledSoftware.cmake` then requests only
`Development.Module`, creating `Python3::Module`, while `sofie_parsers/CMakeLists.txt`
links `ROOTTMVASofiePyParsers` against `Python3::Python` unconditionally. Fixed on master.

### `tools/` — documentation link checker
`k8s_linkcheck.py` finds broken internal links in Hugo documentation sites. Written against
kubernetes/website, where it surfaced fixes now submitted upstream.

**Its main lesson is negative.** Scanning a repository alone is close to worthless: of 71
targets it flagged, **68 resolved fine** through aliases and redirects defined outside the
checkout. The live site is the only reliable oracle. It also has to know that Hugo leaf
bundles use `index.md` while branch bundles use `_index.md` — handling only the latter
produced 168 false positives on its own.

---

## Environment notes

Apple Silicon (MPS) trips over several things that work on CUDA:

- **No float64.** Loaders must cast to float32 *before* moving tensors to the device;
  `.to(device).float()` moves first and crashes.
- **`nn.Transformer`** calls `aten::_nested_tensor_from_mask_left_aligned` in its encoder
  fast path, which MPS does not implement. It only fires in eval, so training looks healthy
  and the crash lands at the first decode. Set `encoder.use_nested_tensor = False` — note the
  attribute is `use_nested_tensor`, while `enable_nested_tensor` is only the constructor
  argument.
- **PyRadiomics** does not build on Python 3.14. On 3.11:
  `pip install "numpy<2" setuptools wheel Cython versioneer tomli` then
  `pip install --no-build-isolation pyradiomics`.
- **TrOCR will not load on transformers 5.x out of the box.** The `microsoft/trocr-*` repos
  ship `vocab.json` and `merges.txt` but no `tokenizer.json`, and transformers 5 removed the
  slow `RobertaTokenizer` it used to convert from, so `TrOCRProcessor.from_pretrained` raises
  — and installing sentencepiece or tiktoken, which the error suggests, does not help. Build
  the processor from parts instead, with `AutoTokenizer.from_pretrained("roberta-base")`;
  that is exact rather than approximate, since TrOCR's decoder is roberta-base with its
  50265-token vocabulary unchanged.
- **`np.isscalar()` returns False for numpy scalars.** Using it to filter PyRadiomics output
  silently drops most features while the run still reports success.

## Licence

MIT. Datasets are not included and remain under their own terms.
