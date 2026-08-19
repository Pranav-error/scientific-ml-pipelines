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

### `sofie/` — operator coverage probe for ROOT's SOFIE PyTorch parser
`probe_pytorch_parser.py` builds one minimal PyTorch model per operator and reports which the
parser can convert. It exists because ROOT's SOFIE inference engine implements 56 ONNX
operators in C++, while its PyTorch parser maps only six of them (Gemm, Conv, Relu, Selu,
Sigmoid, Transpose) — so pooling, batch normalisation and residual connections are
unreachable from PyTorch even though the engine can already execute them. The probe measures
that gap rather than asserting it.

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
- **`np.isscalar()` returns False for numpy scalars.** Using it to filter PyRadiomics output
  silently drops most features while the run still reports success.

## Licence

MIT. Datasets are not included and remain under their own terms.
