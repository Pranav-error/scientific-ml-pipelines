"""Beat the Tesseract page-level baseline on RenAIssance (CER 0.129 folded / 0.156 raw).

Two changes, tested separately so the credit is attributable:

1. **Line segmentation.** Both Tesseract and TrOCR do better on single text lines than on a
   full page. Two segmenters are implemented and selectable with `--seg`, because the
   obvious one turned out to be the weaker one:

   - `projection` — ink per row, Otsu-binarised, smoothed, split at the valleys. Classic,
     and the natural first choice for well-separated horizontal print.
   - `boxes` — the textline boxes from Tesseract's own layout pass (TSV level 4).

   Measured on `nobleza`, projection scores CER 0.354 and aligns only 20/25 pages against
   the page baseline's 0.129; boxes scores 0.201 and aligns 24/25. Projection crops the
   full page width, so running heads, folio numbers and the dark scan edge ride along with
   every line, and it cannot split lines whose ascenders and descenders interleave.

   Note what this means: line segmentation alone does **not** beat the page baseline for
   Tesseract, and is not supposed to — per-line `--psm 7` throws away the page-level
   language context that `--psm 6` exploits. Segmentation exists to make TrOCR possible.

2. **TrOCR.** `microsoft/trocr-base-printed` is a transformer OCR model. It is a *single-line*
   model, which is why segmentation has to come first; feeding it a whole page produces one
   short garbled line.

Every engine is scored through the same alignment and normalisation as the page baseline, so
the numbers are directly comparable.

    python lines_and_trocr.py --engine tesseract-page
    python lines_and_trocr.py --engine tesseract-line --seg boxes
    python lines_and_trocr.py --engine trocr --seg boxes
"""
import os, re, argparse, subprocess
import numpy as np
from PIL import Image

from baseline_ocr import (render, reference_pages, ocr_page, norm, token_sim,
                          split_spread, BOOKS, WORK)

ROOT = os.path.dirname(os.path.abspath(__file__))


def find_lines(img_path, min_h=14, pad=4, thresh_frac=0.06):
    """Return line-image crops via horizontal projection of ink.

    Weaker than `tesseract_line_boxes` on this material — see the module docstring.
    Kept because it is the honest first attempt and the comparison is the finding.
    """
    im = Image.open(img_path).convert("L")
    a = np.array(im, dtype=np.float32)
    # Binarise before projecting. Thresholding the raw greyscale merges adjacent lines,
    # because the paper between them is not bright enough relative to the darkest ink.
    # Otsu separates ink from page first, then the row profile has real valleys.
    hist, _ = np.histogram(a, bins=256, range=(0, 256))
    tot = a.size
    best_t, best_var = 128, -1.0
    w0 = c0 = 0.0
    csum = np.cumsum(hist)
    cmean = np.cumsum(hist * np.arange(256))
    gmean = cmean[-1] / tot
    for t in range(1, 255):
        w0 = csum[t] / tot
        if w0 <= 0 or w0 >= 1:
            continue
        m0 = cmean[t] / csum[t]
        m1 = (cmean[-1] - cmean[t]) / (tot - csum[t])
        var = w0 * (1 - w0) * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    ink = (a < best_t).sum(axis=1).astype(np.float32)   # dark pixels per row
    if ink.max() <= 0:
        return []
    k = 5
    ink = np.convolve(ink, np.ones(k) / k, mode="same")
    on = ink > (ink.max() * thresh_frac)

    lines, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_h:
                lines.append((start, i))
            start = None
    if start is not None and len(on) - start >= min_h:
        lines.append((start, len(on)))

    crops = []
    for n, (y0, y1) in enumerate(lines):
        y0 = max(0, y0 - pad)
        y1 = min(a.shape[0], y1 + pad)
        dest = img_path.replace(".png", f"_l{n:03d}.png")
        if not os.path.exists(dest):
            im.crop((0, y0, im.size[0], y1)).save(dest)
        crops.append(dest)
    return crops


def tesseract_line_boxes(img_path, lang="spa", psm=3, pad=4, min_h=10, min_w=40):
    """Return line-image crops from Tesseract's own layout analysis (TSV level 4).

    The horizontal projection in `find_lines` fails on this material for two reasons that
    only show up on real pages: it crops the full page width, so running heads, folio
    numbers and the dark scan edge ride along with every line, and it cannot separate two
    lines whose ascenders and descenders overlap. Tesseract already runs a page layout
    pass before it recognises anything; asking it for the line boxes reuses that work and
    gives a tight box per line instead of a full-width band.
    """
    r = subprocess.run(["tesseract", img_path, "stdout", "-l", lang,
                        "--psm", str(psm), "tsv"], capture_output=True, text=True)
    im = Image.open(img_path).convert("L")
    W, H = im.size
    boxes = []
    for row in r.stdout.splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12 or f[0] != "4":          # level 4 == textline
            continue
        x, y, w, h = (int(f[6]), int(f[7]), int(f[8]), int(f[9]))
        if h < min_h or w < min_w:
            continue
        boxes.append((y, x, w, h))
    boxes.sort()                                 # reading order: top, then left

    crops = []
    for n, (y, x, w, h) in enumerate(boxes):
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        dest = img_path.replace(".png", f"_b{n:03d}.png")
        if not os.path.exists(dest):
            im.crop((x0, y0, x1, y1)).save(dest)
        crops.append(dest)
    return crops


SEGMENTERS = {"projection": find_lines, "boxes": tesseract_line_boxes}
SEG = "projection"


def segment(page_img):
    return SEGMENTERS[SEG](page_img)


def tesseract_lines(page_img, lang="spa"):
    out = []
    for ln in segment(page_img):
        r = subprocess.run(["tesseract", ln, "stdout", "-l", lang, "--psm", "7"],
                           capture_output=True, text=True)
        out.append(r.stdout.strip())
    return "\n".join(out)


_TR = {}


def trocr_lines(page_img, model="microsoft/trocr-base-printed"):
    import torch
    from transformers import (TrOCRProcessor, VisionEncoderDecoderModel,
                              AutoImageProcessor, AutoTokenizer)
    if not _TR:
        try:
            _TR["p"] = TrOCRProcessor.from_pretrained(model)
        except ValueError:
            # The TrOCR repos ship vocab.json + merges.txt but no tokenizer.json, and
            # transformers 5.x dropped the slow RobertaTokenizer it used to convert from,
            # so the bundled processor will not load. TrOCR's decoder is roberta-base and
            # uses its vocabulary unchanged (50265 tokens), so borrowing that tokenizer is
            # exact, not an approximation.
            _TR["p"] = TrOCRProcessor(
                image_processor=AutoImageProcessor.from_pretrained(model),
                tokenizer=AutoTokenizer.from_pretrained("roberta-base"))
        _TR["m"] = VisionEncoderDecoderModel.from_pretrained(model)
        _TR["d"] = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _TR["m"].to(_TR["d"]).eval()
    proc, mdl, dev = _TR["p"], _TR["m"], _TR["d"]
    out = []
    crops = segment(page_img)
    for i in range(0, len(crops), 8):                 # batch for speed
        batch = [Image.open(c).convert("RGB") for c in crops[i:i + 8]]
        px = proc(images=batch, return_tensors="pt").pixel_values.to(dev)
        with torch.no_grad():
            ids = mdl.generate(px, max_new_tokens=64)
        out += proc.batch_decode(ids, skip_special_tokens=True)
    return "\n".join(out)


ENGINES = {
    "tesseract-page": lambda p: ocr_page(p, "spa", 6),
    "tesseract-line": tesseract_lines,
    "trocr": trocr_lines,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="tesseract-line", choices=list(ENGINES))
    ap.add_argument("--min-sim", type=float, default=0.25)
    ap.add_argument("--books", default="nobleza")
    ap.add_argument("--seg", default="projection", choices=list(SEGMENTERS),
                    help="line segmentation for the line-level engines")
    a = ap.parse_args()
    global SEG
    SEG = a.seg
    import jiwer
    from scipy.optimize import linear_sum_assignment

    engine = ENGINES[a.engine]
    for book in a.books.split(","):
        pdf, gt = BOOKS[book]
        halves = [h for im in render(pdf, 300) for h in split_spread(im)]
        refs = reference_pages(gt)
        print(f"\n=== {book} | engine={a.engine} | seg={a.seg} | {len(halves)} pages ===")

        ocr = {i + 1: engine(p) for i, p in enumerate(halves)}
        pk, rl = sorted(ocr), sorted(refs)
        sim = np.array([[token_sim(norm(ocr[p]), norm(refs[r])) for r in rl] for p in pk])
        ri, ci = linear_sum_assignment(-sim)
        pairs = [(pk[i], rl[j], sim[i][j]) for i, j in zip(ri, ci) if sim[i][j] >= a.min_sim]
        print(f"aligned {len(pairs)}/{len(refs)} reference pages")
        if not pairs:
            print("  no confident alignment"); continue

        for label, kw in [("raw", {}), ("u/v + f/s folded",
                                        {"fold_uv": True, "fold_fs": True})]:
            c, w = [], []
            for pi, rk, _ in pairs:
                p, r = norm(ocr[pi], **kw), norm(refs[rk], **kw)
                if r:
                    c.append(jiwer.cer(r, p)); w.append(jiwer.wer(r, p))
            print(f"  {label:<18} CER {np.mean(c):.4f}   WER {np.mean(w):.4f}")

        pi, rk, s = max(pairs, key=lambda t: t[2])
        print(f"  best page {pi}->{rk} (sim {s:.2f})")
        print(f"    OCR: {norm(ocr[pi])[:95]}")
        print(f"    REF: {norm(refs[rk])[:95]}")


if __name__ == "__main__":
    main()
