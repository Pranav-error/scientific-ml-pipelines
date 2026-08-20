"""Beat the Tesseract page-level baseline on RenAIssance 17th-century Spanish print.

Page baseline: CER 0.1292 (nobleza) / 0.1446 (noble), folded for u/v and long-s.

**Result: `tesseract-page-cropped` beats it on both books** — 0.1136 and 0.1050, paired
per-page bootstrap CI clear of zero in both cases (see `significance.py`). The win did not
come from a better recogniser. It came from noticing that most of the avoidable error was
never recognition error at all: running heads, folio numbers and the marginal note column
are read off the page and scored against a transcription that contains none of them.

The route there, kept in full because the dead ends are the argument:

1. **Line segmentation.** Selectable with `--seg`, because the obvious method is the weaker
   one. `projection` (ink per row, Otsu-binarised, split at valleys) scores 0.3535 and
   aligns 20/25 pages; `boxes` (Tesseract's own layout pass, TSV level 4) scores 0.2008 and
   aligns 24/25. Projection crops full page width, so the running heads and scan edge ride
   along with every line, and it cannot split interleaved ascenders and descenders.

2. **Body filtering** (`--filter body`). Drops non-body boxes by column overlap and line
   height. Worth 0.2008 -> 0.1668 on nobleza and 0.1913 -> 0.1430 on noble, the largest
   single lever found. Two parts, and the second mattered more: the height gate must be
   measured over the *wide* body boxes only. Taking the median over all boxes lets a page
   full of specks collapse its own threshold, which is how h=12 slivers survived on noble.

3. **TrOCR.** `microsoft/trocr-base-printed`, a single-line model, which is why segmentation
   had to come first. Best score 0.1786 (nobleza) / 0.1499 (noble) — still short of the
   page baseline. This is domain mismatch, not a segmentation failure: the checkpoint is
   trained on modern printed English. Its per-line output is often the better *reading*;
   filtering helped it more than it helped Tesseract precisely because running heads were
   eating its score. Fine-tuning on this material is the obvious next step.

   The two MISSING `encoder.pooler.*` keys reported at load are the ViT pooler, which the
   decoder never reads. They are expected and do not affect the score.

4. **Cropping instead of segmenting** (`tesseract-page-cropped`). No line variant beat the
   page baseline, and none was ever likely to: per-line `--psm 7` discards the page-level
   language context that makes `--psm 6` strong here. So keep `--psm 6` and apply the one
   idea that did work — crop the page to the bounding box of the body lines and recognise
   that. A rectangle suffices, since the marginalia sit in their own column beside the body
   and the running head above it.

5. **Two things that did not help.** Both were tried after the win and both are kept,
   because "we tried it and it did not move" is worth more than silence:

   - **Page segmentation mode.** Sweeping `--psm 3/4/6` on the cropped page moves CER by
     less than 0.005 and the best mode *disagrees between the two books* (3 on nobleza,
     4 on noble) — the signature of noise, and the paired CI spans zero both ways. `--psm 6`
     stays the default because nothing beats it, not because it won.
   - **Masking instead of cropping** (`tesseract-page-masked`). Painting every non-body
     region white, rather than cropping to one rectangle, should in principle remove
     marginalia that reach inside the body's bounding box while preserving the line spacing
     `--psm 6` depends on. It scores 0.1160 / 0.1131 against cropping's 0.1136 / 0.1050 —
     indistinguishable on the paired test, so it buys nothing for noticeably more
     machinery. Cropping stays the recommendation on simplicity.

Every engine is scored through the same alignment and normalisation, so the numbers are
directly comparable. Differences are confirmed with a paired per-page test rather than read
off a mean: page-level CER variance is large enough to swamp a real effect, and `noble`
line-vs-page looked like a +0.0016 win that the CI showed to be indistinguishable.

    python lines_and_trocr.py --engine tesseract-page
    python lines_and_trocr.py --engine tesseract-page-cropped
    python lines_and_trocr.py --engine tesseract-page-masked
    python lines_and_trocr.py --engine tesseract-line --seg boxes --filter body
    python lines_and_trocr.py --engine trocr --seg boxes --filter body
    python significance.py --books nobleza,noble \
        --a tesseract-page:projection:none --b tesseract-page-cropped:projection:none

Crops are cached by filename and reused. After changing filter logic, delete the derived
crops (`rm -f work/*_b?*.png work/*_crop.png work/*_mask.png`) or stale ones are
silently reused.
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


def line_boxes_tsv(img_path, lang="spa", psm=3, min_h=10, min_w=40):
    """Raw (y, x, w, h) textline boxes from Tesseract's layout pass (TSV level 4)."""
    r = subprocess.run(["tesseract", img_path, "stdout", "-l", lang,
                        "--psm", str(psm), "tsv"], capture_output=True, text=True)
    boxes = []
    for row in r.stdout.splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12 or f[0] != "4":          # level 4 == textline
            continue
        x, y, w, h = (int(f[6]), int(f[7]), int(f[8]), int(f[9]))
        if h < min_h or w < min_w:
            continue
        boxes.append((y, x, w, h))
    return boxes


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def body_filter(boxes, min_overlap=0.7, max_h_mult=2.2, max_w_mult=1.3,
                min_h_mult=0.45):
    """Drop line boxes that are not body text: running heads, folio numbers, marginalia.

    Tesseract's layout pass returns every textline it finds, and on this material that
    includes a full-width running head at the top of each page, a right-hand column of
    marginal notes, catchwords and folio numbers at the foot, and a page-tall box over the
    dark scan edge. All of it is recognised and concatenated into the page text, where it
    is scored against a reference transcription that contains none of it. That is pure
    added error: TrOCR reads page 16 almost perfectly but prefixes it with the running head
    ("www a nobieza virtnoja 15 1"), and the CER pays for every character of it.

    The body column is found from the page itself rather than hardcoded, so it survives the
    two books having different margins:

      * boxes far taller than a line are structural, not lines (the scan edge at h=1546
        against a median line height of ~66), so they are dropped before measuring;
      * the widest remaining boxes are full body lines, and their median left and right
        edges define the column;
      * a box is body text if most of it lies inside that column and it is not far wider
        than the column.

    Overlap is used rather than a left-edge test so that legitimately short lines survive —
    the last line of a paragraph, centred headings and indented verse are all real text and
    all narrower than the column.
    """
    if not boxes:
        return boxes
    # First pass only removes boxes too tall to be a line, so that the body-line
    # statistics below are not skewed by the page-tall scan edge.
    mh = _median([b[3] for b in boxes])
    normal = [b for b in boxes if b[3] <= max_h_mult * mh] or list(boxes)

    # The widest boxes are full body lines. Take the column AND the line height from
    # them: measuring height over every box lets a noisy page full of specks drag the
    # threshold down, which is how h=12 slivers survived on `noble`.
    wmed = _median([b[2] for b in normal])
    wide = [b for b in normal if b[2] >= wmed] or normal
    x_lo = _median([b[1] for b in wide])
    x_hi = _median([b[1] + b[2] for b in wide])
    col_w = max(1, x_hi - x_lo)
    body_h = _median([b[3] for b in wide]) or 1

    kept = []
    for y, x, w, h in normal:
        if not (min_h_mult * body_h <= h <= max_h_mult * body_h):
            continue
        overlap = max(0, min(x + w, x_hi) - max(x, x_lo))
        if overlap / w >= min_overlap and w <= max_w_mult * col_w:
            kept.append((y, x, w, h))
    return kept or list(boxes)


def tesseract_line_boxes(img_path, lang="spa", psm=3, pad=4, min_h=10, min_w=40):
    """Return line-image crops from Tesseract's own layout analysis (TSV level 4).

    The horizontal projection in `find_lines` fails on this material for two reasons that
    only show up on real pages: it crops the full page width, so running heads, folio
    numbers and the dark scan edge ride along with every line, and it cannot separate two
    lines whose ascenders and descenders overlap. Tesseract already runs a page layout
    pass before it recognises anything; asking it for the line boxes reuses that work and
    gives a tight box per line instead of a full-width band.
    """
    im = Image.open(img_path).convert("L")
    W, H = im.size
    boxes = line_boxes_tsv(img_path, lang, psm, min_h, min_w)
    if FILTER == "body":
        boxes = body_filter(boxes)
    boxes.sort()                                 # reading order: top, then left

    crops = []
    for n, (y, x, w, h) in enumerate(boxes):
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        dest = img_path.replace(".png", f"_b{FILTER[0]}{n:03d}.png")
        if not os.path.exists(dest):
            im.crop((x0, y0, x1, y1)).save(dest)
        crops.append(dest)
    return crops


SEGMENTERS = {"projection": find_lines, "boxes": tesseract_line_boxes}
SEG = "projection"
FILTER = "none"
PAGE_PSM = 6


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


def tesseract_page_cropped(page_img, lang="spa", pad=8):
    """Page-level OCR (`--psm 6`) on the body block only.

    The line experiments showed that most of the avoidable error is not misrecognition but
    text that should never have been read: running heads, folio numbers and the marginal
    note column, none of which appear in the reference transcription. Filtering those out
    helped every line-level engine. But line-level recognition pays its own price — per-line
    `--psm 7` throws away the page-level language context that makes `--psm 6` the strongest
    engine here, which is why no line variant has beaten the page baseline.

    This keeps `--psm 6` and removes the non-body text instead, by cropping the page to the
    bounding box of the body lines before recognising it. A rectangular crop is enough
    because the marginalia sit in their own column to the right of the body and the running
    head sits above it, so both fall outside the box.
    """
    boxes = line_boxes_tsv(page_img)
    kept = body_filter(boxes)
    im = Image.open(page_img).convert("L")
    W, H = im.size
    if not kept:
        return ocr_page(page_img, lang, PAGE_PSM)
    x0 = max(0, min(b[1] for b in kept) - pad)
    y0 = max(0, min(b[0] for b in kept) - pad)
    x1 = min(W, max(b[1] + b[2] for b in kept) + pad)
    y1 = min(H, max(b[0] + b[3] for b in kept) + pad)
    dest = page_img.replace(".png", "_crop.png")
    if not os.path.exists(dest):
        im.crop((x0, y0, x1, y1)).save(dest)
    return ocr_page(dest, lang, PAGE_PSM)


def tesseract_page_masked(page_img, lang="spa", pad=6):
    """Page-level OCR with every non-body region painted out, page geometry preserved.

    `tesseract_page_cropped` removes non-body text with a single rectangle, which is a
    coarse instrument: anything that happens to fall inside the body's bounding box — a
    marginal note that reaches into the column, a smudge between paragraphs, a catchword on
    the last line — is still inside the crop and still gets read.

    This paints the page white except for the kept body boxes, so non-body text is removed
    wherever it sits rather than only outside a rectangle. The page keeps its original
    dimensions and the body lines keep their original positions, which matters because
    `--psm 6` reads a uniform block and is sensitive to line spacing and indentation.
    """
    boxes = line_boxes_tsv(page_img)
    kept = body_filter(boxes)
    if not kept:
        return ocr_page(page_img, lang, PAGE_PSM)
    im = Image.open(page_img).convert("L")
    W, H = im.size
    canvas = Image.new("L", (W, H), 255)
    for y, x, w, h in kept:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        canvas.paste(im.crop((x0, y0, x1, y1)), (x0, y0))
    dest = page_img.replace(".png", "_mask.png")
    if not os.path.exists(dest):
        canvas.save(dest)
    return ocr_page(dest, lang, PAGE_PSM)


ENGINES = {
    "tesseract-page": lambda p: ocr_page(p, "spa", PAGE_PSM),
    "tesseract-page-cropped": tesseract_page_cropped,
    "tesseract-page-masked": tesseract_page_masked,
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
    ap.add_argument("--filter", default="none", choices=["none", "body"],
                    help="drop running heads, folio numbers and marginalia (seg=boxes only)")
    ap.add_argument("--psm", type=int, default=6,
                    help="page segmentation mode for the page-level engines")
    a = ap.parse_args()
    global SEG, FILTER, PAGE_PSM
    SEG, FILTER, PAGE_PSM = a.seg, a.filter, a.psm
    import jiwer
    from scipy.optimize import linear_sum_assignment

    engine = ENGINES[a.engine]
    for book in a.books.split(","):
        pdf, gt = BOOKS[book]
        halves = [h for im in render(pdf, 300) for h in split_spread(im)]
        refs = reference_pages(gt)
        print(f"\n=== {book} | engine={a.engine} | seg={a.seg} | filter={a.filter} | psm={a.psm} | {len(halves)} pages ===")

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
