"""RenAIssance (HumanAI) baseline: Tesseract OCR on 17th-century Spanish print.

Establishes the honest floor before trying anything clever. Off-the-shelf OCR is the right
baseline here because the premise of the RenAIssance project is that tools like this fail on
early modern print.

Two data realities had to be handled before any score meant anything:

1. **The extracts and the transcriptions do not line up.** Each supplied PDF holds 16 pages
   while each transcription carries 25 `PDF pN` markers, so the markers refer to the original
   book's numbering and the extract is a subset. Scoring concatenated text against
   concatenated reference measures misalignment, not OCR quality. Pages are therefore matched
   by content: each OCR'd page is assigned the reference page it most resembles, and only
   confident matches are scored.

2. **The orthography is not modern Spanish.** The transcription notes say u and v are used
   interchangeably, and the long s is transcribed as s though OCR routinely reads it as f.
   Scores are reported raw and after folding those two, which separates "the OCR misread the
   page" from "1600s spelling differs from today's".
"""
import os, re, subprocess, glob, argparse, unicodedata
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(ROOT), "DeepLearnHackathon",
                    "NLPRenaissanceChallenge", "data")
WORK = os.path.join(ROOT, "work")

BOOKS = {
    "nobleza": ("Padilla - Nobleza virtuosa_testExtract.pdf",
                "Padilla - 1 Nobleza virtuosa_testTranscription.docx"),
    "noble":   ("Padilla - 2 Noble perfecto_Extract.pdf",
                "Padilla - 2 Noble perfecto_Transcription.docx"),
}
MARK = re.compile(r"^\s*pdf\s*p?\s*(\d+)\s*$", re.I)


def reference_pages(docx_name):
    """Split the transcription into {page_number: text} on its PDF pN markers."""
    import docx
    paras = [p.text for p in docx.Document(os.path.join(DATA, docx_name)).paragraphs
             if p.text.strip()]
    pages, cur = {}, None
    for p in paras:
        m = MARK.match(p)
        if m:
            cur = int(m.group(1))
            pages[cur] = []
        elif cur is not None:
            pages[cur].append(p)
    return {k: "\n".join(v) for k, v in pages.items() if v}


def render(pdf_name, dpi):
    os.makedirs(WORK, exist_ok=True)
    stem = os.path.join(WORK, re.sub(r"\W+", "_", pdf_name)[:40])
    if not glob.glob(stem + "-*.png"):
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png",
                        os.path.join(DATA, pdf_name), stem], check=True)
    # exclude the _L/_R half-page crops this script writes next to the originals,
    # or a second run re-splits already-split pages into quarters
    return sorted(f for f in glob.glob(stem + "-*.png")
                  if not re.search(r"_[LR]\.png$", f))


def split_spread(img):
    """Each scan is a two-page spread of the open book, so the OCR of one image
    contains two book pages (visible as two running headers). Cut down the middle
    and return the left and right halves as separate page images."""
    from PIL import Image
    out = []
    im = Image.open(img)
    w, h = im.size
    for side, box in (("L", (0, 0, w // 2, h)), ("R", (w // 2, 0, w, h))):
        dest = img.replace(".png", f"_{side}.png")
        if not os.path.exists(dest):
            im.crop(box).save(dest)
        out.append(dest)
    return out


def ocr_page(img, lang, psm):
    r = subprocess.run(["tesseract", img, "stdout", "-l", lang, "--psm", str(psm)],
                       capture_output=True, text=True)
    return r.stdout


def token_sim(a, b):
    """Containment of the reference's words in the OCR output, ignoring very short
    words. Containment rather than Jaccard because an OCR page also picks up running
    headers and catchwords that the transcription omits, which would unfairly punish
    a symmetric measure."""
    A = {w for w in a.split() if len(w) > 3}
    B = {w for w in b.split() if len(w) > 3}
    if not B:
        return 0.0
    return len(A & B) / len(B)


def norm(s, fold_uv=False, fold_fs=False):
    s = unicodedata.normalize("NFC", s).replace("ſ", "s")   # long s
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    if fold_uv:
        s = s.replace("v", "u")
    if fold_fs:
        s = s.replace("f", "s")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--lang", default="spa")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--min-sim", type=float, default=0.25,
                    help="minimum content similarity to accept a page match")
    a = ap.parse_args()
    import jiwer

    for book, (pdf, gt) in BOOKS.items():
        imgs = render(pdf, a.dpi)
        refs = reference_pages(gt)
        print(f"\n=== {book} ===")
        print(f"rendered pages: {len(imgs)}   reference pages: {len(refs)} "
              f"({min(refs)}..{max(refs)})")

        halves = [h for im in imgs for h in split_spread(im)]
        print(f"split into {len(halves)} single book pages")
        ocr = {i + 1: ocr_page(p, a.lang, a.psm) for i, p in enumerate(halves)}

        # Content alignment. Character-level SequenceMatcher on a prefix is far too
        # brittle here: OCR noise and line-break differences wreck it, and greedy
        # first-come assignment lets an early bad page consume the right reference.
        # Token containment is robust to both, and the assignment is solved globally
        # so one page's mistake cannot cascade.
        pk = sorted(ocr)
        rk_list = sorted(refs)
        sim = [[token_sim(norm(ocr[pi]), norm(refs[rk])) for rk in rk_list] for pi in pk]
        pairs = []
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(-np.array(sim))
            cand = [(pk[i], rk_list[j], sim[i][j]) for i, j in zip(ri, ci)]
        except ImportError:
            cand, used = [], set()
            for i, pi in enumerate(pk):
                order = sorted(range(len(rk_list)), key=lambda j: -sim[i][j])
                for j in order:
                    if rk_list[j] not in used:
                        used.add(rk_list[j]); cand.append((pi, rk_list[j], sim[i][j])); break
        for pi, rkm, sc in cand:
            if sc >= a.min_sim:
                pairs.append((pi, rkm, sc, ocr[pi], refs[rkm]))
        pairs.sort(key=lambda t: -t[2])

        print(f"aligned {len(pairs)}/{len(halves)} pages (similarity >= {a.min_sim})")
        if not pairs:
            print("  no confident alignment - scores would be meaningless, skipping")
            continue

        for label, kw in [("raw", {}), ("u/v folded", {"fold_uv": True}),
                          ("u/v + f/s folded", {"fold_uv": True, "fold_fs": True})]:
            cers, wers = [], []
            for _, _, _, pred, ref in pairs:
                p, r = norm(pred, **kw), norm(ref, **kw)
                if r:
                    cers.append(jiwer.cer(r, p))
                    wers.append(jiwer.wer(r, p))
            print(f"  {label:<18} CER {sum(cers)/len(cers):.4f}   WER {sum(wers)/len(wers):.4f}")

        pi, rk, s, pred, ref = pairs[0]
        print(f"  best-aligned example: rendered p{pi} -> reference p{rk} (sim {s:.2f})")
        print(f"    OCR: {norm(pred)[:95]}")
        print(f"    REF: {norm(ref)[:95]}")


if __name__ == "__main__":
    main()
