"""Is the line pipeline actually better than the page baseline, or is the gap noise?

On `noble` the line pipeline scores CER 0.1430 against the page baseline's 0.1446. That is
a margin of 0.0016 over 25 pages, which is far too small to read off a mean and call a win.
This scores both engines page by page, pairs them on the same reference page, and asks two
questions the mean cannot answer:

  * the paired mean difference and its bootstrap confidence interval
  * how often each engine wins a page (a sign test, which ignores outlier magnitude)

Pairing matters: pages differ enormously in difficulty, and that variance swamps the
between-engine difference unless the same page is compared against itself.

    python significance.py --books noble
"""
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
import jiwer

from baseline_ocr import render, reference_pages, norm, token_sim, split_spread, BOOKS
import lines_and_trocr as L


def page_cers(book, engine_name, seg, filt, min_sim=0.25):
    """Return {reference_page: CER} for one engine configuration."""
    L.SEG, L.FILTER = seg, filt
    engine = L.ENGINES[engine_name]
    pdf, gt = BOOKS[book]
    halves = [h for im in render(pdf, 300) for h in split_spread(im)]
    refs = reference_pages(gt)
    ocr = {i + 1: engine(p) for i, p in enumerate(halves)}
    pk, rl = sorted(ocr), sorted(refs)
    sim = np.array([[token_sim(norm(ocr[p]), norm(refs[r])) for r in rl] for p in pk])
    ri, ci = linear_sum_assignment(-sim)
    out = {}
    for i, j in zip(ri, ci):
        if sim[i][j] < min_sim:
            continue
        p, r = norm(ocr[pk[i]], fold_uv=True, fold_fs=True), norm(refs[rl[j]], fold_uv=True,
                                                                  fold_fs=True)
        if r:
            out[rl[j]] = jiwer.cer(r, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="noble")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--a", default="tesseract-page:projection:none",
                    help="reference config, engine:seg:filter")
    ap.add_argument("--b", default="tesseract-line:boxes:body",
                    help="challenger config, engine:seg:filter")
    a = ap.parse_args()
    cfg_a, cfg_b = a.a.split(":"), a.b.split(":")
    rng = np.random.default_rng(0)

    for book in a.books.split(","):
        base = page_cers(book, *cfg_a)
        line = page_cers(book, *cfg_b)
        common = sorted(set(base) & set(line))
        b = np.array([base[k] for k in common])
        l = np.array([line[k] for k in common])
        d = b - l                                    # positive => B is better

        idx = rng.integers(0, len(d), size=(a.boot, len(d)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        wins = int((d > 0).sum())

        print(f"\n=== {book} | {len(common)} paired pages ===")
        print(f"  A {a.a:<38} CER {b.mean():.4f}")
        print(f"  B {a.b:<38} CER {l.mean():.4f}")
        print(f"  paired mean diff    {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  B wins {wins}/{len(d)} pages")
        print("  verdict:", "B is better" if lo > 0 else
              "A is better" if hi < 0 else "indistinguishable — CI spans zero")


if __name__ == "__main__":
    main()
