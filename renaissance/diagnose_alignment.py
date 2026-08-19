"""Why does no OCR page match any reference page? Diagnose before trusting any score."""
import glob, os
from difflib import SequenceMatcher
from baseline_ocr import render, reference_pages, ocr_page, norm, BOOKS

for book, (pdf, gt) in BOOKS.items():
    imgs = render(pdf, 300)
    refs = reference_pages(gt)
    print(f"\n=== {book}: {len(imgs)} images, {len(refs)} reference pages ===")
    ref_lens = {k: len(norm(v)) for k, v in refs.items()}
    print("reference page lengths:", dict(list(ref_lens.items())[:8]), "...")

    for pi in (0, 3, 7):
        if pi >= len(imgs):
            continue
        t = ocr_page(imgs[pi], "spa", 6)
        n = norm(t)
        scores = sorted(((SequenceMatcher(None, n[:1500], norm(v)[:1500]).ratio(), k)
                         for k, v in refs.items()), reverse=True)[:3]
        print(f"\n-- rendered page {pi+1}: {len(n)} chars")
        print(f"   OCR: {n[:120]}")
        print(f"   best refs: {[(k, round(s,3)) for s, k in scores]}")
        k = scores[0][1]
        print(f"   REF p{k}: {norm(refs[k])[:120]}")
