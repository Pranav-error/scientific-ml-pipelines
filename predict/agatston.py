"""COCA Common Task: Agatston coronary calcium scoring from the calcium_xml annotations.

The XMLs are OsiriX ROI exports (Apple plist). Each ROI carries Area and peak HU, which
is everything the Agatston score needs — so this runs without downloading the 21 GB of
gated DICOMs.

Agatston score (Agatston et al. 1990):
    per lesion:  area_mm2 * density_factor(peak_HU)
    density factor: 130-199 -> 1, 200-299 -> 2, 300-399 -> 3, >=400 -> 4
    lesions below 130 HU are not calcium and are excluded
    total = sum over all lesions on all slices

Caveat kept explicit: the classical definition also requires a minimum lesion area
(commonly 1 mm^2, i.e. >=3 contiguous pixels) and assumes 3 mm slices. Both are
reported below so the effect of the choice is visible rather than hidden.
"""
import os, plistlib, glob, argparse, collections
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
XMLDIR = os.path.join(ROOT, "data", "gated", "calcium_xml")


def density_factor(peak_hu):
    if peak_hu >= 400: return 4
    if peak_hu >= 300: return 3
    if peak_hu >= 200: return 2
    if peak_hu >= 130: return 1
    return 0


def parse(path, area_units="cm2", min_area_mm2=1.0):
    """Return one row per ROI (lesion on a slice)."""
    with open(path, "rb") as fh:
        d = plistlib.load(fh)
    pid = os.path.splitext(os.path.basename(path))[0]
    rows = []
    for img in d.get("Images", []):
        idx = img.get("ImageIndex")
        for roi in img.get("ROIs", []):
            area = roi.get("Area")
            if area is None:
                continue
            area_mm2 = area * 100.0 if area_units == "cm2" else area
            peak = roi.get("Max", 0) or 0
            rows.append({
                "patient": pid, "slice": idx, "artery": roi.get("Name", "?"),
                "area_mm2": area_mm2, "peak_hu": peak, "mean_hu": roi.get("Mean"),
                "n_points": roi.get("NumberOfPoints"),
                "density": density_factor(peak),
                "counted": bool(density_factor(peak) > 0 and area_mm2 >= min_area_mm2),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area", type=float, default=0.0, help="min lesion area mm^2")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(XMLDIR, "*.xml")))
    print(f"annotation files: {len(files)}")
    rows = []
    bad = []
    for f in files:
        try:
            rows += parse(f, min_area_mm2=args.min_area)
        except Exception as e:
            bad.append((os.path.basename(f), str(e)[:60]))
    print(f"parse failures: {len(bad)}")
    for b in bad[:5]:
        print("   ", b)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no ROIs parsed")
        return
    df.to_csv(os.path.join(ROOT, "lesions.csv"), index=False)
    print(f"lesions: {len(df)} across {df.patient.nunique()} patients")

    print("\nlesion area (mm^2):", f"min={df.area_mm2.min():.2f} med={df.area_mm2.median():.2f} "
          f"max={df.area_mm2.max():.2f}")
    print("peak HU:           ", f"min={df.peak_hu.min():.0f} med={df.peak_hu.median():.0f} "
          f"max={df.peak_hu.max():.0f}")
    print("\nlesions by artery:")
    for a, n in df.artery.value_counts().items():
        print(f"   {a:<28} {n}")
    print("\nlesions by density factor:")
    print("  ", dict(sorted(collections.Counter(df.density).items())))
    excluded = (~df.counted).sum()
    print(f"\nexcluded (HU<130 or area<{args.min_area}mm^2): {excluded} of {len(df)}")

    use = df[df.counted]
    scores = (use.area_mm2 * use.density).groupby(use.patient).sum()
    # patients whose every lesion was excluded still score 0
    scores = scores.reindex(sorted(df.patient.unique(), key=int), fill_value=0.0)
    scores.name = "agatston"
    scores.to_csv(os.path.join(ROOT, "agatston_scores.csv"))

    def cat(s):
        if s == 0: return "0 (none)"
        if s < 100: return "1-99 (mild)"
        if s < 400: return "100-399 (moderate)"
        return "400+ (severe)"

    print(f"\n===== AGATSTON SCORES ({len(scores)} patients) =====")
    print(f"min={scores.min():.1f}  median={scores.median():.1f}  "
          f"mean={scores.mean():.1f}  max={scores.max():.1f}")
    print("\nrisk categories:")
    vc = scores.map(cat).value_counts()
    for k in ["0 (none)", "1-99 (mild)", "100-399 (moderate)", "400+ (severe)"]:
        if k in vc:
            print(f"   {k:<22} {vc[k]:>4}  ({100*vc[k]/len(scores):.1f}%)")
    print("\nwrote lesions.csv + agatston_scores.csv")


if __name__ == "__main__":
    main()
