"""COCA Project 2: PyRadiomics feature extraction on annotated calcium lesions.

Run with the radiomics venv (Python 3.11), NOT the main one:
    ~/Documents/gsoc-prep/.venv-radiomics/bin/python radiomics_pipeline.py

Pipeline:
  1. load a patient's gated DICOM series, sorted by ImagePositionPatient z
  2. rasterise each calcium_xml ROI polygon (Point_px) into a binary mask on its slice
  3. verify the mask actually lands on calcium before trusting it — the XML records the
     lesion's Mean/Max HU, so compare against the pixels the mask selects. A silent
     slice-indexing mismatch would otherwise produce plausible-looking garbage features.
  4. extract PyRadiomics features per lesion

Step 3 is the part most pipelines skip. ImageIndex is assumed to be a 0-based index into
the sorted series, and that assumption is checked rather than trusted.
"""
import os, sys, glob, plistlib, re, argparse, collections
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
XMLDIR = os.path.join(ROOT, "data", "gated", "calcium_xml")
DCMROOT = os.path.join(ROOT, "data", "gated", "patient")

PT = re.compile(r"\(([-\d.]+),\s*([-\d.]+)")


def load_series(pid):
    """Return (volume HxWxN in HU, list of pydicom datasets sorted by z)."""
    files = glob.glob(os.path.join(DCMROOT, str(pid), "**", "*.dcm"), recursive=True)
    if not files:
        return None, []
    ds = []
    for f in files:
        try:
            d = pydicom.dcmread(f)
            if hasattr(d, "pixel_array"):
                ds.append(d)
        except Exception:
            pass
    if not ds:
        return None, []
    def zpos(d):
        ipp = getattr(d, "ImagePositionPatient", None)
        return float(ipp[2]) if ipp else float(getattr(d, "InstanceNumber", 0))
    ds.sort(key=zpos)
    vol = np.stack([d.pixel_array.astype(np.float32) for d in ds])
    # DICOM -> Hounsfield units
    inter = float(getattr(ds[0], "RescaleIntercept", 0))
    slope = float(getattr(ds[0], "RescaleSlope", 1))
    vol = vol * slope + inter
    return vol, ds


def polygon_mask(points_px, shape):
    pts = []
    for p in points_px:
        m = PT.match(p)
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))
    if len(pts) < 3:
        return None
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--max-patients", type=int, default=10)
    args = ap.parse_args()

    pids = sorted([d for d in os.listdir(DCMROOT)] if os.path.isdir(DCMROOT) else [], key=int)
    pids = pids[:args.max_patients]
    print(f"patients with DICOMs: {pids}")

    from radiomics import featureextractor
    import logging
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("shape2D")
    extractor.settings["force2D"] = True

    rows, checks = [], []
    for pid in pids:
        xml = os.path.join(XMLDIR, f"{pid}.xml")
        if not os.path.exists(xml):
            print(f"  {pid}: no XML, skipping")
            continue
        vol, ds = load_series(pid)
        if vol is None:
            print(f"  {pid}: no readable DICOMs, skipping")
            continue
        d = plistlib.load(open(xml, "rb"))
        n_ok = 0
        for img in d.get("Images", []):
            idx = img.get("ImageIndex")
            if idx is None or idx >= len(ds):
                checks.append({"patient": pid, "slice": idx, "issue": "index out of range",
                               "n_slices": len(ds)})
                continue
            sl = vol[idx]
            for roi in img.get("ROIs", []):
                mask = polygon_mask(roi.get("Point_px", []), sl.shape)
                if mask is None or mask.sum() == 0:
                    continue
                vals = sl[mask > 0]
                # ---- the verification: does the mask land where the XML says? ----
                checks.append({
                    "patient": pid, "slice": idx, "artery": roi.get("Name"),
                    "xml_mean": roi.get("Mean"), "mask_mean": float(vals.mean()),
                    "xml_max": roi.get("Max"), "mask_max": float(vals.max()),
                    "n_px": int(mask.sum()),
                })
                if args.verify_only:
                    continue
                try:
                    im = sitk.GetImageFromArray(sl[np.newaxis].astype(np.float32))
                    ma = sitk.GetImageFromArray(mask[np.newaxis].astype(np.uint8))
                    fv = extractor.execute(im, ma, label=1)
                    r = {"patient": pid, "slice": idx, "artery": roi.get("Name")}
                    # NOTE: do NOT filter with np.isscalar here — PyRadiomics returns
                    # numpy scalars (np.float64), for which np.isscalar() is False, so
                    # that silently drops almost every feature. Coerce and catch instead.
                    for k, v in fv.items():
                        if k.startswith("diagnostics"):
                            continue
                        try:
                            r[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                    rows.append(r)
                    n_ok += 1
                except Exception as e:
                    checks.append({"patient": pid, "slice": idx,
                                   "issue": f"extract failed: {str(e)[:60]}"})
        print(f"  {pid}: {len(ds)} slices, {n_ok} lesions extracted")

    ck = pd.DataFrame(checks)
    ck.to_csv(os.path.join(ROOT, "mask_verification.csv"), index=False)
    good = ck.dropna(subset=["xml_max"]) if "xml_max" in ck else pd.DataFrame()
    if len(good):
        good = good.copy()
        good["max_err"] = (good.mask_max - good.xml_max).abs()
        good["mean_err"] = (good.mask_mean - good.xml_mean).abs()
        print(f"\n===== MASK ALIGNMENT CHECK ({len(good)} lesions) =====")
        print(f"  |mask_max - xml_max|:  median={good.max_err.median():.1f} "
              f"p90={good.max_err.quantile(.9):.1f} max={good.max_err.max():.1f}")
        print(f"  |mask_mean - xml_mean|: median={good.mean_err.median():.1f} "
              f"p90={good.mean_err.quantile(.9):.1f}")
        agree = (good.max_err < 1).mean()
        print(f"  exact peak-HU agreement: {agree*100:.1f}% of lesions")
        print("  -> near-zero error means ImageIndex maps directly to sorted-slice order.")
        print("  -> large error means the slice indexing assumption is WRONG; fix before")
        print("     trusting any extracted feature.")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(ROOT, "radiomics_features.csv"), index=False)
        print(f"\nextracted {len(df)} lesions x {df.shape[1]-3} features "
              f"-> radiomics_features.csv")


if __name__ == "__main__":
    main()
