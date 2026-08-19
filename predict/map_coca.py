"""Map the COCA file layout before downloading 28.9 GB of DICOMs.

Pulls just the file_name/size columns from both tables (cheap), works out how many
patients there are and how big each is, and locates the small but important files:
scores.xlsx (labels) and any .xml calcium segmentations.
"""
import os, sys, re, collections

if not os.environ.get("REDIVIS_API_TOKEN"):
    sys.exit("REDIVIS_API_TOKEN not set")

import redivis, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ds = redivis.organization("AIMI").dataset("coca_coronary_calcium_and_chest_ct_s")


def summarize(tname):
    t = ds.table(tname)
    df = t.to_pandas_dataframe(variables=["file_name", "size", "file_id"])
    print(f"\n{'='*60}\n{tname}: {len(df)} files, {df['size'].sum()/1e9:.2f} GB")

    ext = collections.Counter(os.path.splitext(n)[1].lower() for n in df.file_name)
    print("  extensions:", dict(ext.most_common()))

    # non-DICOM files are the interesting small ones (labels, annotations)
    small = df[~df.file_name.str.lower().str.endswith(".dcm")]
    print(f"  non-DICOM files: {len(small)}")
    for _, r in small.head(15).iterrows():
        print(f"    {r.file_name}  ({r['size']} bytes)")

    # patient id = first path segment that looks like a number
    def pid(n):
        parts = n.split("/")
        for p in parts:
            if p.isdigit():
                return p
        return None
    df["pid"] = df.file_name.map(pid)
    per = df.dropna(subset=["pid"]).groupby("pid").agg(files=("file_name", "size"),
                                                       bytes=("size", "sum"))
    print(f"  patients: {len(per)}")
    if len(per):
        print(f"  files/patient: min={per.files.min()} med={int(per.files.median())} max={per.files.max()}")
        print(f"  MB/patient:    min={per.bytes.min()/1e6:.1f} med={per.bytes.median()/1e6:.1f} "
              f"max={per.bytes.max()/1e6:.1f}")
        print("  smallest 8 patients:", list(per.sort_values('bytes').index[:8]))
    return df


if __name__ == "__main__":
    frames = {}
    for tn in ("non-gated", "gated"):
        try:
            frames[tn] = summarize(tn)
        except Exception as e:
            print(f"{tn}: FAILED {e}")
    for tn, df in frames.items():
        df.to_csv(os.path.join(ROOT, f"filelist_{tn}.csv"), index=False)
    print("\nwrote filelist_*.csv")
