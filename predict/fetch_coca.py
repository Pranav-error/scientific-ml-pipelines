"""Selective COCA download.

Order of operations matters here: the labels (scores.xlsx) and the 451 calcium
segmentation XMLs total ~1.5 MB, while the DICOMs are 28.9 GB. Pull the small
high-value files first, then only as many patients' scans as you actually need.

  python fetch_coca.py --meta                  # scores.xlsx + all calcium XMLs (~1.5 MB)
  python fetch_coca.py --patients 10           # + DICOMs for 10 annotated patients (~250 MB)
"""
import os, sys, argparse, collections

if not os.environ.get("REDIVIS_API_TOKEN"):
    sys.exit("REDIVIS_API_TOKEN not set")

import redivis, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
ds = redivis.organization("AIMI").dataset("coca_coronary_calcium_and_chest_ct_s")


def files_df(tname):
    csv = os.path.join(ROOT, f"filelist_{tname}.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv)
    return ds.table(tname).to_pandas_dataframe(variables=["file_name", "size", "file_id"])


def grab(tname, rows, subdir):
    out = os.path.join(DATA, subdir)
    os.makedirs(out, exist_ok=True)
    t = ds.table(tname)
    done = 0
    for _, r in rows.iterrows():
        dest = os.path.join(out, r.file_name)
        if os.path.exists(dest):
            done += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        t.file(r.file_id).download(os.path.dirname(dest), overwrite=False)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(rows)}")
    print(f"  {subdir}: {done} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="store_true", help="labels + calcium XMLs only")
    ap.add_argument("--patients", type=int, default=0, help="how many annotated patients' DICOMs")
    args = ap.parse_args()

    ng = files_df("non-gated")
    g = files_df("gated")

    if args.meta or args.patients:
        print("== scores.xlsx ==")
        grab("non-gated", ng[ng.file_name.str.endswith(".xlsx")], "non-gated")
        print("== calcium XMLs ==")
        grab("gated", g[g.file_name.str.endswith(".xml")], "gated")

    if args.patients:
        xml_pids = {os.path.splitext(os.path.basename(n))[0]
                    for n in g.file_name if n.endswith(".xml")}
        def pid(n):
            for p in n.split("/"):
                if p.isdigit():
                    return p
            return None
        g = g.assign(pid=g.file_name.map(pid))
        dcm = g[g.file_name.str.lower().str.endswith(".dcm") & g.pid.isin(xml_pids)]
        per = dcm.groupby("pid")["size"].sum().sort_values()
        chosen = list(per.index[:args.patients])          # smallest first
        sel = dcm[dcm.pid.isin(chosen)]
        print(f"\n== DICOMs for {len(chosen)} annotated patients "
              f"({sel['size'].sum()/1e6:.0f} MB): {chosen} ==")
        grab("gated", sel, "gated")


if __name__ == "__main__":
    main()
