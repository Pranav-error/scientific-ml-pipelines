"""PREDICT prep — explore the COCA dataset on Redivis and pull a working subset.

COCA is 55,165 files across 2 tables. Do NOT try to download it all: the reference
2026 submission (Quant-Quasar/ML4SCI-PrediCT-Radiomics) worked with 30 patients.
This script lists what's there, previews the tables, and downloads DICOMs for a
small number of patients only.

Requires a Redivis API token with data access:
    stanford.redivis.com -> avatar -> Workspace -> API tokens
    export REDIVIS_API_TOKEN=...
"""
import os, sys, argparse

if not os.environ.get("REDIVIS_API_TOKEN"):
    sys.exit("REDIVIS_API_TOKEN is not set. Create one at stanford.redivis.com "
             "(avatar -> Workspace -> API tokens) and `export REDIVIS_API_TOKEN=...`")

import redivis

ROOT = os.path.dirname(os.path.abspath(__file__))
ORG = "AIMI"
DATASET = "coca_coronary_calcium_and_chest_ct_s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=0,
                    help="how many patients' files to download (0 = explore only)")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()

    ds = redivis.organization(ORG).dataset(DATASET)
    info = ds.get()
    print(f"dataset: {info.properties.get('name')}  v{info.properties.get('version', {}).get('tag')}")
    print(f"files: {info.properties.get('fileCount')}  size: {info.properties.get('numBytes')}")

    print("\n=== TABLES ===")
    tables = ds.list_tables()
    for t in tables:
        p = t.get().properties
        print(f"  {p['name']:<40} rows={p.get('numRows')} vars={p.get('variableCount')}")

    for t in tables:
        p = t.get().properties
        print(f"\n--- {p['name']}: variables ---")
        for v in t.list_variables():
            vp = v.properties
            print(f"    {vp['name']:<28} {vp.get('type','')}")
        try:
            df = t.to_pandas_dataframe(max_results=5)
            print(f"--- {p['name']}: first rows ---")
            print(df.to_string(max_colwidth=40))
        except Exception as e:
            print(f"    (preview failed: {e})")

    if args.patients:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"\n=== downloading files for {args.patients} patients -> {args.outdir} ===")
        files = ds.list_files(max_results=args.patients * 400)
        seen, n = set(), 0
        for f in files:
            name = f.properties.get("name", "")
            pid = name.split("/")[0] if "/" in name else name.split("_")[0]
            if pid not in seen:
                if len(seen) >= args.patients:
                    break
                seen.add(pid)
            f.download(args.outdir, overwrite=False)
            n += 1
        print(f"downloaded {n} files for patients: {sorted(seen)}")


if __name__ == "__main__":
    main()
