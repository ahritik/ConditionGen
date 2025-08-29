import os, argparse, json

TPL = """# EDA Report

## NPZ Summary
- Windows: {n_windows}

### Artifact counts
{artifact_counts}

### Intensity histogram (10 bins)
{intensity_hist}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda_json", required=True)
    ap.add_argument("--to_html", action="store_true")
    ap.add_argument("--out_html", default="out/eda/EDA_Report.html")
    args = ap.parse_args()
    with open(args.eda_json) as f: s = json.load(f)
    md = TPL.format(**s)
    out_md = os.path.splitext(args.out_html)[0].replace(".html","") + ".md"
    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    with open(out_md, "w") as f: f.write(md)
    if args.to_html:
        # trivial markdown to html
        html = "<html><body><pre>" + md + "</pre></body></html>"
        with open(args.out_html, "w") as f: f.write(html)
    print("Report written:", out_md, args.out_html if args.to_html else "")

if __name__ == "__main__":
    main()
