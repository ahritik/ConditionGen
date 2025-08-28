# scripts/eda_report.py
import os, argparse, json
from datetime import datetime

TEMPLATE = """# EDA Report

Generated: {ts}

## TUAR Summary
- CSV files: {csv_files}
- Rows: {rows}

### Label counts
{label_lines}

## NPZ Summary
- Windows: {N}
- Class distribution:
{class_lines}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuar_json", type=str, required=True)
    ap.add_argument("--npz_json", type=str, required=True)
    ap.add_argument("--out_md", type=str, default="out/eda/EDA_Report.md")
    ap.add_argument("--to_html", action="store_true")
    args = ap.parse_args()

    with open(args.tuar_json) as f:
        tj = json.load(f)
    with open(args.npz_json) as f:
        nj = json.load(f)

    label_lines = "\n".join([f"- {k}: {v}" for k,v in tj.get("label_counts",{}).items()])
    class_lines = "\n".join([f"- {k}: {v}" for k,v in nj.get("counts",{}).items()])

    md = TEMPLATE.format(
        ts=datetime.utcnow().isoformat(),
        csv_files=tj.get("csv_files",0),
        rows=tj.get("rows",0),
        label_lines=label_lines or "(none)",
        N=nj.get("N",0),
        class_lines=class_lines or "(none)"
    )
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)
    print(f"Wrote {args.out_md}")

    if args.to_html:
        try:
            import markdown
            html = markdown.markdown(md)
            out_html = os.path.splitext(args.out_md)[0] + ".html"
            with open(out_html, "w") as f:
                f.write(html)
            print(f"Wrote {out_html}")
        except Exception as e:
            print(f"[WARN] Couldn't create HTML: {e} (pip install markdown)")

if __name__ == "__main__":
    main()
