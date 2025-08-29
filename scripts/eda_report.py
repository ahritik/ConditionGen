import os, argparse, json

MD_TPL = """# EDA Report

## Meta
- Sampling rate (fs): {fs} Hz
- Window length: {win_sec} s
- Sampled windows (for plots): {sample_n_used}

## Totals
{totals_md}

## Class Balance (train)
{art_md}

- **Seizure prevalence** (train/val/test): {seiz_prev}

## Age & Montage (train)
{age_md}

{montage_md}

## Band Power (overall)
{band_power_md}

## Figures
- Artifact distribution (train): `artifact_distribution_train.png`
- Intensity histogram (train): `intensity_hist_train.png`
- Age bin distribution (train): `agebin_distribution_train.png`
- Seizure prevalence per split: `seizure_prevalence.png`
- Welch PSD (overall): `psd_overall.png`
- Welch PSD by artifact: `psd_by_artifact.png`
- Mean channel covariance: `covariance_mean.png`
- Mean ACF: `acf_mean.png`
- Band power (overall): `band_power_overall.png`

"""

def _md_counts(d):
    if not d: return "- (none)"
    # ensure stable ordering: artifacts or numeric keys
    if all(k.isdigit() for k in d.keys()):
        keys = list(map(int, d.keys()))
        keys.sort()
        return "\n".join([f"- {k}: {d[str(k)]}" for k in keys])
    else:
        keys = sorted(d.keys())
        return "\n".join([f"- {k}: {d[k]}" for k in keys])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda_json", required=True)
    ap.add_argument("--to_html", action="store_true")
    ap.add_argument("--out_html", default="out/eda/EDA_Report.html")
    args = ap.parse_args()

    with open(args.eda_json) as f:
        S = json.load(f)

    fs = S.get("meta",{}).get("fs", 200)
    win_sec = S.get("meta",{}).get("win_sec", 4.0)
    sample_n_used = S.get("sample_n_used", 0)

    totals = S.get("totals", {})
    totals_md = "\n".join([f"- {k}: {v}" for k,v in totals.items()])

    # artifacts (train)
    art_train = S.get("artifact_counts",{}).get("train",{})
    tot_train = max(1, totals.get("train", 1))
    art_lines = []
    for k in sorted(art_train.keys()):
        c = art_train[k]
        art_lines.append(f"- {k}: {c} ({100.0*c/tot_train:.2f}%)")
    art_md = "\n".join(art_lines) if art_lines else "- (none)"

    # seizure prevalence
    seiz = S.get("seizure_counts",{})
    seiz_prev = ", ".join([f"{sp}: { (seiz.get(sp,0)/max(1, totals.get(sp,1))):.3f}" for sp in totals.keys()])

    # age bins & montage (train)
    age_md = _md_counts(S.get("agebin_counts",{}).get("train",{}))
    montage_md = _md_counts(S.get("montage_counts",{}).get("train",{}))

    # band power
    bp = S.get("band_power_overall", {})
    bp_lines = [f"- {k}: {v:.4g}" for k,v in bp.items()] if bp else ["- (n/a)"]
    band_power_md = "\n".join(bp_lines)

    md = MD_TPL.format(
        fs=fs, win_sec=win_sec, sample_n_used=sample_n_used,
        totals_md=totals_md, art_md=art_md, seiz_prev=seiz_prev,
        age_md=age_md, montage_md=montage_md, band_power_md=band_power_md
    )

    out_md = os.path.splitext(args.out_html)[0].replace(".html","") + ".md"
    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    with open(out_md, "w") as f: f.write(md)

    if args.to_html:
        # simple texty HTML; images not embedded, paths listed in report
        html = "<html><body><pre>" + md + "</pre></body></html>"
        with open(args.out_html, "w") as f: f.write(html)

    print("Report written:", out_md, args.out_html if args.to_html else "")

if __name__ == "__main__":
    main()
