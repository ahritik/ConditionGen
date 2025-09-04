# save as format_tables.py and run:  python format_tables.py
import os, json, math, glob

ARTS = ["none","eye","muscle","chewing","shiver","electrode"]
RUNS = ["base","ft"]
MODELS = ["tiny","resnet1d","eegnet"]

def load(path):
    with open(path) as f: return json.load(f)

def fmt(x, k=4):
    return f"{x:.{k}f}"

def table1():
    for run in RUNS:
        rows = []
        for a in ARTS:
            psd = load(f"out/metrics_{run}/psd_{a}.json")
            ca  = load(f"out/metrics_{run}/covacf_{a}.json")
            d = psd["band_rel_err"]
            cov = ca["cov_fro"]
            acf = ca["acf_l2"] / math.sqrt(150)  # scaled, clearer magnitude
            n_fake = psd.get("n", 0)
            rows.append([a, fmt(d["delta"]), fmt(d["theta"]), fmt(d["alpha"]), fmt(d["beta"]),
                         fmt(cov,3), fmt(acf,3), str(n_fake)])
        print(f"\n# Table 1 — Fidelity ({run})")
        print("| Artifact | Δδ | Δθ | Δα | Δβ | Cov Fro ↓ | ACF L2↓ (scaled) | n_fake |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            print("| " + " | ".join(r) + " |")

def table2():
    for run in RUNS:
        print(f"\n## Table 2 — Specificity (recovery, {run})")
        for m in MODELS:
            print(f"### {m}")
            print("| Artifact | IM |")
            print("|---|---:|")
            for a in ARTS:
                # try a couple of common keys
                d = load(f"out/clf_eval_{run}/recovery_{a}_{m}.json")
                im = (d.get("recovery",{}) or d).get("IM") \
                     or d.get("intended_match") \
                     or d.get("intended_match_rate")
                # fallback if nested differently
                if im is None and "metrics" in d: im = d["metrics"].get("IM")
                print(f"| {a} | {fmt(float(im) if im is not None else 0.0,3)} |")

if __name__ == "__main__":
    table1()
    table2()
