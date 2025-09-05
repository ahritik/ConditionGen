#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/eda_viz.py
----------------
Visualization of EDA outputs produced by utils/eda.py.

Reads CSV/JSON from <base>/eda/<split>/ and writes PNG figures to:
  <base>/eda_viz/<split>/...

Charts per split
----------------
- artifact_counts.png       (bar)
- age_bin_counts.png        (bar)
- montage_id_counts.png     (bar)
- channel_rms.png           (bar)
- intensity_hist.png        (if present: from intensity_hist.json)

Combined charts
---------------
- artifact_stacked.png      (stacked bars across splits, if >=2 splits)

Usage
-----
python utils/eda_viz.py --base out/tuar_npz
"""

from __future__ import annotations
import os, json, argparse, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load_label_map(base: str):
    f = os.path.join(base, "label_map.json")
    if os.path.exists(f):
        try:
            j = json.load(open(f))
            names = j.get("artifact_names")
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass
    return ["none","eye","muscle","chewing","shiver","electrode"]


def per_split_viz(base: str, split: str, class_names):
    in_dir = os.path.join(base, "eda", split)
    out_dir = os.path.join(base, "eda_viz", split)
    if not os.path.isdir(in_dir):
        return False
    os.makedirs(out_dir, exist_ok=True)

    # Artifact distribution
    fp_art = os.path.join(in_dir, "artifact_counts.csv")
    if os.path.exists(fp_art):
        df = pd.read_csv(fp_art)
        fig = plt.figure(figsize=(8, 4.5))
        plt.bar(df["artifact"], df["count"])
        plt.title(f"Artifact Distribution — {split}")
        plt.xlabel("Artifact")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "artifact_counts.png"), dpi=150)
        plt.close(fig)

    # Age bin distribution
    fp_age = os.path.join(in_dir, "age_bin_counts.csv")
    if os.path.exists(fp_age):
        df = pd.read_csv(fp_age)
        fig = plt.figure(figsize=(6, 4))
        plt.bar(df["age_bin"].astype(str), df["count"])
        plt.title(f"Age-bin Distribution — {split}")
        plt.xlabel("Age bin (0: <18, 1: 18–39, 2: 40–64, 3: 65+)")
        plt.ylabel("Count")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "age_bin_counts.png"), dpi=150)
        plt.close(fig)

    # Montage id distribution
    fp_m = os.path.join(in_dir, "montage_id_counts.csv")
    if os.path.exists(fp_m):
        df = pd.read_csv(fp_m)
        fig = plt.figure(figsize=(6, 4))
        plt.bar(df["montage_id"].astype(str), df["count"])
        plt.title(f"Montage IDs — {split}")
        plt.xlabel("montage_id")
        plt.ylabel("Count")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "montage_id_counts.png"), dpi=150)
        plt.close(fig)

    # Channel RMS
    fp_ch = os.path.join(in_dir, "channel_stats.csv")
    if os.path.exists(fp_ch):
        df = pd.read_csv(fp_ch)
        fig = plt.figure(figsize=(8, 4))
        plt.bar(df["channel"].astype(str), df["rms"])
        plt.title(f"Channel RMS — {split}")
        plt.xlabel("Channel index")
        plt.ylabel("RMS")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "channel_rms.png"), dpi=150)
        plt.close(fig)

    # Intensity histogram
    fp_int = os.path.join(in_dir, "intensity_hist.json")
    if os.path.exists(fp_int):
        j = json.load(open(fp_int))
        if j.get("present", False):
            bins = np.array(j["bins"], dtype=float)
            counts = np.array(j["counts"], dtype=float)
            centers = 0.5 * (bins[:-1] + bins[1:])
            fig = plt.figure(figsize=(7, 4))
            plt.bar(centers, counts, width=(bins[1]-bins[0]), align="center")
            plt.title(f"Intensity Histogram — {split}")
            plt.xlabel("Intensity")
            plt.ylabel("Count")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "intensity_hist.png"), dpi=150)
            plt.close(fig)

    print(f"[viz] wrote figures to {out_dir}")
    return True


def combined_artifact_stacked(base: str, splits: list, class_names):
    # Collect per-split artifact counts
    rows = []
    for s in splits:
        fp = os.path.join(base, "eda", s, "artifact_counts.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            # ensure order matches class_names
            df = df.set_index("artifact").reindex(class_names).reset_index()
            rows.append((s, df["count"].to_numpy()))
    if len(rows) < 2:
        return

    out_dir = os.path.join(base, "eda_viz")
    os.makedirs(out_dir, exist_ok=True)

    # stacked bar by split
    fig = plt.figure(figsize=(9, 5))
    indices = np.arange(len(rows))
    bottom = np.zeros(len(rows), dtype=float)

    for ci, cname in enumerate(class_names):
        vals = np.array([counts[ci] for _, counts in rows], dtype=float)
        plt.bar([r[0] for r in rows], vals, bottom=bottom, label=cname)
        bottom += vals

    plt.title("Artifact Distribution Across Splits (stacked)")
    plt.xlabel("Split")
    plt.ylabel("Count")
    plt.legend(loc="upper right", ncol=2, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "artifact_stacked.png"), dpi=150)
    plt.close(fig)
    print(f"[viz] wrote combined stacked chart to {os.path.join(out_dir, 'artifact_stacked.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base folder with eda/<split>/* produced by eda.py")
    ap.add_argument("--splits", nargs="+", default=["train","val","eval","test"],
                    help="Which split names to look for")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    class_names = load_label_map(base)

    existing = []
    for s in args.splits:
        ok = per_split_viz(base, s, class_names)
        if ok: existing.append(s)

    if len(existing) >= 2:
        combined_artifact_stacked(base, existing, class_names)

if __name__ == "__main__":
    main()
