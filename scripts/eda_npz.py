import os, glob, json, argparse, math
import numpy as np
from collections import Counter, defaultdict
from scipy.signal import welch
import matplotlib.pyplot as plt

from utils.constants import ARTIFACT_SET, BANDS

def _load_meta(npz_dir):
    meta_p = os.path.join(npz_dir, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f: 
            return json.load(f)
    return {"fs": 200, "win_sec": 4.0}

def _iter_npz(npz_dir, split=None):
    patt = "*.npz" if split is None else f"{split}_*.npz"
    for f in sorted(glob.glob(os.path.join(npz_dir, patt))):
        with np.load(f) as z:
            yield f, z

def summarize(npz_dir, sample_n=5000, splits=("train","val","test")):
    meta = _load_meta(npz_dir)
    fs = int(meta.get("fs", 200))
    win_sec = float(meta.get("win_sec", 4.0))

    totals = {sp: 0 for sp in splits}
    art_counts = {sp: Counter() for sp in splits}
    age_counts = {sp: Counter() for sp in splits}
    seiz_counts = {sp: 0 for sp in splits}
    montage_counts = {sp: Counter() for sp in splits}

    intens_all = {sp: [] for sp in splits}
    intens_by_art = {sp: defaultdict(list) for sp in splits}

    # Sampling buffers for spectral/covariance/ACF
    maxN = int(sample_n)
    sample_X = []
    sample_y_art = []

    for sp in splits:
        for f, z in _iter_npz(npz_dir, split=sp):
            x = z["x"]                     # [N,C,T]
            a = z["y_artifact"]
            g = z["y_agebin"]
            m = z["y_montage"]
            s = z["y_seizure"]
            inten = z["intensity"]
            n = x.shape[0]

            totals[sp] += n
            for idx in a:
                art_counts[sp][ARTIFACT_SET[int(idx)]] += 1
            for idx in g:
                age_counts[sp][int(idx)] += 1
            for idx in m:
                montage_counts[sp][int(idx)] += 1
            seiz_counts[sp] += int(np.sum(s))

            intens_all[sp].append(inten)
            for lbl, it in zip(a, inten):
                intens_by_art[sp][ARTIFACT_SET[int(lbl)]].append(float(it))

            # Subsample windows for heavy metrics
            need = maxN - len(sample_X)
            if need > 0:
                take = min(need, n)
                sample_X.append(x[:take])
                sample_y_art.append(a[:take])

    # stack
    if len(sample_X) > 0:
        sample_X = np.concatenate(sample_X, axis=0)
        sample_y_art = np.concatenate(sample_y_art, axis=0)
    else:
        sample_X = np.zeros((0,8,int(fs*win_sec)), dtype=np.float32)
        sample_y_art = np.zeros((0,), dtype=np.int64)

    # finalize stats
    intens_all = {sp: (np.concatenate(v) if len(v)>0 else np.zeros(0)) for sp,v in intens_all.items()}
    intens_by_art = {sp: {k: np.array(v, dtype=float) for k,v in d.items()} for sp,d in intens_by_art.items()}

    # Compute PSD overlays (overall + by artifact) from sample
    psd = {}
    band_power = {}
    freqs = None
    if sample_X.shape[0] > 0:
        # average over channels per window for plotting simplicity
        Xmean = sample_X.mean(axis=1)  # [N,T]
        freqs, Pxx = welch(Xmean, fs=fs, nperseg=min(512, Xmean.shape[-1]))
        PSD_overall = Pxx.mean(axis=0)  # [F]
        psd["overall"] = PSD_overall.tolist()

        # by artifact
        for ai, name in enumerate(ARTIFACT_SET):
            sel = (sample_y_art == ai)
            if np.sum(sel) > 0:
                freqs_a, Pxx_a = welch(Xmean[sel], fs=fs, nperseg=min(512, Xmean.shape[-1]))
                psd[name] = Pxx_a.mean(axis=0).tolist()

        # band powers (overall)
        bp = {}
        for bname,(lo,hi) in BANDS.items():
            m = (freqs>=lo) & (freqs<hi)
            bp[bname] = float(PSD_overall[m].mean()) if np.any(m) else float("nan")
        band_power["overall"] = bp

    # Average channel covariance from sample
    cov_mean = None
    if sample_X.shape[0] > 0:
        covs = []
        for i in range(sample_X.shape[0]):
            covs.append(np.cov(sample_X[i]))
        cov_mean = np.stack(covs).mean(axis=0)

    # Mean ACF from sample
    def acf(x, nlags=150):
        x = x - x.mean()
        ac = np.correlate(x, x, mode='full')[len(x)-1: len(x)+nlags-1]
        ac /= (np.arange(len(ac))[::-1] + 1e-6)
        ac /= ac[0] + 1e-8
        return ac
    acf_mean = None
    if sample_X.shape[0] > 0:
        acs = []
        for i in range(min(sample_X.shape[0], 1000)):
            # take first channel
            acs.append(acf(sample_X[i,0], nlags=150))
        acf_mean = np.stack(acs).mean(axis=0)

    # Pack JSON
    summary = {
        "meta": meta,
        "totals": totals,
        "artifact_counts": {sp: dict(art_counts[sp]) for sp in splits},
        "agebin_counts": {sp: dict(age_counts[sp]) for sp in splits},
        "montage_counts": {sp: dict(montage_counts[sp]) for sp in splits},
        "seizure_counts": seiz_counts,
        "intensity_hist_10bins": {sp: np.histogram(intens_all[sp], bins=10, range=(0,1))[0].tolist() for sp in splits},
        "intensity_stats": {
            sp: {
                "mean": float(np.mean(intens_all[sp])) if intens_all[sp].size else 0.0,
                "std": float(np.std(intens_all[sp])) if intens_all[sp].size else 0.0,
            } for sp in splits
        },
        "psd": {"freqs": freqs.tolist() if freqs is not None else [], **psd},
        "band_power_overall": band_power.get("overall", {}),
        "cov_mean": cov_mean.tolist() if cov_mean is not None else [],
        "acf_mean": acf_mean.tolist() if acf_mean is not None else [],
        "sample_n_used": int(sample_X.shape[0]),
    }
    return summary

def _bar(ax, labels, values, rotation=0):
    ax.bar(labels, values)
    ax.set_xticklabels(labels, rotation=rotation)

def _save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def make_plots(npz_dir, out_dir, summary):
    os.makedirs(out_dir, exist_ok=True)
    meta = summary.get("meta", {})
    fs = meta.get("fs", 200)

    # 1) Artifact distribution (train)
    train_art = summary["artifact_counts"].get("train", {})
    labs = list(ARTIFACT_SET)
    vals = [train_art.get(k,0) for k in labs]
    fig, ax = plt.subplots()
    _bar(ax, labs, vals, rotation=30)
    ax.set_title("Artifact distribution (train)")
    _save_fig(os.path.join(out_dir, "artifact_distribution_train.png"))

    # 2) Intensity histogram (train)
    fig, ax = plt.subplots()
    ax.bar(np.arange(10), summary["intensity_hist_10bins"]["train"])
    ax.set_title("Intensity histogram (train)")
    ax.set_xlabel("bins (0..1)")
    _save_fig(os.path.join(out_dir, "intensity_hist_train.png"))

    # 3) Age bin distribution (train)
    age_map = {0:"0-12",1:"13-25",2:"26-60",3:"61+"}
    train_age = summary["agebin_counts"].get("train", {})
    labs = [age_map.get(int(k), str(k)) for k in sorted(map(int, train_age.keys()))]
    vals = [train_age[str(k)] if isinstance(next(iter(train_age.keys()),0), str) else train_age[k] for k in sorted(map(int, train_age.keys()))]
    fig, ax = plt.subplots()
    _bar(ax, labs, vals, rotation=0)
    ax.set_title("Age bin distribution (train)")
    _save_fig(os.path.join(out_dir, "agebin_distribution_train.png"))

    # 4) Seizure prevalence per split
    fig, ax = plt.subplots()
    splits = list(summary["totals"].keys())
    prev = [summary["seizure_counts"].get(sp,0)/max(1, summary["totals"].get(sp,1)) for sp in splits]
    _bar(ax, splits, prev, rotation=0)
    ax.set_title("Seizure prevalence per split")
    _save_fig(os.path.join(out_dir, "seizure_prevalence.png"))

    # 5) PSD overlays (overall + by artifact) from sample
    psd = summary.get("psd", {})
    if psd.get("freqs", []):
        freqs = np.array(psd["freqs"])
        # overall
        if "overall" in psd:
            fig, ax = plt.subplots()
            ax.plot(freqs, np.array(psd["overall"]))
            ax.set_xlabel("Hz")
            ax.set_ylabel("PSD")
            ax.set_title("Welch PSD (overall, channel-mean)")
            _save_fig(os.path.join(out_dir, "psd_overall.png"))
        # overlay by artifact
        fig, ax = plt.subplots()
        for k in ARTIFACT_SET:
            if k in psd:
                ax.plot(freqs, np.array(psd[k]), label=k)
        ax.set_xlabel("Hz")
        ax.set_ylabel("PSD")
        ax.set_title("Welch PSD by artifact (channel-mean)")
        ax.legend()
        _save_fig(os.path.join(out_dir, "psd_by_artifact.png"))

    # 6) Channel covariance heatmap
    cov_m = np.array(summary.get("cov_mean", []))
    if cov_m.size:
        fig, ax = plt.subplots()
        im = ax.imshow(cov_m, aspect='auto')
        ax.set_title("Mean channel covariance")
        fig.colorbar(im, ax=ax)
        _save_fig(os.path.join(out_dir, "covariance_mean.png"))

    # 7) Mean ACF curve
    acf_m = np.array(summary.get("acf_mean", []))
    if acf_m.size:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(acf_m)), acf_m)
        ax.set_title("Mean ACF (lag up to 150)")
        ax.set_xlabel("lag")
        _save_fig(os.path.join(out_dir, "acf_mean.png"))

    # 8) Band power bars (overall)
    bp = summary.get("band_power_overall", {})
    if bp:
        fig, ax = plt.subplots()
        labs = list(bp.keys())
        vals = [bp[k] for k in labs]
        _bar(ax, labs, vals)
        ax.set_title("Band power (overall)")
        _save_fig(os.path.join(out_dir, "band_power_overall.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_n", type=int, default=5000, help="windows to sample for heavy metrics & plots")
    ap.add_argument("--splits", type=str, nargs="*", default=["train","val","test"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = summarize(args.npz_dir, sample_n=args.sample_n, splits=tuple(args.splits))
    with open(os.path.join(args.out_dir, "npz_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    make_plots(args.npz_dir, args.out_dir, summary)
    print("Wrote:", os.path.join(args.out_dir, "npz_summary.json"))
    print("Plots saved to:", args.out_dir)

if __name__=="__main__":
    main()
