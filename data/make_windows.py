#!/usr/bin/env python3
import argparse, os, re, json, math, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import mne

ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]

CANON_CH = ["Fp1","Fp2","C3","C4","P3","P4","O1","O2"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuar_root", type=str, required=True, help="Root folder containing EDF files")
    ap.add_argument("--ann_csv", type=str, default=None, help="Optional annotations CSV (edf_path,start_sec,end_sec,artifact)")
    ap.add_argument("--out", type=str, required=True, help="Output NPZ path")
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--win_s", type=float, default=4.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--bandpass", type=float, nargs=2, default=[0.5,45.0])
    ap.add_argument("--notch", type=float, default=60.0)
    ap.add_argument("--canon_ch", type=str, nargs="+", default=CANON_CH)
    ap.add_argument("--subject_regex", type=str, default=r"(?P<subject>[^/\\]+)")
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()

def find_edf_files(root: str) -> List[str]:
    exts = (".edf",".bdf",".gdf")
    files = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(dirpath,f))
    return sorted(files)

def load_annotations_csv(csv_path: str) -> Dict[str, List[Tuple[float,float,str]]]:
    if csv_path is None:
        return {}
    rows = defaultdict(list)
    with open(csv_path,"r") as f:
        header = f.readline().strip().split(",")
        cols = {h:i for i,h in enumerate(header)}
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split(",")
            edf = parts[cols["edf_path"]]
            t0 = float(parts[cols["start_sec"]]); t1 = float(parts[cols["end_sec"]])
            lab = parts[cols["artifact"]].strip().lower()
            if lab not in ARTIFACT_SET:
                lab = "none"
            rows[edf].append((t0,t1,lab))
    return rows

def map_channels(raw: mne.io.BaseRaw, canon_ch: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    # Return data (C,T) mapped to canonical order; missing channels zero-filled, mask indicates present
    present = []
    for ch in canon_ch:
        if ch in raw.ch_names:
            present.append(True)
        else:
            # try variants like 'EEG Fp1-Ref' etc.
            found = None
            for name in raw.ch_names:
                base = name.replace("EEG","").replace("-Ref","").replace("-LE","").replace("-A1","").strip()
                if base == ch:
                    found = name; break
            present.append(found is not None)
    data = []
    mask = []
    for ok, ch in zip(present, canon_ch):
        if ok:
            pick = mne.pick_channels(raw.ch_names, include=[ch], ordered=False)
            if len(pick)==0:
                # try relaxed match
                for name in raw.ch_names:
                    base = name.replace("EEG","").replace("-Ref","").replace("-LE","").replace("-A1","").strip()
                    if base == ch:
                        pick = mne.pick_channels(raw.ch_names, include=[name], ordered=False)
                        break
            x = raw.get_data(picks=pick)
            data.append(x[0])
            mask.append(1.0)
        else:
            data.append(np.zeros(raw.n_times, dtype=np.float32))
            mask.append(0.0)
    return np.stack(data,0).astype(np.float32), np.array(mask, dtype=np.float32)

def segment_windows(x: np.ndarray, fs: int, win_s: float, overlap: float):
    C,T = x.shape
    win = int(win_s*fs)
    hop = int(win*(1.0-overlap))
    idx = []
    for start in range(0, T-win+1, hop):
        idx.append((start, start+win))
    return idx

def per_window_label(segments: List[Tuple[float,float,str]], t0: float, t1: float) -> str:
    # Any-overlap strategy; choose majority by duration if multiple
    if not segments: return "none"
    votes = defaultdict(float)
    for (s0, s1, lab) in segments:
        inter = max(0.0, min(t1,s1)-max(t0,s0))
        if inter > 0:
            votes[lab] += inter
    if not votes: return "none"
    return max(votes.items(), key=lambda kv: kv[1])[0]

def compute_intensity(win: np.ndarray, fs: int) -> float:
    # Simple RMS normalized within-recording later; here return RMS
    return float(np.sqrt(np.mean(win**2)))

def main():
    args = parse_args()
    np.random.seed(args.seed)
    files = find_edf_files(args.tuar_root)
    ann = load_annotations_csv(args.ann_csv)
    if len(files)==0:
        raise SystemExit(f"No EDF-like files found under {args.tuar_root}")
    print(f"Found {len(files)} recordings")

    X, Y, INT, SUBJ, MASK = [], [], [], [], []
    meta = {"artifact_set": ARTIFACT_SET, "canon_ch": args.canon_ch, "fs": args.fs, "win_s": args.win_s, "overlap": args.overlap}
    subj_map = {}
    subj_re = re.compile(args.subject_regex)

    for i,edf in enumerate(files):
        print(f"[{i+1}/{len(files)}] {edf}")
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        except Exception as e:
            print("  !! skip (read error):", e); continue

        # Filter chain
        try:
            if args.notch>0:
                raw.notch_filter(freqs=[args.notch], picks='eeg')
            raw.filter(l_freq=args.bandpass[0], h_freq=args.bandpass[1], picks='eeg')
        except Exception as e:
            print("  !! filter error, skipping:", e); continue

        # Resample
        try:
            raw.resample(args.fs)
        except Exception as e:
            print("  !! resample error, skipping:", e); continue

        # Map channels
        x, mask = map_channels(raw, args.canon_ch)  # (C,T)
        # Z-score per channel (robust)
        x = (x - np.median(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True)+1e-8)

        # Subject id
        m = subj_re.search(edf.replace("\\","/"))
        sid = m.group("subject") if m and "subject" in m.groupdict() else Path(edf).parts[-2]
        if sid not in subj_map: subj_map[sid] = len(subj_map)
        sid_i = subj_map[sid]

        # Annotation segments for file
        segs = ann.get(edf, [])
        idx = segment_windows(x, args.fs, args.win_s, args.overlap)

        # Per-recording RMS stats for intensity normalization
        rms_all = []

        for (a,b) in idx:
            win = x[:,a:b]  # (C,win)
            t0, t1 = a/args.fs, b/args.fs
            lab = per_window_label(segs, t0, t1)
            y = ARTIFACT_SET.index(lab) if lab in ARTIFACT_SET else 0
            rms = compute_intensity(win, args.fs)
            rms_all.append(rms)

        # normalize RMS to [0,1] within recording
        if len(rms_all)==0: continue
        rmin, rmax = float(np.percentile(rms_all, 5)), float(np.percentile(rms_all, 95))
        p = 0
        for (a,b) in idx:
            win = x[:,a:b]
            t0, t1 = a/args.fs, b/args.fs
            lab = per_window_label(segs, t0, t1)
            y = ARTIFACT_SET.index(lab) if lab in ARTIFACT_SET else 0
            rms = np.clip((rms_all[p]-rmin)/(rmax-rmin+1e-8), 0.0, 1.0)
            p+=1
            X.append(win.astype(np.float32))
            Y.append(y)
            INT.append(float(rms))
            SUBJ.append(sid_i)
            MASK.append(mask.astype(np.float32))

    if len(X)==0:
        raise SystemExit("No windows produced. Check annotations or channel mapping.")

    X = np.stack(X,0)      # (N,C,T)
    Y = np.array(Y, np.int64)
    INT = np.array(INT, np.float32)
    SUBJ = np.array(SUBJ, np.int64)
    MASK = np.stack(MASK,0)  # (N,C)

    np.savez_compressed(args.out, X=X, y=Y, intensity=INT, subject=SUBJ, ch_mask=MASK, meta=json.dumps(meta), subjects=json.dumps(subj_map))
    print("Saved:", args.out, "shape:", X.shape)

if __name__ == "__main__":
    main()
