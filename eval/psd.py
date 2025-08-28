# eval/psd.py
"""
Compute PSD band errors between real and synthetic windows.
"""
import numpy as np
import scipy.signal as sps

BANDS = {"delta": (0.5,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30)}

def bandpower(sig, fs, f1, f2):
    f, Pxx = sps.welch(sig, fs=fs, nperseg=256)
    mask = (f>=f1) & (f<=f2)
    return np.trapz(Pxx[mask], f[mask])

def psd_band_errors(real, synth, fs=200):
    """
    real, synth: arrays (N,C,T) normalized.
    Returns dict of absolute band errors averaged across channels and samples.
    """
    errs = {b: [] for b in BANDS}
    for xr, xs in zip(real, synth):
        for ch in range(xr.shape[0]):
            for b,(f1,f2) in BANDS.items():
                pr = bandpower(xr[ch], fs, f1, f2)
                ps = bandpower(xs[ch], fs, f1, f2)
                errs[b].append(abs(pr-ps))
    return {k: float(np.mean(v)) for k,v in errs.items()}
