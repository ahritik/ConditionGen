# eval/cov_acf.py
import numpy as np

def channel_covariance_distance(real, synth):
    """
    Computes Frobenius distance between mean channel covariance matrices.
    """
    def covs(X):
        # X: (N,C,T)
        cov_list = []
        for x in X:
            cov_list.append(np.cov(x))
        return np.mean(cov_list, axis=0)
    Cr = covs(real)
    Cs = covs(synth)
    return float(np.linalg.norm(Cr - Cs, ord="fro"))

def acf_distance(real, synth, max_lag=50):
    """
    Mean absolute difference of autocorrelation up to max_lag.
    """
    def acf(sig, max_lag):
        sig = sig - sig.mean()
        ac = np.correlate(sig, sig, mode="full")
        mid = len(ac)//2
        ac = ac[mid:mid+max_lag+1]
        ac = ac / (ac[0] + 1e-8)
        return ac
    diffs = []
    for xr, xs in zip(real, synth):
        C = xr.shape[0]
        for c in range(C):
            ar = acf(xr[c], max_lag)
            as_ = acf(xs[c], max_lag)
            diffs.append(np.abs(ar - as_).mean())
    return float(np.mean(diffs))
