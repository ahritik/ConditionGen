#!/usr/bin/env python3
import argparse, json, numpy as np

def corr_mat(x):
    # x: (C,T)
    xm = x - x.mean(axis=-1, keepdims=True)
    xs = xm / (xm.std(axis=-1, keepdims=True)+1e-8)
    return xs @ xs.T / xs.shape[1]

def acf_dist(x, L=200):
    # x: (C,T)
    C,T = x.shape
    d=0.0; n=0
    for c in range(C):
        s = x[c]
        s = (s - s.mean())/(s.std()+1e-8)
        acf = np.correlate(s, s, mode="full")[T-1:T+L] / np.arange(T, T-L, -1)
        # target: identity (1 at lag0, near 0 later); compare across sets later
        # here just return energy beyond lag0 as simple proxy
        d += float(np.sum(acf[1:]**2)); n+=1
    return d/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=str, required=True)
    ap.add_argument("--gen", type=str, required=True)
    args = ap.parse_args()

    zr = np.load(args.real, allow_pickle=True); Xr=zr["X"]
    zg = np.load(args.gen, allow_pickle=True); Xg=zg["X"]
    n = min(len(Xr), len(Xg))
    Xr = Xr[:n]; Xg=Xg[:n]
    Cr = np.stack([corr_mat(x) for x in Xr],0).mean(0)
    Cg = np.stack([corr_mat(x) for x in Xg],0).mean(0)
    frob = float(np.sqrt(((Cr-Cg)**2).sum()))
    acf_r = np.mean([acf_dist(x) for x in Xr])
    acf_g = np.mean([acf_dist(x) for x in Xg])
    out = {"corr_frobenius": frob, "acf_energy_r": acf_r, "acf_energy_g": acf_g}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
