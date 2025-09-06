# utils/normalize_fakes.py
import os, glob, numpy as np, argparse
ap=argparse.ArgumentParser()
ap.add_argument("--real_dir", required=True)
ap.add_argument("--synth_parent", required=True)
a=ap.parse_args()

reals=[]
for f in sorted(glob.glob(os.path.join(a.real_dir, "*.npz"))):
    with np.load(f) as z:
        x=z["x"].astype(np.float32)
        if x.ndim==2: x=x[None,...]
        reals.append(x)
R=np.concatenate(reals,0)
target = R.std(axis=-1).mean(axis=0)  # [C]

for d in sorted(glob.glob(os.path.join(a.synth_parent, "synth_*"))):
    p=os.path.join(d,"samples.npy")
    if not os.path.exists(p): continue
    X=np.load(p).astype(np.float32)
    if X.ndim==2: X=X[None,...]
    cur=X.std(axis=-1)+1e-6
    alpha=(target[None,:]/cur).clip(0.5,2.0)
    X=(X.transpose(0,2,1)*alpha).transpose(0,2,1)
    np.save(p,X); print("normalized", d, X.shape)
