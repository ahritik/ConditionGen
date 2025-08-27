#!/usr/bin/env python3
import argparse, os, json, math, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.unet1d_film import UNet1D
from models.conditioning import CondEmbedding
from models.diffusion import DDPM

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class WindowDataset(Dataset):
    def __init__(self, npz_path, split=(0.7,0.15,0.15), split_part="train", seed=13):
        z = np.load(npz_path, allow_pickle=True)
        X = z["X"]; y = z["y"]; inten = z["intensity"]; subj = z["subject"]; mask=z["ch_mask"]
        meta = json.loads(z["meta"].item())
        subjects = json.loads(z["subjects"].item())
        self.meta = meta
        # subject-level split
        subj_ids = sorted(set(subj.tolist()))
        rng = np.random.RandomState(seed)
        rng.shuffle(subj_ids)
        n = len(subj_ids); ntrain=int(n*split[0]); nval=int(n*split[1])
        train_ids = set(subj_ids[:ntrain])
        val_ids   = set(subj_ids[ntrain:ntrain+nval])
        test_ids  = set(subj_ids[ntrain+nval:])

        if split_part=="train":
            keep = [i for i in range(len(X)) if subj[i] in train_ids]
        elif split_part=="val":
            keep = [i for i in range(len(X)) if subj[i] in val_ids]
        else:
            keep = [i for i in range(len(X)) if subj[i] in test_ids]

        self.X = X[keep].astype(np.float32)
        self.y = y[keep].astype(np.int64)
        self.inten = inten[keep].astype(np.float32)
        self.mask = mask[keep].astype(np.float32)
        self.C = self.X.shape[1]
        self.T = self.X.shape[2]

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        return {
            "x": x,
            "y": self.y[idx],
            "i": self.inten[idx:idx+1],
        }

def stft_l1(win=128, hop=64):
    import torch.nn.functional as F
    window = torch.hann_window(win)
    def loss(x_true, x_hat):
        # x_*: (B,C,T)
        B,C,T = x_true.size()
        xt = x_true.reshape(B*C, T)
        xh = x_hat.reshape(B*C, T)
        Xt = torch.stft(xt, n_fft=win, hop_length=hop, window=window.to(xt.device), return_complex=True)
        Xh = torch.stft(xh, n_fft=win, hop_length=hop, window=window.to(xh.device), return_complex=True)
        return (Xt.abs() - Xh.abs()).abs().mean()
    return loss

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--widths", type=int, nargs="+", default=[64,128,256])
    ap.add_argument("--resblocks", type=int, default=2)
    ap.add_argument("--stft_win", type=int, default=128)
    ap.add_argument("--stft_hop", type=int, default=64)
    ap.add_argument("--stft_lambda", type=float, default=0.1)
    ap.add_argument("--cfg_pdrop", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--mps", action="store_true")
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    dev = torch.device("cpu")
    if torch.cuda.is_available(): dev = torch.device("cuda")
    elif args.mps and torch.backends.mps.is_available(): dev = torch.device("mps")
    print("Device:", dev)

    train_ds = WindowDataset(args.data, split_part="train", seed=args.seed)
    val_ds   = WindowDataset(args.data, split_part="val", seed=args.seed)
    C,T = train_ds.C, train_ds.T
    print("Train size:", len(train_ds), "Val size:", len(val_ds), "C,T:", C, T)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

    # Cond embeddings
    artifact_k = 7
    cond_dim = 128
    cond_embed = CondEmbedding(artifact_k=artifact_k, intensity_dim=1, d_model=cond_dim).to(dev)

    net = UNet1D(channels=C, widths=tuple(args.widths), resblocks=args.resblocks, time_dim=256, cond_dim=cond_dim).to(dev)
    ddpm = DDPM(net, timesteps=1000, schedule="cosine").to(dev)

    opt = torch.optim.AdamW(list(net.parameters()) + list(cond_embed.parameters()), lr=args.lr)

    def do_eval(dl):
        net.eval(); cond_embed.eval()
        losses=[]
        with torch.no_grad():
            for batch in dl:
                x0 = torch.tensor(batch["x"], device=dev)
                y  = torch.tensor(batch["y"], device=dev)
                i  = torch.tensor(batch["i"], device=dev)
                t  = torch.randint(0, ddpm.T, (x0.size(0),), device=dev)
                cond = cond_embed(y, i)
                eps = torch.randn_like(x0)
                xt = ddpm.q_sample(x0, t, eps)
                eps_pred = net(xt, t, cond)
                losses.append(((eps_pred - eps)**2).mean().item())
        return float(np.mean(losses))

    best = 1e9
    os.makedirs("ckpts", exist_ok=True)
    scaler = torch.amp.GradScaler("mps") if (args.amp and dev.type=="mps") else None
    spec_loss = stft_l1(args.stft_win, args.stft_hop)

    pbar = tqdm(range(args.steps), desc="train")
    step=0
    net.train(); cond_embed.train()
    while step < args.steps:
        for batch in train_dl:
            step += 1
            x0 = torch.tensor(batch["x"], device=dev)
            y  = torch.tensor(batch["y"], device=dev)
            i  = torch.tensor(batch["i"], device=dev)
            t  = torch.randint(0, ddpm.T, (x0.size(0),), device=dev)
            cond = cond_embed(y, i)

            if scaler:
                with torch.amp.autocast("mps"):
                    loss, logs = ddpm.p_losses(x0, t, cond, stft_loss=spec_loss, stft_lambda=args.stft_lambda)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss, logs = ddpm.p_losses(x0, t, cond, stft_loss=spec_loss, stft_lambda=args.stft_lambda)
                opt.zero_grad(); loss.backward(); opt.step()

            if step % 100 == 0:
                val = do_eval(val_dl)
                pbar.set_postfix(loss=float(loss.item()), val=val, mse=logs.get("mse",0.0), stft=logs.get("stft",0.0))
                if val < best:
                    best = val
                    torch.save({"net": net.state_dict(), "cond": cond_embed.state_dict()}, "ckpts/best.pt")
            if step >= args.steps: break
        pbar.update(1)
    print("Best val:", best)

if __name__ == "__main__":
    main()
