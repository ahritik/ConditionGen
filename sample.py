import os, argparse, json, numpy as np, torch
from models.unet1d_film import UNet1DFiLM
from models.diffusion import Diffusion1D, EMA
from utils.constants import ARTIFACT_SET

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--artifact", type=str, default="none", choices=ARTIFACT_SET)
    ap.add_argument("--intensity", type=float, default=0.5)
    ap.add_argument("--seizure", type=int, default=0)
    ap.add_argument("--age_bin", type=int, default=2)
    ap.add_argument("--montage_id", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--out_dir", type=str, default="out/samples")
    args = ap.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    cond_dim = len(ARTIFACT_SET) + 1 + 4 + 1
    net = UNet1DFiLM(c_in=8, c_hidden=(64,128,256), cond_dim=cond_dim)
    model = Diffusion1D(net, timesteps=1000, stft_lambda=0.0)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if args.use_ema and ckpt.get("ema") is not None:
        # Load EMA weights into model
        i = 0
        for p in model.parameters():
            if not p.requires_grad: continue
            p.data.copy_(ckpt["ema"][i])
            i += 1

    model = model.to(device)
    model.eval()

    B = args.n
    T = 800  # 4s@200Hz
    C = 8

    # Build cond and null-cond for guidance
    art_idx = ARTIFACT_SET.index(args.artifact)
    art_onehot = torch.zeros(B, len(ARTIFACT_SET), device=device); art_onehot[:, art_idx] = 1.0
    seiz = torch.full((B,1), float(args.seizure), device=device)
    age = torch.zeros(B,4, device=device); age[:, args.age_bin] = 1.0
    mont = torch.full((B,1), float(args.montage_id), device=device)
    cond = torch.cat([art_onehot, seiz, age, mont], dim=-1)
    cond_null = torch.cat([torch.nn.functional.one_hot(torch.tensor(0, device=device), num_classes=len(ARTIFACT_SET)).float().unsqueeze(0).repeat(B,1),
                           torch.zeros(B,1, device=device),
                           torch.zeros(B,4, device=device),
                           mont], dim=-1)

    with torch.no_grad():
        x = model.ddim_sample((B,C,T), cond, steps=args.steps, guidance_scale=args.guidance, cond_null=cond_null, device=device)

    np.save(os.path.join(args.out_dir, "samples.npy"), x.cpu().numpy())
    meta = {k: getattr(args, k) for k in vars(args)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved samples to {args.out_dir}")

if __name__ == "__main__":
    main()
