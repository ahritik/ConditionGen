# sample.py
import os, argparse, torch, numpy as np
from models.unet1d_film import UNet1DFiLM
from models.conditioning import ConditionEmbed
from models.diffusion import Diffusion
from utils.constants import ARTIFACT_SET, MONTAGE_IDS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--length", type=int, default=800, help="Samples length T (e.g., 4s@200Hz=800)")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--widths", type=str, default="64,128,256")
    ap.add_argument("--sampler", type=str, default="heun2", choices=["heun2","ddim"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--out_npz", type=str, default="out/samples.npz")
    ap.add_argument("--artifact", type=str, default="none", choices=ARTIFACT_SET)
    ap.add_argument("--intensity", type=float, default=0.0)
    ap.add_argument("--seizure", type=int, default=0)
    ap.add_argument("--age_bin", type=int, default=2)
    ap.add_argument("--montage", type=str, default="canon8")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    widths = tuple(int(x) for x in args.widths.split(","))
    film_provider = ConditionEmbed(film_dim=widths[0])
    net = UNet1DFiLM(in_channels=args.channels, widths=widths, num_res_blocks=2, film_provider=film_provider).to(device)
    diff = Diffusion(net, T=1000, stft_lambda=0.0, device=device).to(device)

    ck = torch.load(args.ckpt, map_location="cpu")
    diff.load_state_dict(ck["diff"])
    if args.use_ema:
        # copy EMA weights into model
        shadow = ck["ema"]
        diff.load_state_dict(shadow, strict=False)

    # Build cond dict
    art_idx = ARTIFACT_SET.index(args.artifact)
    montage_id = MONTAGE_IDS.get(args.montage, 0)

    cond = {
        "artifact": torch.full((args.n,), art_idx, dtype=torch.long, device=device),
        "intensity": torch.full((args.n,1), float(args.intensity), dtype=torch.float32, device=device),
        "seizure": torch.full((args.n,1), float(args.seizure), dtype=torch.float32, device=device),
        "age_bin": torch.full((args.n,), int(args.age_bin), dtype=torch.long, device=device),
        "montage_id": torch.full((args.n,), int(montage_id), dtype=torch.long, device=device),
    }

    x_T = torch.randn(args.n, args.channels, args.length, device=device)
    if args.sampler == "heun2":
        x = diff.heun2_sample(x_T, cond, steps=args.steps)
    else:
        x = diff.ddim_sample(x_T, cond, steps=args.steps, eta=0.0)

    X = x.detach().cpu().numpy()
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, x=X)
    print(f"Wrote samples to {args.out_npz}")

if __name__ == "__main__":
    main()
