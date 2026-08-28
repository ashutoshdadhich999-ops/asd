"""
Compares the spiking audio model against each strong non-spiking baseline
(1D U-Net, Dilated/WaveNet-style CNN, Residual TCN) in turn, in addition
to the topology-matched ablation baseline. If the spiking model remains
competitive against these harder baselines, that is a substantially
stronger result than beating only the matched-topology ANN.

Usage (from repo root):
    python scripts/run_baseline_comparison.py
    python scripts/run_baseline_comparison.py --epochs 10 --audio-subset-size 3000
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchaudio

from src.datasets import AudioDS
from src.models_audio import StrongAudioNet, NonSpikeAudioNet
from src.models_audio_baselines import BASELINES
from src.corruption_registry import build_corruption
from src.train import train_audio_model
from src.evaluate import evaluate_audio


def parse_args():
    p = argparse.ArgumentParser(description="Spiking vs strong ANN baselines (audio)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--audio-len", type=int, default=8000)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--timesteps-audio", type=int, default=40)
    p.add_argument("--num-steps-audio", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--audio-subset-size", type=int, default=6000)
    p.add_argument("--corruption", type=str, default="poisson",
                    choices=["poisson", "gaussian", "bernoulli"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="outputs/baseline_comparison_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    os.makedirs("./data", exist_ok=True)
    base = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True)
    subset = Subset(base, range(min(args.audio_subset_size, len(base))))
    tr_size = int(0.85 * len(subset))
    tr_sub, te_sub = random_split(subset, [tr_size, len(subset) - tr_size])

    train_loader = DataLoader(AudioDS(tr_sub, args.audio_len, args.audio_sr),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(AudioDS(te_sub, args.audio_len, args.audio_sr),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    corruption = build_corruption(args.corruption, T=args.timesteps_audio, device=device)

    # ---- Spiking model (trained once, compared against every baseline) ----
    torch.manual_seed(args.seed)
    model_s = StrongAudioNet(num_steps=args.num_steps_audio,
                              T_audio=args.timesteps_audio).to(device)
    model_s = train_audio_model(model_s, "Spiking Audio", train_loader, corruption,
                                 args.timesteps_audio, args.epochs, args.lr, device)
    res_s = evaluate_audio(model_s, "Spiking Audio", test_loader, corruption,
                            args.timesteps_audio, device, seed=args.seed)

    architectures = {"matched": NonSpikeAudioNet(T_audio=args.timesteps_audio).to(device)}
    for name, Cls in BASELINES.items():
        architectures[name] = Cls(T_audio=args.timesteps_audio).to(device)

    results = {"spiking": res_s}
    for name, model in architectures.items():
        print("\n" + "=" * 70)
        print(f"BASELINE: {name}")
        print("=" * 70)
        torch.manual_seed(args.seed)
        model = train_audio_model(model, f"Non-Spiking Audio ({name})", train_loader,
                                   corruption, args.timesteps_audio, args.epochs, args.lr, device)
        res = evaluate_audio(model, f"Non-Spiking Audio ({name})", test_loader, corruption,
                              args.timesteps_audio, device, seed=args.seed)
        results[name] = res

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'MSE':>10} {'SI-SDR Imp (dB)':>18} {'SNR Imp (dB)':>15}")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:<20} {r['MSE']:>10.5f} {r['SI-SDR Imp']:>18.2f} {r['SNR Imp']:>15.2f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.save_path}")


if __name__ == "__main__":
    main()
