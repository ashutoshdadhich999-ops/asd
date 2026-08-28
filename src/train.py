"""Training loops for the image and audio denoisers."""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_img_model(model, name, train_loader, diff, timesteps, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\nTraining {name}...")
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, _ in tqdm(train_loader, leave=False, desc=f"{name} Ep {ep + 1}"):
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            xt = diff.q_sample(x, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        print(f"Epoch {ep + 1}/{epochs}  Loss: {total / len(train_loader):.4f}")
    return model


def train_audio_model(model, name, train_loader, corruption, T_audio, epochs, lr, device):
    """
    Args:
        corruption: any object exposing `.corrupt(x0, t, device) -> (noisy, target)`,
            e.g. a PoissonDiffusion / GaussianAudioDiffusion / BernoulliAudioDiffusion
            instance from src/corruption_registry.py. This makes the training
            loop agnostic to which forward corruption process is being used,
            which is what src/corruption_registry.py + scripts/run_noise_ablation.py
            rely on.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\nTraining {name}...")
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x0 in tqdm(train_loader, leave=False, desc=f"{name} Ep {ep + 1}"):
            x0 = x0.to(device)
            t = torch.randint(0, T_audio, (x0.size(0),), device=device)
            noisy, target = corruption.corrupt(x0, t, device)
            pred = model(noisy, t)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        print(f"Epoch {ep + 1}/{epochs}  Loss: {total / len(train_loader):.5f}")
    return model
