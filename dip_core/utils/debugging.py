import torch
import torch.fft as fft
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

import copy

from tqdm import tqdm, trange


# ============================================================
# 1) SPECTRAL BIAS TEST  — track high-frequency energy
# ============================================================

def high_freq_energy(img: torch.Tensor, cutoff_ratio=0.3):
    """
    Computes (high-frequency_energy, total_energy).
    Idea: Neural nets learn low freq patterns first.
    If high_freq stays tiny while loss plateaus → spectral bias.

    img : 2D (H x W) or 3D (C x H x W). We reduce to grayscale via mean.
    cutoff_ratio : radius% that defines high-frequency annulus.
    """
    if img.dim() == 3:
        img = img.mean(0)  # collapse channels

    F = fft.fftshift(fft.fft2(img))
    mag = torch.abs(F)

    H, W = mag.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    r = torch.sqrt((yy - H/2)**2 + (xx - W/2)**2)
    cutoff = cutoff_ratio * min(H, W) / 2

    mask = r > cutoff
    hf = mag[mask].sum().item()
    total = mag.sum().item()
    return hf, total


# ============================================================
# 2) ACTIVATION + GRADIENT HOOKS  — saturation / dead neurons
# ============================================================

def attach_activation_gradient_hooks(model):
    """
    Returns dicts (acts, grads) filled AFTER a forward+backward pass.

    Idea:
    - If activations push to extremes (±1 for tanh, all zeros for ReLU)
    - OR gradients ~0
    → neurons are saturated or dead.
    """
    acts, grads = {}, {}

    def fwd_hook(name):
        def hook(m, inp, out):
            acts[name] = out.detach().cpu()
        return hook

    def bwd_hook(name):
        def hook(m, grad_input, grad_output):
            grads[name] = grad_output[0].detach().cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.register_forward_hook(fwd_hook(name))
            module.register_full_backward_hook(bwd_hook(name))

    return acts, grads


# ============================================================
# 3) PER-LAYER GRADIENT NORM ANALYSIS — conditioning
# ============================================================

def gradient_norm_stats(model):
    """
    Returns array of gradient norms per parameter.

    Idea:
    - If min << max by factor > 1000 → poor conditioning.
    - If all norms tiny → flat loss landscape.
    """
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms.append((name, p.grad.data.norm().item()))
    return norms


def summarize_gradient_norms(norms):
    """
    norms: list of (name, value)
    Returns (min, mean, max, std).
    """
    if len(norms) == 0:
        return None
    vals = np.array([v for _, v in norms])
    return vals.min(), vals.mean(), vals.max(), vals.std()


# ============================================================
# 4) RANDOM-TARGET MEMORIZATION TEST — implicit architecture bias
# ============================================================

def can_memorize_random_target(model, image, steps=200, lr=1e-3, device='cpu'):
    """
    Trains model to fit a random image with MSE.
    Returns final loss.

    Idea:
    - If model cannot fit pure noise → architecture imposes a strong prior.
    - If it fits easily → architecture has high capacity (no hard bottleneck).
    """
    model = model.to(device)
    target = torch.randn_like(image).to(device)
    inp = image.to(device).detach()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in trange(steps):
        opt.zero_grad()
        out = model(inp)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()

    return loss.item()


# ============================================================
# 5) LR SWEEP — optimizer dynamics
# ============================================================

def lr_sweep(model, loss_fn, data_fn, lrs, steps=10, device='cpu'):
    """
    Performs a very small LR sweep.
    Returns list of (lr, loss_after_steps).

    model   : your model class (will be deep-copied per LR)
    loss_fn : function (output, target) -> loss
    data_fn : function that returns (input, target) batch
    lrs     : iterable of learning rates
    steps   : steps per LR

    Idea:
    - Find LR zone where loss decreases fastest.
    - If your current LR sits in flat region → training stuck.
    """

    results = []

    for lr in lrs:
        m = copy.deepcopy(model).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=lr)

        for _ in range(steps):
            inp, target = data_fn()
            inp, target = inp.to(device), target.to(device)
            opt.zero_grad()
            out = m(inp)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()

        results.append((lr, loss.item()))

    return results


class Debugger:
    """
    Modular & extendable debugging framework for neural networks.

    Features:
    - Built-in tests (spectral, activations, grads, norms, lr-sweep, memorization)
    - Plugin system: add new test functions dynamically
    - Unified logging system
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

        # central logging store
        self.log = defaultdict(list)

        # install forward/backward hooks once
        self.acts, self.grads = attach_activation_gradient_hooks(self.model)

        # registry of available debugging tests
        self.tests = {
            "spectral": self._run_spectral,
            "activations": self._run_activations,
            "gradients": self._run_gradients,
            "grad_norms": self._run_grad_norms,
            "memorization": self._run_memorization,
            "lr_sweep": self._run_lr_sweep
        }

    # ---------------------------------------------------------
    # STATIC EXTENSION API
    # ---------------------------------------------------------
    def add_test(self, name, fn):
        """
        Add a new test dynamically.
        Example:
            dbg.add_test("jacobian_rank", test_jacobian_rank)
        """
        self.tests[name] = fn

    def run(self, name, *args, **kwargs):
        """
        Run any test by name.
        Example:
            dbg.run("spectral", img)
        """
        if name not in self.tests:
            raise ValueError(f"Unknown test: {name}")
        return self.tests[name](*args, **kwargs)

    # ---------------------------------------------------------
    # Internal built-in test wrappers
    # ---------------------------------------------------------
    def _run_spectral(self, img, tag="spectral_ratio"):
        hf, total = high_freq_energy(img)
        ratio = hf / (total + 1e-8)
        self.log[tag].append(ratio)
        return ratio

    def _run_activations(self, tag="acts"):
        stats = {}
        for name, a in self.acts.items():
            stats[name] = {
                "min":  float(a.min()),
                "max":  float(a.max()),
                "mean": float(a.mean()),
                "std":  float(a.std())
            }
        self.log[tag].append(stats)
        return stats

    def _run_gradients(self, tag="grads"):
        stats = {}
        for name, g in self.grads.items():
            stats[name] = {
                "mean": float(g.mean()),
                "std":  float(g.std())
            }
        self.log[tag].append(stats)
        return stats

    def _run_grad_norms(self, tag="grad_norm_stats"):
        norms = gradient_norm_stats(self.model)
        summary = summarize_gradient_norms(norms)
        self.log[tag].append({"raw": norms, "summary": summary})
        return summary

    def _run_memorization(self, image, steps=200, lr=1e-3, tag="memorization_loss"):
        loss_val = can_memorize_random_target(self.model, image, steps, lr, self.device)
        self.log[tag].append(loss_val)
        return loss_val

    def _run_lr_sweep(self, loss_fn, data_fn, lrs, steps=10, tag="lr_sweep"):
        results = lr_sweep(self.model, loss_fn, data_fn, lrs, steps, self.device)
        self.log[tag] = results
        return results

    # Unified log access
    def get_log(self):
        return dict(self.log)


def save_log(log_dict, path):
    """
    Saves debugger log as JSON after converting tensors to lists.
    """
    def clean(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        return obj

    clean_log = clean(log_dict)
    with open(path, "w") as f:
        json.dump(clean_log, f, indent=2)

def plot_spectral(log_dict, key="spectral_ratio", title="Spectral Ratio Over Time"):
    """
    Plots spectral high-freq ratio vs training steps.
    """
    if key not in log_dict:
        print("No spectral data found.")
        return

    vals = log_dict[key]

    plt.figure()
    plt.plot(vals)
    plt.xlabel("Step")
    plt.ylabel("High-frequency energy ratio")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_grad_norms(log_dict, key="grad_norm_stats"):
    """
    Shows evolution of mean and std of gradient norms.
    """
    if key not in log_dict:
        print("No gradient norm data found.")
        return

    means, stds = [], []

    for entry in log_dict[key]:
        if entry["summary"] is None:
            continue
        _, mean, _, std = entry["summary"]
        means.append(mean)
        stds.append(std)

    plt.figure()
    plt.plot(means, label="mean")
    plt.plot(stds, label="std")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Gradient norm")
    plt.title("Gradient Norm Evolution")
    plt.grid(True)
    plt.show()

def plot_lr_sweep(results, title="LR Sweep"):
    """
    Plots final loss vs learning rate (log scale).
    """
    lrs = [x[0] for x in results]
    losses = [x[1] for x in results]

    plt.figure()
    plt.plot(lrs, losses, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Loss")
    plt.title(title)
    plt.grid(True)
    plt.show()

