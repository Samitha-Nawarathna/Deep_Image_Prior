import torch
import torch.fft as fft
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import copy
import functools
from tqdm import tqdm, trange

# ============================================================
# 1) HELPER FUNCTIONS (Calculations, Plotting & Saving)
# ============================================================

def save_log(log_dict, path):
    """Saves dictionary as JSON, handling Tensor/Numpy conversion."""
    def clean(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        if isinstance(obj, (np.generic, np.number)):
            return obj.item()
        return obj

    clean_log = clean(log_dict)
    with open(path, "w") as f:
        json.dump(clean_log, f, indent=2)

# --- Plotting/Saving Helpers (Used by Result API) ---

def plot_spectral_helper(data, title):
    # data: list of float ratios
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel("Step")
    plt.ylabel("High-frequency energy ratio")
    plt.title(title)
    plt.grid(True, alpha=0.3)

def plot_grad_norms_helper(data, title):
    # data: tuple (list_of_means, list_of_stds)
    means, stds = data
    plt.figure(figsize=(10, 6))
    plt.plot(means, label="Mean Gradient Norm")
    plt.fill_between(range(len(means)), 
                     np.array(means) - np.array(stds), 
                     np.array(means) + np.array(stds), 
                     alpha=0.2, label="Std Dev")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_lr_sweep_helper(data, title):
    # data: tuple (lrs, losses)
    lrs, losses = data
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)

def plot_activations_helper(data, title):
    # data: dict {layer_name: {min, max, mean, std}}
    # Plots mean activation per layer
    names = list(data.keys())
    means = [d['mean'] for d in data.values()]
    stds = [d['std'] for d in data.values()]
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(names))
    plt.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5)
    plt.xticks(x_pos, names, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()

def save_image_helper(tensor, path):
    """
    Saves a tensor as an image.
    Expects tensor shape: (C, H, W) or (H, W).
    """
    if tensor.dim() == 3:
        # CHW -> HWC
        tensor = tensor.permute(1, 2, 0)
    
    # Normalize to 0-1 if outside range
    arr = tensor.detach().cpu().numpy()
    if arr.min() < 0 or arr.max() > 1:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1) # Grayscale

    plt.imsave(path, arr, cmap='gray' if arr.ndim == 2 else None)

def save_weight_video_helper(weights_history, title, save_path):
    """
    Creates an animation of weights evolving over time.
    weights_history: list of weight tensors/arrays.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.axis('off')

    # Prepare first frame
    w0 = weights_history[0]
    if isinstance(w0, torch.Tensor):
        w0 = w0.detach().cpu().numpy()
    
    # Handle 4D Conv weights: Flatten or Tile
    def flatten_weights(weight):
        if weight.ndim == 4: # (Out, In, H, W) -> Tile into 2D grid
            out_c, in_c, h, w = weight.shape
            # Simple approach: take average across input channels, arrange output channels in grid
            weight = np.mean(weight, axis=1) # (Out, H, W)
            grid_size = int(np.ceil(np.sqrt(out_c)))
            # Pad to square
            padded = np.zeros((grid_size**2, h, w))
            padded[:out_c] = weight
            # Reshape to grid
            weight = padded.reshape(grid_size, grid_size, h, w).transpose(0, 2, 1, 3).reshape(grid_size*h, grid_size*w)
        return weight

    w0_flat = flatten_weights(w0)
    im = ax.imshow(w0_flat, cmap='viridis', animated=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame_idx):
        w = weights_history[frame_idx]
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
        w_flat = flatten_weights(w)
        
        # Normalize heatmap color range per frame or global? 
        # Per frame allows seeing relative patterns better.
        im.set_array(w_flat)
        im.set_clim(vmin=w_flat.min(), vmax=w_flat.max())
        ax.set_title(f"{title} - Step {frame_idx}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(weights_history), blit=True)
    
    # Save as GIF
    ani.save(save_path, writer='pillow', fps=5)
    plt.close(fig)


# ============================================================
# 2) CALCULATION KERNELS
# ============================================================

def high_freq_energy(img: torch.Tensor, cutoff_ratio=0.3):
    if img.dim() == 3:
        img = img.mean(0)
    # Explicit dims for robustness against 1D/3D inputs
    F = fft.fftshift(fft.fft2(img, dim=(-2, -1)))
    mag = torch.abs(F)
    H, W = mag.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    r = torch.sqrt((yy - H/2)**2 + (xx - W/2)**2)
    cutoff = cutoff_ratio * min(H, W) / 2
    mask = r > cutoff
    hf = mag[mask].sum().item()
    total = mag.sum().item()
    return hf, total

def attach_activation_gradient_hooks(model):
    acts, grads = {}, {}
    def fwd_hook(name):
        def hook(m, inp, out):
            acts[name] = out.detach().cpu()
        return hook
    def bwd_hook(name):
        def hook(m, grad_input, grad_output):
            if grad_output[0] is not None:
                grads[name] = grad_output[0].detach().cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d)):
            module.register_forward_hook(fwd_hook(name))
            module.register_full_backward_hook(bwd_hook(name))
    return acts, grads

def gradient_norm_stats(model):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms.append((name, p.grad.data.norm().item()))
    return norms

def summarize_gradient_norms(norms):
    if not norms: return None
    vals = np.array([v for _, v in norms])
    return vals.min(), vals.mean(), vals.max(), vals.std()

def can_memorize_random_target(model, image, steps=200, lr=1e-3, device='cpu'):
    model = model.to(device)
    inp = image.to(device).detach()
    target = torch.randn_like(inp).to(device) # I2I: Target shape == Input shape
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    loss_curve = []
    final_out = None
    
    for _ in trange(steps, desc="I2I Memorization", leave=False):
        opt.zero_grad()
        out = model(inp)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        loss_curve.append(loss.item())
        final_out = out.detach()

    return loss.item(), loss_curve, final_out, target

def lr_sweep(model, loss_fn, data_fn, lrs, steps=10, device='cpu'):
    results = []
    for lr in tqdm(lrs, desc="LR Sweep", leave=False):
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


# ============================================================
# 3) DECORATOR & DEBUGGER CLASS
# ============================================================

def log_results_on_run(func):
    """
    Decorator for Debugger.run(). Checks for Result API in return value
    and handles artifact saving (JSON, Plots, Images, Videos).
    """
    @functools.wraps(func)
    def wrapper(self, name, *args, **kwargs):
        # 1. Check control flag (default to self.log_enabled)
        log_run = kwargs.pop('log_run', self.log_enabled)

        # 2. Execute Test
        stats = func(self, name, *args, **kwargs)

        # 3. Handle Logging
        if not log_run or not isinstance(stats, dict):
            return stats
        
        # Check Result API presence
        valid_keys = ['json_summary', 'plot_artifacts', 'image_artifacts', 'video_artifacts']
        if not any(k in stats for k in valid_keys):
            return stats

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(self.log_dir, f"{name}_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)
            
            # a) JSON
            if 'json_summary' in stats:
                save_log(stats['json_summary'], os.path.join(run_dir, "summary.json"))
            
            # b) Plots (Figures)
            if 'plot_artifacts' in stats:
                for key, artifact in stats['plot_artifacts'].items():
                    if 'plot_fn' in artifact and 'data' in artifact:
                        artifact['plot_fn'](artifact['data'], title=f"{name}: {key}")
                        plt.savefig(os.path.join(run_dir, f"{key}.png"))
                        plt.close()
            
            # c) Images (Raw Tensors)
            if 'image_artifacts' in stats:
                for key, tensor in stats['image_artifacts'].items():
                    # If dict format for flexibility, extract data
                    if isinstance(tensor, dict) and 'data' in tensor:
                        tensor = tensor['data']
                    
                    if isinstance(tensor, torch.Tensor):
                        save_path = os.path.join(run_dir, f"{key}.png")
                        save_image_helper(tensor, save_path)
            
            # d) Videos (Animations)
            if 'video_artifacts' in stats:
                for key, artifact in stats['video_artifacts'].items():
                    if 'plot_fn' in artifact and 'data' in artifact:
                        save_path = os.path.join(run_dir, f"{key}.gif")
                        print(f"Generating video for {key}...")
                        artifact['plot_fn'](artifact['data'], title=f"{name}: {key}", save_path=save_path)

        except Exception as e:
            print(f"[Debugger] Error logging '{name}': {e}")
            import traceback
            traceback.print_exc()

        return stats
    return wrapper


class Debugger:
    def __init__(self, model, device='cpu', log_dir=None):
        self.model = model
        self.device = device
        
        # Internal storage
        self.log = defaultdict(list)
        
        # Setup Logging
        self.log_dir = log_dir
        self.log_enabled = log_dir is not None
        if self.log_enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[Debugger] Logging enabled. Root: {self.log_dir}")

        # Hooks
        self.acts, self.grads = attach_activation_gradient_hooks(self.model)

        # Test Registry
        self.tests = {
            "spectral": self._run_spectral,
            "activations": self._run_activations,
            "gradients": self._run_gradients,
            "grad_norms": self._run_grad_norms,
            "memorization": self._run_memorization,
            "lr_sweep": self._run_lr_sweep,
            "weight_saturation": self._run_weight_saturation # New method
        }

    def add_test(self, name, fn):
        self.tests[name] = fn

    @log_results_on_run
    def run(self, name, *args, **kwargs):
        if name not in self.tests:
            raise ValueError(f"Unknown test: {name}")
        return self.tests[name](*args, **kwargs)

    # --- Internal Test Implementations ---

    def _run_spectral(self, img, real_img=None, tag="spectral_ratio"):
        # Assumes img is generated output
        hf, total = high_freq_energy(img)
        ratio = hf / (total + 1e-8)
        
        self.log[tag].append(ratio)
        
        result = {
            "json_summary": {"current_ratio": ratio, "step": len(self.log[tag])},
            "plot_artifacts": {
                "history": {
                    "data": self.log[tag],
                    "plot_fn": plot_spectral_helper
                }
            },
            "image_artifacts": {
                "generated_output": img
            }
        }
        
        # Optionally save real image if provided
        if real_img is not None:
            result["image_artifacts"]["real_output"] = real_img
            
        return result

    def _run_grad_norms(self, tag="grad_norm_stats"):
        norms = gradient_norm_stats(self.model)
        summary = summarize_gradient_norms(norms)
        
        if summary is not None:
            self.log[tag].append({"summary": summary})
        
        history_means = [x['summary'][1] for x in self.log[tag]]
        history_stds = [x['summary'][3] for x in self.log[tag]]

        return {
            "json_summary": {"current_norms": norms, "current_stats": summary},
            "plot_artifacts": {
                "history": {
                    "data": (history_means, history_stds),
                    "plot_fn": plot_grad_norms_helper
                }
            }
        }

    def _run_activations(self, tag="acts"):
        stats = {}
        for name, a in self.acts.items():
            if a.numel() == 0: continue
            stats[name] = {
                "min": float(a.min()), "max": float(a.max()),
                "mean": float(a.mean()), "std": float(a.std()) if a.numel() > 1 else 0.0
            }
        
        return {
            "json_summary": stats,
            "plot_artifacts": {
                "layer_stats": {
                    "data": stats,
                    "plot_fn": plot_activations_helper
                }
            }
        }

    def _run_gradients(self, tag="grads"):
        stats = {}
        for name, g in self.grads.items():
            if g.numel() == 0: continue
            stats[name] = {
                "mean": float(g.mean()),
                "std": float(g.std()) if g.numel() > 1 else 0.0
            }
        return {"json_summary": stats}

    def _run_memorization(self, image, steps=200, lr=1e-3):
        # Updated to capture output and target images
        final_loss, loss_curve, final_out, target = can_memorize_random_target(self.model, image, steps, lr, self.device)
        return {
            "json_summary": {"final_loss": final_loss, "loss_curve": loss_curve},
            "plot_artifacts": {
                "convergence": {
                    "data": loss_curve,
                    "plot_fn": lambda d, title: (plt.figure(), plt.plot(d), plt.title(title), plt.ylabel("MSE Loss"), plt.xlabel("Step"))
                }
            },
            "image_artifacts": {
                "real_target": target[0], # Save 1st sample from batch
                "generated_final": final_out[0]
            }
        }

    def _run_lr_sweep(self, loss_fn, data_fn, lrs, steps=10):
        results = lr_sweep(self.model, loss_fn, data_fn, lrs, steps, self.device)
        lrs_val = [r[0] for r in results]
        loss_val = [r[1] for r in results]
        return {
            "json_summary": {"results": results},
            "plot_artifacts": {
                "sweep_curve": {
                    "data": (lrs_val, loss_val),
                    "plot_fn": plot_lr_sweep_helper
                }
            }
        }

    def _run_weight_saturation(self, tag="weight_hist"):
        """
        Captures weight matrices for heatmap visualization over time.
        Must be called repeatedly during training.
        """
        # 1. Capture current weights
        current_weights = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                current_weights[name] = p.detach().cpu().clone()
        
        self.log[tag].append(current_weights)

        # 2. Structure data for video generation (re-organize by layer)
        # We need a dict of lists: {layer_name: [step1, step2, ...]}
        history_by_layer = defaultdict(list)
        for step_weights in self.log[tag]:
            for name, w in step_weights.items():
                history_by_layer[name].append(w)

        # 3. Create Video Artifacts
        video_artifacts = {}
        # Only generate artifacts if we have history
        if len(self.log[tag]) > 1:
            for name, w_list in history_by_layer.items():
                # Filter out very small vectors (biases) if needed, or visualize them as strips
                if w_list[0].ndim >= 2 or w_list[0].numel() > 100:
                   video_artifacts[f"saturation_{name}"] = {
                       "data": w_list,
                       "plot_fn": save_weight_video_helper
                   }

        return {
            "json_summary": {"steps_captured": len(self.log[tag])},
            "video_artifacts": video_artifacts
        }

    def get_log(self):
        return dict(self.log)


# ============================================================
# 4) TEST ENTRY POINT
# ============================================================

class SimpleI2INet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.final = torch.nn.Conv2d(32, channels, 1)
    def forward(self, x):
        return self.final(torch.relu(self.conv2(torch.relu(self.conv1(x)))))

def get_i2i_data(batch_size=4, channels=3, size=32):
    return torch.randn(batch_size, channels, size, size), torch.randn(batch_size, channels, size, size)

if __name__ == '__main__':
    print("--- 🤖 Starting Refactored Debugger Test ---")
    
    # 1. Setup with Logging Directory
    LOG_DIR = "debugger_test_output_v2"
    MODEL = SimpleI2INet(3)
    debugger = Debugger(MODEL, log_dir=LOG_DIR)
    
    # 2. Simulation Loop (Spectral & Weight Saturation)
    print(f"\n[TEST 1] Training Loop (10 steps) - Tracking Spectral & Weights...")
    for step in range(1, 11):
        inp, tgt = get_i2i_data(1)
        out = MODEL(inp)
        loss = torch.nn.MSELoss()(out, tgt)
        loss.backward()
        
        # Log Logic: Only save artifacts on the last step
        is_last = (step == 10)
        
        # Pass both Generated and Real Output to spectral test
        debugger.run("spectral", img=out[0], real_img=tgt[0], log_run=is_last) 
        
        # Track weights for video generation
        debugger.run("weight_saturation", log_run=is_last)
        
        MODEL.zero_grad()
    
    print(f"  Check '{LOG_DIR}' for spectral output images and weight saturation videos.")

    # 3. One-Shot Tests
    print(f"\n[TEST 2] Running Memorization (Auto-logging)...")
    inp, _ = get_i2i_data(1)
    debugger.run("memorization", image=inp, steps=20)

    print(f"\n--- Done. Verify contents of '{LOG_DIR}' ---")