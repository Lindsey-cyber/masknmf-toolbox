import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalShift(nn.Module):
    """Shift a feature map by `shift` pixels along a given direction.

    Paired with subsequent same-padding 3x3 conv layers, this guarantees
    the receptive field of every output position strictly excludes its
    own input pixel (blind-spot property).
    """

    def __init__(self, direction: str, shift: int):
        super().__init__()
        assert direction in ('up', 'down', 'left', 'right')
        assert shift >= 2
        self.direction = direction
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); F.pad order is (left, right, top, bottom)
        s = self.shift
        if self.direction == 'up':
            return F.pad(x, (0, 0, s, 0))[:, :, :-s, :]
        elif self.direction == 'down':
            return F.pad(x, (0, 0, 0, s))[:, :, s:, :]
        elif self.direction == 'left':
            return F.pad(x, (s, 0, 0, 0))[:, :, :, :-s]
        elif self.direction == 'right':
            return F.pad(x, (0, s, 0, 0))[:, :, :, s:]


class DirectionalBranch(nn.Module):
    """Shift + n stacked 3x3 conv layers; receptive field is a strict half-plane."""

    def __init__(self, in_ch: int, hidden_ch: int, direction: str, n_layers: int = 3):
        super().__init__()
        # shift >= n_layers + 1 ensures the RF strictly excludes the center row/col
        self.shift = DirectionalShift(direction, shift=n_layers + 1)
        layers = []
        c_in = in_ch
        for _ in range(n_layers):
            layers.append(nn.Conv2d(c_in, hidden_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = hidden_ch
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(self.shift(x))


class BlindSpotCNN(nn.Module):
    """4-branch blind-spot 2D CNN spatial denoiser.

    Four directional branches (up/down/left/right) run in parallel; their
    features are concatenated and fused by 1x1 convs into a scalar prediction.
    With hidden_ch=32 and n_conv_per_branch=3 on a 32x32 block the per-branch
    RF is ~7 pixels and the total parameter count is ~100k.
    """

    def __init__(self, hidden_ch: int = 32, n_conv_per_branch: int = 3):
        super().__init__()
        self.up    = DirectionalBranch(1, hidden_ch, 'up',    n_conv_per_branch)
        self.down  = DirectionalBranch(1, hidden_ch, 'down',  n_conv_per_branch)
        self.left  = DirectionalBranch(1, hidden_ch, 'left',  n_conv_per_branch)
        self.right = DirectionalBranch(1, hidden_ch, 'right', n_conv_per_branch)

        fused_ch = 4 * hidden_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        fu = self.up(x)
        fd = self.down(x)
        fl = self.left(x)
        fr = self.right(x)
        f = torch.cat([fu, fd, fl, fr], dim=1)
        return self.fuse(f)

class SpatialDenoiserWrapper:
    """Wrap a trained BlindSpotCNN as a callable spatial denoiser for masknmf.

    masknmf's spatial_denoiser slot expects an input of shape (H, W, N).
    Internally, this wrapper:
      - converts to (N, 1, H, W) for the CNN
      - per-component standardizes input to match training distribution
      - aligns the sign of the top-k denoised components to the originals
      - hard-gates: only the top-k strongest components are kept; the rest
        are zeroed out
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        standardize: bool = True,
        keep_k: int = 3,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.standardize = standardize
        self.keep_k = keep_k

    def to(self, device: str) -> 'SpatialDenoiserWrapper':
        self.device = device
        self.model = self.model.to(device).eval()
        return self

    @torch.no_grad()
    def __call__(self, spatial_basis: torch.Tensor) -> torch.Tensor:
        # spatial_basis: (H, W, N)
        H, W, N = spatial_basis.shape
        if N == 0:
            return spatial_basis

        x = spatial_basis.permute(2, 0, 1).unsqueeze(1).to(self.device)  # (N, 1, H, W)

        if self.standardize:
            std = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            x_in = x / std
        else:
            x_in = x
            std = 1.0

        y = self.model(x_in)

        if self.standardize:
            y = y * std

        keep_k = min(self.keep_k, N)

        # Sign-align the top keep_k components to the original input
        y_aligned = y.clone()
        for i in range(keep_k):
            ref = x[i:i + 1]
            den = y[i:i + 1]
            corr = torch.sum(ref * den)
            sign = torch.sign(corr)
            if sign == 0:
                sign = torch.tensor(1.0, device=den.device, dtype=den.dtype)
            y_aligned[i:i + 1] = den * sign

        # Hard gate: keep only the first keep_k
        out_x = torch.zeros_like(x)
        out_x[:keep_k] = y_aligned[:keep_k]

        out = out_x.squeeze(1).permute(1, 2, 0).to(
            dtype=spatial_basis.dtype, device=spatial_basis.device
        )
        return out


def _augment(batch: torch.Tensor) -> torch.Tensor:
    """Random flips / 90 deg rotations / sign flip. batch: (B, 1, H, W)."""
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[2])
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[3])
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        batch = torch.rot90(batch, k=k, dims=[2, 3])
    if torch.rand(1).item() < 0.5:
        batch = -batch
    return batch


def collect_spatial_bases_from_blocks(block_basis_list: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-block spatial bases into a single (N_total, H, W) tensor.

    Args:
        block_basis_list: list of (H, W, n_k) tensors, one per PMD block.

    Returns:
        Tensor of shape (N_total, H, W).
    """
    if len(block_basis_list) == 0:
        raise ValueError("No blocks provided.")
    H, W = block_basis_list[0].shape[:2]
    for b in block_basis_list:
        if b.shape[:2] != (H, W):
            raise ValueError(
                f"Inconsistent block size: got {b.shape[:2]}, expected {(H, W)}"
            )
    all_basis = torch.cat([b.detach().cpu() for b in block_basis_list], dim=2)  # (H, W, N_total)
    return all_basis.permute(2, 0, 1).contiguous()


def extract_component_patches(
    pmd_result,
    d1: int,
    d2: int,
    block_h: int = 32,
    block_w: int = 32,
) -> torch.Tensor:
    """Crop a (block_h, block_w) patch centered on each PMD component's centroid.

    Args:
        pmd_result: PMDArray-like object exposing a spatial basis attribute
            (`u`, `u_sparse`, or `spatial_basis`).
        d1, d2: full FOV spatial dimensions.
        block_h, block_w: patch size.

    Returns:
        Tensor of shape (N, block_h, block_w).
    """
    u = None
    for attr in ("u", "u_sparse", "spatial_basis"):
        if hasattr(pmd_result, attr):
            u = getattr(pmd_result, attr)
            break
    if u is None:
        raise AttributeError(
            "Could not find spatial basis on PMD result "
            "(expected attribute `u`, `u_sparse`, or `spatial_basis`)."
        )

    if hasattr(u, "to_dense"):
        u_dense = u.to_dense()
    else:
        u_dense = u
    if isinstance(u_dense, torch.Tensor):
        u_dense = u_dense.detach().cpu().numpy()
    u_dense = np.asarray(u_dense, dtype=np.float32)  # (P, N)
    N = u_dense.shape[1]
    U = u_dense.reshape(d1, d2, N)

    patches = np.zeros((N, block_h, block_w), dtype=np.float32)
    i_idx = np.arange(d1, dtype=np.float32)
    j_idx = np.arange(d2, dtype=np.float32)
    for k in range(N):
        comp = np.abs(U[:, :, k])
        w = comp.sum()
        if w < 1e-8:
            ci, cj = d1 // 2, d2 // 2
        else:
            ci = int((comp.sum(axis=1) * i_idx).sum() / w)
            cj = int((comp.sum(axis=0) * j_idx).sum() / w)
        i0 = int(max(0, min(ci - block_h // 2, d1 - block_h)))
        j0 = int(max(0, min(cj - block_w // 2, d2 - block_w)))
        patches[k] = U[i0:i0 + block_h, j0:j0 + block_w, k]
    return torch.from_numpy(patches)

def train_blindspot_denoiser(
    basis_tensor: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_ch: int = 32,
    n_conv_per_branch: int = 3,
    device: str = 'cuda',
    val_fraction: float = 0.1,
    augment: bool = True,
    verbose: bool = True,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train a BlindSpotCNN.

    Args:
        basis_tensor: (N_total, H, W) stack of spatial basis vectors.

    Returns:
        (best_model, train_losses, val_losses)
    """
    assert basis_tensor.ndim == 3, f"Expected (N, H, W), got {basis_tensor.shape}"
    N_total, H, W = basis_tensor.shape

    train_losses: List[float] = []
    val_losses: List[float] = []

    # Per-component standardization so the network sees unit-variance input
    stds = basis_tensor.reshape(N_total, -1).std(dim=1).clamp(min=1e-8)
    data = basis_tensor / stds.view(N_total, 1, 1)
    data = data.unsqueeze(1)  # (N_total, 1, H, W)

    perm = torch.randperm(N_total)
    n_val = max(1, int(N_total * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_data = data[train_idx].to(device)
    val_data = data[val_idx].to(device)

    model = BlindSpotCNN(hidden_ch=hidden_ch, n_conv_per_branch=n_conv_per_branch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    n_train = train_data.shape[0]

    for epoch in range(epochs):
        model.train()
        perm_e = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm_e[i:i + batch_size]
            batch = train_data[idx]
            if augment:
                batch = _augment(batch)
            pred = model(batch)
            # Blind-spot property => MSE is safe over all positions, no mask needed
            loss = F.mse_loss(pred, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(val_data)
            val_loss = F.mse_loss(val_pred, val_data).item()

        avg_train_loss = total_loss / n_batches
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"[epoch {epoch:3d}] train {avg_train_loss:.5f} "
                  f"val {val_loss:.5f} (best {best_val:.5f})")

    model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses

def build_trained_spatial_denoiser(
    block_basis_list: List[torch.Tensor],
    device: str = 'cuda',
    epochs: int = 100,
    batch_size: int = 64,
    verbose: bool = True,
) -> SpatialDenoiserWrapper:
    """Collect block bases -> train BlindSpotCNN -> verify -> wrap for masknmf.

    The returned wrapper can be passed directly as `spatial_denoiser` to
    `pmd_decomposition` for a subsequent round.
    """
    if verbose:
        print(f"Collecting {len(block_basis_list)} blocks of spatial bases...")
    basis_tensor = collect_spatial_bases_from_blocks(block_basis_list)
    N, H, W = basis_tensor.shape
    if verbose:
        print(f"Total components: {N}, block size: {H}x{W}")

    if verbose:
        print("Training blind-spot CNN...")
    model, _, _ = train_blindspot_denoiser(
        basis_tensor,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    if verbose:
        print("Verifying blind-spot property...")
    ok = verify_blindspot(model, H=H, W=W, device=device)
    if not ok:
        raise RuntimeError("Blind-spot verification FAILED. Architecture has a bug.")
    if verbose:
        print("  OK: blind-spot property verified.")

    wrapper = SpatialDenoiserWrapper(model, device=device, standardize=True)

    if verbose:
        print("Checking roughness stability on pure Gaussian noise...")
    stats = roughness_stability_check(wrapper, H=H, W=W, device=device)
    if verbose:
        print(f"  mean output std: {stats['mean_output_std']:.3f}")
        print(f"  CV across trials: {stats['coefficient_of_variation']:.3f}")
        if stats['coefficient_of_variation'] > 0.3:
            print("  WARNING: high variance across noise trials. "
                  "threshold_heuristic may produce unstable roughness cutoffs.")

    return wrapper

def verify_blindspot(
    model: nn.Module,
    H: int = 32,
    W: int = 32,
    device: str = 'cuda',
    n_trials: int = 5,
    atol: float = 1e-5,
) -> bool:
    """Perturb a single input pixel; output at the same position must not change."""
    model = model.to(device).eval()
    x = torch.randn(1, 1, H, W, device=device)
    with torch.no_grad():
        y0 = model(x)
    ok = True
    for _ in range(n_trials):
        i = int(torch.randint(0, H, (1,)).item())
        j = int(torch.randint(0, W, (1,)).item())
        x2 = x.clone()
        x2[0, 0, i, j] += 10.0
        with torch.no_grad():
            y1 = model(x2)
        diff = (y1[0, 0, i, j] - y0[0, 0, i, j]).abs().item()
        if diff > atol:
            print(f"  BLIND-SPOT VIOLATION at ({i},{j}): diff = {diff:.2e}")
            ok = False
    return ok


def roughness_stability_check(
    denoiser: SpatialDenoiserWrapper,
    H: int = 32,
    W: int = 32,
    n_components: int = 1,
    n_trials: int = 100,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Statistics of denoiser output std on pure Gaussian noise inputs.

    masknmf's `threshold_heuristic` calibrates spatial-roughness cutoffs on
    Gaussian noise; an unstable denoiser will produce unreliable thresholds.
    """
    outs_std: List[float] = []
    for _ in range(n_trials):
        noise = torch.randn(H, W, n_components, device=device)
        with torch.no_grad():
            out = denoiser(noise)
        outs_std.append(out.std().item())
    arr = np.asarray(outs_std)
    return {
        'mean_output_std': float(arr.mean()),
        'std_of_output_std': float(arr.std()),
        'coefficient_of_variation': float(arr.std() / (arr.mean() + 1e-8)),
    }
