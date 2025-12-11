import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf


# =====================================================================
# Neural Network Architecture
# =====================================================================

class MaskedConv2d(nn.Conv2d):
    """Conv2d with center pixel masked to prevent information leakage."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask = torch.ones_like(self.weight)
        kH, kW = self.weight.shape[-2:]
        mask[:, :, kH // 2, kW // 2] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight * self.mask, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class ConvBlock2d(nn.Module):
    """Convolution block with LeakyReLU activation."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, use_mask: bool = False):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        ConvClass = MaskedConv2d if use_mask else nn.Conv2d
        self.conv = ConvClass(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=padding)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class BlindSpotSpatial(nn.Module):
    """Blind-spot network backbone that never sees the center pixel."""
    
    def __init__(self, out_channels: int = 1, final_activation: Optional[nn.Module] = None):
        super().__init__()
        # Regular convolution path
        self.reg_conv1 = ConvBlock2d(1, 16, kernel_size=3, dilation=1, use_mask=False)
        self.reg_conv2 = ConvBlock2d(16, 32, kernel_size=3, dilation=1, use_mask=False)
        self.reg_conv3 = ConvBlock2d(32, 48, kernel_size=3, dilation=1, use_mask=False)
        self.reg_conv4 = ConvBlock2d(48, 64, kernel_size=3, dilation=1, use_mask=False)
        self.reg_conv5 = ConvBlock2d(64, 80, kernel_size=3, dilation=1, use_mask=False)

        # Blind-spot convolution path
        self.bsconv1 = ConvBlock2d(1, 16, kernel_size=3, dilation=1, use_mask=True)
        self.bsconv2 = ConvBlock2d(16, 32, kernel_size=3, dilation=2, use_mask=True)
        self.bsconv3 = ConvBlock2d(32, 48, kernel_size=3, dilation=3, use_mask=True)
        self.bsconv4 = ConvBlock2d(48, 64, kernel_size=3, dilation=4, use_mask=True)
        self.bsconv5 = ConvBlock2d(64, 80, kernel_size=3, dilation=5, use_mask=True)
        self.bsconv6 = ConvBlock2d(80, 96, kernel_size=3, dilation=6, use_mask=True)

        self.final = nn.Conv2d(16 + 32 + 48 + 64 + 80 + 96, out_channels, kernel_size=1)
        self.final_activation = final_activation or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular path
        enc1 = self.reg_conv1(x)
        enc2 = self.reg_conv2(enc1)
        enc3 = self.reg_conv3(enc2)
        enc4 = self.reg_conv4(enc3)
        enc5 = self.reg_conv5(enc4)

        # Blind-spot path
        bs1 = self.bsconv1(x)
        bs2 = self.bsconv2(enc1)
        bs3 = self.bsconv3(enc2)
        bs4 = self.bsconv4(enc3)
        bs5 = self.bsconv5(enc4)
        bs6 = self.bsconv6(enc5)

        out = torch.cat([bs1, bs2, bs3, bs4, bs5, bs6], dim=1)
        return self.final_activation(self.final(out))


class SpatialNetwork(nn.Module):
    """Predicts mean and total variance for denoising."""
    
    def __init__(self):
        super().__init__()
        self.mean_backbone = BlindSpotSpatial(out_channels=1, final_activation=None)
        self.var_backbone = BlindSpotSpatial(out_channels=1, final_activation=nn.Softplus())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean_backbone(x), self.var_backbone(x)


class TotalVarianceSpatialDenoiser(pl.LightningModule):
    """PyTorch Lightning module for training the spatial denoiser."""
    
    def __init__(self, learning_rate: float = 1e-3, max_epochs: int = 5):
        super().__init__()
        self.spatial_network = SpatialNetwork()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spatial_network(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        mu_x, total_variance = self(batch)
        total_variance = torch.clamp(total_variance, min=1e-8)
        log_lik = torch.nansum(torch.log(total_variance))
        log_lik += torch.nansum((batch - mu_x) ** 2 / total_variance)
        loss = log_lik / batch.numel()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# =====================================================================
# Data Processing Utilities
# =====================================================================

def detect_nonzero_bbox(img: np.ndarray, 
                       threshold_quantile: float = 0.01, 
                       min_size: int = 32) -> Tuple[int, int, int, int]:
    """
    Detect bounding box of non-zero regions in an image.
    
    Args:
        img: Input image (H, W)
        threshold_quantile: Quantile threshold to determine "non-zero" regions
        min_size: Minimum bounding box size
    
    Returns:
        Tuple of (row_min, row_max, col_min, col_max)
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    img_abs = np.abs(img)
    
    # Use quantile threshold to filter noise
    if np.any(img_abs > 0):
        threshold = np.quantile(img_abs[img_abs > 0], threshold_quantile)
    else:
        return 0, img.shape[0], 0, img.shape[1]
    
    mask = img_abs > threshold
    
    if not np.any(mask):
        return 0, img.shape[0], 0, img.shape[1]
    
    # Find boundary rows and columns
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        return 0, img.shape[0], 0, img.shape[1]
    
    row_min, row_max = row_indices[0], row_indices[-1] + 1
    col_min, col_max = col_indices[0], col_indices[-1] + 1
    
    # Ensure minimum size
    row_span = row_max - row_min
    col_span = col_max - col_min
    
    if row_span < min_size:
        center = (row_min + row_max) // 2
        row_min = max(0, center - min_size // 2)
        row_max = min(img.shape[0], row_min + min_size)
        row_min = max(0, row_max - min_size)
    
    if col_span < min_size:
        center = (col_min + col_max) // 2
        col_min = max(0, center - min_size // 2)
        col_max = min(img.shape[1], col_min + min_size)
        col_min = max(0, col_max - min_size)
    
    return row_min, row_max, col_min, col_max


def extract_component_patches(spatial_components: torch.Tensor, 
                              padding: int = 2,
                              threshold_quantile: float = 0.01,
                              min_size: int = 32) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Extract valid patch regions from spatial components.
    
    Args:
        spatial_components: Shape (num_components, H, W)
        padding: Padding around detected bounding box
        threshold_quantile: Quantile threshold for detection
        min_size: Minimum patch size
    
    Returns:
        patches: List of extracted patches
        metadata: List of metadata dicts for each patch
    """
    num_comp, H, W = spatial_components.shape
    patches = []
    metadata = []
    
    print(f"\n{'='*60}")
    print("Extracting valid patches from spatial components...")
    print(f"{'='*60}")
    
    for idx in range(num_comp):
        comp = spatial_components[idx]
        
        # Detect bounding box
        row_min, row_max, col_min, col_max = detect_nonzero_bbox(
            comp, threshold_quantile=threshold_quantile, min_size=min_size
        )
        
        # Add padding
        row_min = max(0, row_min - padding)
        row_max = min(H, row_max + padding)
        col_min = max(0, col_min - padding)
        col_max = min(W, col_max + padding)
        
        # Crop patch
        patch = comp[row_min:row_max, col_min:col_max]
        
        # Filter out invalid patches
        if patch.numel() < min_size * min_size or torch.std(patch) < 1e-6:
            continue
        
        patches.append(patch)
        metadata.append({
            'component_idx': idx,
            'bbox': (row_min, row_max, col_min, col_max),
            'original_shape': (H, W),
            'patch_shape': patch.shape
        })
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{num_comp} components...")
    
    print(f"\nExtracted {len(patches)} valid patches from {num_comp} components")
    print(f"{'='*60}\n")
    
    return patches, metadata


class SpatialComponentPatchDataset(Dataset):
    """Dataset for training on extracted spatial component patches."""
    
    def __init__(self, 
                 patches: List[torch.Tensor],
                 metadata: List[Dict],
                 target_size: Tuple[int, int] = (128, 128),
                 padding_mode: str = 'reflect'):
        """
        Args:
            patches: List of cropped patches with varying sizes
            metadata: Metadata for each patch
            target_size: Target size for uniform batching (H, W)
            padding_mode: Padding mode ('reflect', 'constant', 'replicate')
        """
        self.patches = [p.float().cpu() for p in patches]
        self.metadata = metadata
        self.target_h, self.target_w = target_size
        self.padding_mode = padding_mode
        
        # Compute normalization parameters
        self.means = []
        self.norms = []
        
        for patch in self.patches:
            mean = patch.mean()
            centered = patch - mean
            norm = torch.linalg.norm(centered)
            if norm < 1e-6:
                norm = torch.tensor(1.0)
            
            self.means.append(mean)
            self.norms.append(norm)
        
        print(f"Dataset created: {len(self.patches)} patches")
        total_memory = sum(p.element_size() * p.nelement() for p in self.patches)
        print(f"Memory footprint: {total_memory / 1024**3:.2f} GB")
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def _pad_to_target_size(self, patch: torch.Tensor) -> torch.Tensor:
        """Pad or crop patch to target size."""
        h, w = patch.shape
        
        if h >= self.target_h and w >= self.target_w:
            # Center crop if larger
            start_h = (h - self.target_h) // 2
            start_w = (w - self.target_w) // 2
            return patch[start_h:start_h+self.target_h, start_w:start_w+self.target_w]
        
        # Compute padding
        pad_h = max(0, self.target_h - h)
        pad_w = max(0, self.target_w - w)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        patch_4d = patch.unsqueeze(0).unsqueeze(0)
        padded = F.pad(
            patch_4d,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode=self.padding_mode
        )
        
        return padded.squeeze(0).squeeze(0)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = self.patches[idx]
        mean = self.means[idx]
        norm = self.norms[idx]
        
        # Normalize
        normalized = (patch - mean) / norm
        
        # Pad to target size
        padded = self._pad_to_target_size(normalized)
        
        # Add channel dimension
        return padded.unsqueeze(0)


# =====================================================================
# PMD Integration Wrapper
# =====================================================================

class PMDSpatialDenoiser(nn.Module):
    """Wrapper that adapts the trained spatial denoiser for PMD usage."""
    
    def __init__(self,
                 trained_model: TotalVarianceSpatialDenoiser,
                 noise_variance_quantile: float = 0.05,
                 padding: int = 12):
        """
        Args:
            trained_model: Trained TotalVarianceSpatialDenoiser
            noise_variance_quantile: Quantile for noise variance estimation
            padding: Padding for spatial components during denoising
        """
        super().__init__()
        self.net = trained_model.spatial_network
        self.noise_variance_quantile = noise_variance_quantile
        self._padding = padding
    
    def _estimate_noise_variance(self, spatial_basis: torch.Tensor) -> torch.Tensor:
        """Estimate noise variance for each spatial component."""
        H, W, num_comp = spatial_basis.shape
        device = spatial_basis.device
        noise_vars = torch.zeros(num_comp, device=device)
        
        with torch.no_grad():
            for idx in range(num_comp):
                comp = spatial_basis[:, :, idx]
                comp_mean = comp.mean()
                comp_centered = comp - comp_mean
                comp_norm = torch.clamp(torch.linalg.norm(comp_centered), min=1e-6)
                comp_normalized = comp_centered / comp_norm
                
                if self._padding > 0:
                    comp_input = F.pad(
                        comp_normalized.unsqueeze(0).unsqueeze(0),
                        (self._padding,) * 4,
                        mode='reflect'
                    )
                else:
                    comp_input = comp_normalized.unsqueeze(0).unsqueeze(0)
                
                _, total_var = self.net(comp_input)
                
                if self._padding > 0:
                    p = self._padding
                    total_var_center = total_var[:, :, p:-p, p:-p]
                else:
                    total_var_center = total_var
                
                noise_vars[idx] = torch.quantile(
                    total_var_center.flatten(),
                    self.noise_variance_quantile
                )
        
        return noise_vars
    
    def _denoise_single_component(self,
                                  component: torch.Tensor,
                                  component_idx: int,
                                  noise_variance: torch.Tensor) -> torch.Tensor:
        """Denoise a single spatial component using Wiener filtering."""
        comp_mean = component.mean()
        comp_centered = component - comp_mean
        comp_norm = torch.clamp(torch.linalg.norm(comp_centered), min=1e-6)
        comp_normalized = comp_centered / comp_norm
        
        if self._padding > 0:
            comp_padded = F.pad(
                comp_normalized.unsqueeze(0).unsqueeze(0),
                (self._padding,) * 4,
                mode='reflect'
            )
        else:
            comp_padded = comp_normalized.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            mu_x, total_var = self.net(comp_padded)
            
            noise_var = noise_variance[component_idx].view(1, 1, 1, 1)
            total_var_clamped = torch.clamp(total_var, min=noise_var)
            signal_var = torch.clamp(total_var_clamped - noise_var, min=0.0)
            
            # Wiener filtering weights
            weight_signal = noise_var / total_var_clamped
            weight_observation = signal_var / total_var_clamped
            
            denoised_normalized = weight_signal * mu_x + weight_observation * comp_padded
        
        if self._padding > 0:
            p = self._padding
            denoised_normalized = denoised_normalized[:, :, p:-p, p:-p]
        
        # Denormalize
        denoised = denoised_normalized.squeeze(0).squeeze(0) * comp_norm + comp_mean
        return denoised
    
    def forward(self, spatial_basis: torch.Tensor) -> torch.Tensor:
        """
        Denoise all spatial components.
        
        Args:
            spatial_basis: Tensor of shape (H, W, num_comp)
        
        Returns:
            Denoised spatial basis of same shape
        """
        H, W, num_comp = spatial_basis.shape
        noise_variance = self._estimate_noise_variance(spatial_basis)
        
        denoised_components = []
        for i in range(num_comp):
            comp = spatial_basis[:, :, i]
            denoised_comp = self._denoise_single_component(comp, i, noise_variance)
            denoised_components.append(denoised_comp)
        
        return torch.stack(denoised_components, dim=2)


# =====================================================================
# Main Training Function
# =====================================================================

def train_spatial_denoiser(
    spatial_components: torch.Tensor,
    config: Optional[Dict] = None,
    output_dir: Optional[str] = None
) -> Tuple[TotalVarianceSpatialDenoiser, Dict]:
    """
    Train a spatial denoiser on PMD spatial components.
    
    Args:
        spatial_components: Tensor of shape (num_components, H, W)
        config: Configuration dictionary with training parameters
        output_dir: Directory to save model and logs
    
    Returns:
        trained_model: Trained denoiser model
        training_info: Dictionary with training statistics
    """
    
    # Default configuration
    default_config = {
        'patch_h': 40,
        'patch_w': 40,
        'crop_padding': 2,
        'crop_threshold_quantile': 0.01,
        'min_patch_size': 32,
        'train_patch_subset': 500000,
        'batch_size': 32,
        'num_workers': 0,
        'learning_rate': 1e-4,
        'max_epochs': 5,
        'gradient_accumulation_steps': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    cfg = {**default_config, **(config or {})}
    device = cfg['device']
    
    print(f"\n{'='*60}")
    print("Starting Spatial Denoiser Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Extract patches from spatial components
    patches, metadata = extract_component_patches(
        spatial_components,
        padding=cfg['crop_padding'],
        threshold_quantile=cfg['crop_threshold_quantile'],
        min_size=cfg['min_patch_size']
    )
    
    # Print patch statistics
    patch_sizes = [p.shape for p in patches]
    print(f"\n📊 Patch Statistics:")
    print(f"  Number of patches: {len(patches)}")
    print(f"  Size range: {min(min(s) for s in patch_sizes)} - {max(max(s) for s in patch_sizes)} pixels")
    print(f"  Average size: {np.mean([p.numel() for p in patches]):.0f} pixels")
    
    # Create dataset
    dataset = SpatialComponentPatchDataset(
        patches=patches,
        metadata=metadata,
        target_size=(cfg['patch_h'], cfg['patch_w']),
        padding_mode='reflect'
    )
    
    # Use subset if specified
    if cfg['train_patch_subset'] is not None:
        subset_n = min(int(cfg['train_patch_subset']), len(dataset))
        indices = torch.randperm(len(dataset))[:subset_n].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using {subset_n} random patches for training")
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['num_workers'] > 0,
    )
    print(f"DataLoader ready with {len(train_loader)} batches")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize model
    model = TotalVarianceSpatialDenoiser(
        learning_rate=cfg['learning_rate'],
        max_epochs=cfg['max_epochs']
    )
    
    # Setup trainer
    logger = TensorBoardLogger("lightning_logs", name="spatial_denoiser")
    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        log_every_n_steps=10,
        devices=1,
        accelerator="gpu" if device == "cuda" else "cpu",
        precision="16-mixed" if device == "cuda" else 32,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg['gradient_accumulation_steps'],
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    
    print(f"\nStarting training...")
    print(f"Effective batch size: {cfg['batch_size'] * cfg['gradient_accumulation_steps']}")
    
    # Train
    trainer.fit(model, train_loader)
    print("Training complete!")
    
    # Report peak memory
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    # Save model
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_path = out_path / "spatial_denoiser_state_dict.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    
    # Test blind-spot property
    model.to(device)
    leak_diff = test_blindspot_leakage(model.spatial_network, device=device)
    print(f"\nBlindspot leakage test: {leak_diff:.2e}")
    if leak_diff > 1e-6:
        print("⚠️  WARNING: Potential information leakage detected!")
    else:
        print("✓ Blindspot property verified")
    
    # Compile training info
    training_info = {
        'num_patches': len(patches),
        'num_components': spatial_components.shape[0],
        'patch_size_range': (min(min(s) for s in patch_sizes), max(max(s) for s in patch_sizes)),
        'training_patches': len(dataset),
        'blindspot_leakage': leak_diff,
        'config': cfg
    }
    
    return model, training_info


def create_pmd_denoiser(trained_model: TotalVarianceSpatialDenoiser,
                       noise_variance_quantile: float = 0.7,
                       padding: int = 12,
                       device: str = 'cuda') -> PMDSpatialDenoiser:
    """
    Create a PMD-compatible spatial denoiser from a trained model.
    
    Args:
        trained_model: Trained TotalVarianceSpatialDenoiser
        noise_variance_quantile: Quantile for noise estimation (0.0-1.0)
        padding: Padding to use during inference
        device: Device to place the model on
    
    Returns:
        PMD-compatible denoiser ready for use
    """
    pmd_denoiser = PMDSpatialDenoiser(
        trained_model=trained_model,
        noise_variance_quantile=noise_variance_quantile,
        padding=padding
    )
    pmd_denoiser.to(device)
    pmd_denoiser.eval()
    
    print(f"\n✓ PMD spatial denoiser created")
    print(f"  Noise variance quantile: {noise_variance_quantile}")
    print(f"  Padding: {padding}")
    print(f"  Device: {device}")
    
    return pmd_denoiser


def test_blindspot_leakage(model: nn.Module, H: int = 64, W: int = 64, 
                          device: str = 'cuda') -> float:
    """Test if the blind-spot property holds."""
    model.eval()
    x = torch.randn(1, 1, H, W, device=device)
    
    with torch.no_grad():
        out1, _ = model(x)
        
        x_perturbed = x.clone()
        x_perturbed[0, 0, H//2, W//2] += 10.0
        out2, _ = model(x_perturbed)
        
        diff = torch.abs(out1[0, 0, H//2, W//2] - out2[0, 0, H//2, W//2])
    
    return diff.item()