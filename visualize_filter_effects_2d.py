"""
Visualize ramp filter effects on actual 2D projection data in frequency domain.
"""

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse
from pathlib import Path


def compute_window(n, window_type):
    """Compute window function for ramp filter in frequency domain."""
    window_type = window_type.lower()
    
    if window_type in ['ram-lak', 'none']:
        return np.ones(n, dtype=np.float32)
    
    freqs = np.fft.fftfreq(n)
    omega = np.abs(freqs) * 2
    
    if window_type == 'shepp-logan':
        window = np.sinc(omega / 2)
    elif window_type == 'cosine':
        window = np.cos(omega * np.pi / 2)
    elif window_type == 'hamming':
        window = 0.54 + 0.46 * np.cos(omega * np.pi)
    elif window_type == 'hann':
        window = (1 + np.cos(omega * np.pi)) / 2
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    return window.astype(np.float32)


def apply_ramp_filter_2d(projection_2d, du, window_type):
    """Apply ramp filter to a single 2D projection in frequency domain."""
    nu, nv = projection_2d.shape
    
    # FFT along u-axis
    F = np.fft.fft(projection_2d, axis=0)
    
    # Build ramp filter
    freqs = np.fft.fftfreq(nu, d=du)
    ramp = np.abs(freqs)
    w = compute_window(nu, window_type)
    ramp_filter = ramp * w * (2.0 * du)
    
    # Apply filter (broadcast across v dimension)
    F_filtered = F * ramp_filter[:, np.newaxis]
    
    return F_filtered


def load_projections_mhd(mhd_path: str):
    """Load projections from MHD file."""
    print(f"Loading projections from: {mhd_path}")
    img = sitk.ReadImage(mhd_path)
    arr = sitk.GetArrayFromImage(img)
    
    # Transpose to (n_proj, nu, nv)
    if arr.ndim == 3:
        arr = np.transpose(arr, (0, 2, 1))
    
    print(f"  Shape: {arr.shape}")
    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Visualize ramp filter effects on projections")
    parser.add_argument("--projections", type=str, default="projections.mhd",
                       help="Path to projections MHD file")
    parser.add_argument("--du", type=float, default=0.4,
                       help="Detector pixel size in mm")
    parser.add_argument("--angle-idx", type=int, default=0,
                       help="Which projection angle to visualize (default: 0)")
    parser.add_argument("--output", type=str, default="filter_effects_frequency_domain.png",
                       help="Output filename")
    
    args = parser.parse_args()
    
    # Load projections
    projections = load_projections_mhd(args.projections)
    n_proj, nu, nv = projections.shape
    
    # Select a projection
    angle_idx = min(args.angle_idx, n_proj - 1)
    projection = projections[angle_idx]
    
    print(f"\nUsing projection angle {angle_idx} of {n_proj}")
    print(f"Projection shape: {projection.shape}")
    
    # Apply filters
    window_types = ['none', 'shepp-logan', 'cosine', 'hamming', 'hann']
    filtered_projections = {}
    
    for wtype in window_types:
        print(f"  Processing {wtype.upper()}...", end=" ", flush=True)
        F_filtered = apply_ramp_filter_2d(projection, args.du, wtype)
        filtered_projections[wtype] = F_filtered
        print("✓")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Ramp Filter Effects in Frequency Domain (Projection Angle {angle_idx})', 
                 fontsize=16, fontweight='bold')
    
    # Compute vmin, vmax for consistent scaling
    all_spectra = [np.abs(F) for F in filtered_projections.values()]
    vmin_log = -3  # log10 scale
    vmax_log = 2
    
    axes_flat = axes.flatten()
    
    for idx, (wtype, F_filtered) in enumerate(filtered_projections.items()):
        ax = axes_flat[idx]
        
        # Compute magnitude spectrum (log scale for better visibility)
        magnitude = np.abs(F_filtered)
        magnitude_log = np.log10(magnitude + 1e-10)
        
        # Display (shift zero frequency to center for better visualization)
        magnitude_shifted = np.fft.fftshift(magnitude_log, axes=0)
        
        im = ax.imshow(magnitude_shifted.T, cmap='hot', aspect='auto',
                       vmin=vmin_log, vmax=vmax_log, origin='lower')
        
        ax.set_title(f'{wtype.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Spatial Frequency (u)', fontsize=10)
        ax.set_ylabel('Detector Row (v)', fontsize=10)
        
        plt.colorbar(im, ax=ax, label='log10(Magnitude)')
    
    # Hide the last unused subplot
    axes_flat[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {args.output}")
    
    # Print statistics
    print("\n" + "="*80)
    print("FREQUENCY DOMAIN STATISTICS")
    print("="*80)
    print(f"{'Filter':<15} {'Mean Magnitude':<20} {'Std Magnitude':<20} {'Max Magnitude':<20}")
    print("-"*80)
    
    for wtype, F_filtered in filtered_projections.items():
        mag = np.abs(F_filtered)
        mean_mag = np.mean(mag)
        std_mag = np.std(mag)
        max_mag = np.max(mag)
        print(f"{wtype.upper():<15} {mean_mag:<20.4e} {std_mag:<20.4e} {max_mag:<20.4e}")
    
    print("="*80)
    print("\nInterpretation:")
    print("  • Brighter regions = higher frequency content")
    print("  • Ram-Lak (None): Preserves all high frequencies → noisier")
    print("  • Shepp-Logan: Good balance of frequencies")
    print("  • Cosine/Hamming/Hann: Progressively suppress high frequencies → smoother")


if __name__ == "__main__":
    main()
