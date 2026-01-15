"""
Visualize the different ramp filter responses.
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_window(n, window_type):
    """Compute window function for ramp filter in frequency domain."""
    window_type = window_type.lower()
    
    if window_type in ['ram-lak', 'none']:
        return np.ones(n, dtype=np.float32)
    
    # Create normalized frequency array [0, 1]
    freqs = np.fft.fftfreq(n)
    omega = np.abs(freqs) * 2  # Normalized to [0, 1] at Nyquist
    
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


# Create frequency domain
nu = 512
du = 1.0  # arbitrary pixel size
freqs = np.fft.fftfreq(nu, d=du)
ramp = np.abs(freqs)

# Compute all filter variants
window_types = ['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann']
filters = {}

for wtype in window_types:
    w = compute_window(nu, wtype)
    filters[wtype] = ramp * w * (2.0 * du)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ramp Filter Responses in Frequency Domain', fontsize=16, fontweight='bold')

# Plot 1: All filters together
ax = axes[0, 0]
for wtype in window_types:
    ax.plot(freqs[:nu//2], filters[wtype][:nu//2], label=wtype.upper(), linewidth=2)
ax.set_xlabel('Normalized Frequency', fontsize=11)
ax.set_ylabel('Filter Response', fontsize=11)
ax.set_title('All Filters Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, freqs[nu//2]])

# Plot 2: Linear scale zoom
ax = axes[0, 1]
for wtype in window_types:
    ax.plot(freqs[:nu//2], filters[wtype][:nu//2], label=wtype.upper(), linewidth=2)
ax.set_xlabel('Normalized Frequency', fontsize=11)
ax.set_ylabel('Filter Response', fontsize=11)
ax.set_title('Linear Scale (Zoomed)', fontsize=12, fontweight='bold')
ax.set_xlim([0, freqs[nu//4]])  # Zoom to lower frequencies
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Log scale
ax = axes[1, 0]
for wtype in window_types:
    # Avoid log(0) by adding small epsilon
    response = filters[wtype][:nu//2] + 1e-10
    ax.semilogy(freqs[:nu//2], response, label=wtype.upper(), linewidth=2)
ax.set_xlabel('Normalized Frequency', fontsize=11)
ax.set_ylabel('Filter Response (log scale)', fontsize=11)
ax.set_title('Logarithmic Scale', fontsize=12, fontweight='bold')
ax.set_xlim([0, freqs[nu//2]])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Plot 4: Window functions only
ax = axes[1, 1]
for wtype in window_types:
    w = compute_window(nu, wtype)
    ax.plot(freqs[:nu//2], w[:nu//2], label=wtype.upper(), linewidth=2)
ax.set_xlabel('Normalized Frequency', fontsize=11)
ax.set_ylabel('Window Response', fontsize=11)
ax.set_title('Apodization Windows Only', fontsize=12, fontweight='bold')
ax.set_xlim([0, freqs[nu//2]])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('ramp_filters_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: ramp_filters_comparison.png")

# Create a summary comparison table
print("\n" + "="*70)
print("RAMP FILTER CHARACTERISTICS")
print("="*70)
print(f"{'Filter Type':<15} {'Peak Response':<15} {'Sharpness':<15} {'Noise Level':<15}")
print("-"*70)

for wtype in window_types:
    response = filters[wtype]
    peak = np.max(response)
    
    # Measure "sharpness" as steepness at edges (high-frequency content)
    high_freq_power = np.sum(np.abs(response[3*nu//8:nu//2])**2)
    
    # Noise suppression (inverse relationship with high freq)
    noise_level = "HIGH" if wtype == 'ram-lak' else ("MEDIUM" if wtype in ['cosine'] else "LOW")
    
    print(f"{wtype.upper():<15} {peak:<15.4f} {high_freq_power:<15.2e} {noise_level:<15}")

print("="*70)
print("\nRecommendations:")
print("  • Shepp-Logan: Best balance (most commonly used in medical imaging)")
print("  • Hann/Hamming: More noise reduction, slightly blurrier")
print("  • Cosine: Moderate between Shepp-Logan and Ram-Lak")
print("  • Ram-Lak: Sharpest but noisiest (for high SNR data only)")
