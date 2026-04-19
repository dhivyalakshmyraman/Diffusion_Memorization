"""
Section 4 — Score Spikes Around Samples
========================================
Creates a synthetic 1D dataset, computes the empirical smoothed density
and the score function s(x)=∇_x log p_t(x), and visualizes the sharp
spikes that point toward training samples = memorization directions.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(77)

# ── Synthetic 1D training data ─────────────────────────────────────
samples = np.array([-3.0, -1.0, 0.5, 2.0, 3.5])
N = len(samples)
x = np.linspace(-6, 7, 2000)

# ── Noise levels (diffusion time) ──────────────────────────────────
sigmas = [2.0, 1.0, 0.4, 0.15]

fig, axes = plt.subplots(len(sigmas), 2, figsize=(14, 3.2 * len(sigmas)))

for row, sigma in enumerate(sigmas):
    # ── Smoothed density: p_t(x) = (1/N) Σ N(x; x_i, σ²) ────────
    density = np.zeros_like(x)
    for xi in samples:
        density += np.exp(-0.5 * ((x - xi) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    density /= N

    # ── Score function: s(x) = ∇_x log p_t(x) ────────────────────
    # s(x) = p'(x) / p(x)
    # p'(x) = -(1/N) Σ (x-x_i)/σ² · N(x; x_i, σ²)
    grad_density = np.zeros_like(x)
    for xi in samples:
        gauss = np.exp(-0.5 * ((x - xi) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        grad_density += -(x - xi) / (sigma ** 2) * gauss
    grad_density /= N

    score = grad_density / (density + 1e-30)

    # ── Left panel: density ──────────────────────────────────────
    ax = axes[row, 0]
    ax.fill_between(x, density, alpha=0.3, color='steelblue')
    ax.plot(x, density, 'b-', lw=2)
    for xi in samples:
        ax.axvline(xi, color='red', ls=':', alpha=0.5, lw=1)
    ax.scatter(samples, np.zeros_like(samples), c='red', s=60, zorder=5,
               marker='^', label='Training points')
    ax.set_ylabel('p_t(x)', fontsize=12)
    ax.set_title(f'σ_t = {sigma}', fontsize=13)
    if row == 0:
        ax.legend(fontsize=10)

    # ── Right panel: score function ──────────────────────────────
    ax = axes[row, 1]
    ax.plot(x, score, 'darkgreen', lw=2)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    for xi in samples:
        ax.axvline(xi, color='red', ls=':', alpha=0.5, lw=1)
    ax.set_ylabel('s(x) = ∇log p_t', fontsize=12)
    ax.set_title(f'Score function at σ_t = {sigma}', fontsize=13)
    ax.set_ylim(np.percentile(score, 1) * 1.2, np.percentile(score, 99) * 1.2)

axes[-1, 0].set_xlabel('x', fontsize=12)
axes[-1, 1].set_xlabel('x', fontsize=12)
plt.suptitle('Score Function Spikes → Memorization Directions', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "score_spikes.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: score_spikes.png")

# ── PLOT 2: Score field as arrows (2D quiver-style on 1D) ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for idx, sigma in enumerate([1.5, 0.2]):
    ax = axes[idx]
    x_q = np.linspace(-5.5, 5.5, 60)

    density_q = np.zeros_like(x_q)
    grad_q = np.zeros_like(x_q)
    for xi in samples:
        g = np.exp(-0.5 * ((x_q - xi) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        density_q += g
        grad_q += -(x_q - xi) / (sigma ** 2) * g
    density_q /= N
    grad_q /= N
    score_q = grad_q / (density_q + 1e-30)

    # Normalize arrow lengths for visibility
    max_s = np.percentile(np.abs(score_q), 95)
    score_norm = np.clip(score_q / (max_s + 1e-10), -1, 1)

    ax.quiver(x_q, np.zeros_like(x_q), score_norm, np.zeros_like(x_q),
              score_norm, cmap='coolwarm', scale=25, width=0.004, alpha=0.8)
    ax.scatter(samples, np.zeros_like(samples), c='red', s=100, zorder=5,
               marker='o', edgecolors='black', linewidths=1.5)
    ax.set_title(f'Score Vectors (σ_t = {sigma})', fontsize=13)
    ax.set_xlabel('x', fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(-5.5, 5.5)

plt.suptitle('Score Vectors Point Toward Training Samples', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "score_vectors.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: score_vectors.png")

print("""
  INTERPRETATION
  ──────────────
  • At LARGE σ_t (high noise), the smoothed density is broad and
    featureless. The score function is smooth — pointing toward the
    center of mass. This corresponds to learning the smooth structure.
  • At SMALL σ_t (low noise), δ-function spikes emerge around each
    training point. The score develops SHARP PEAKS pointing directly
    at individual x_i — these are the MEMORIZATION directions.
  • In diffusion training, these spiky directions are fit by the
    SMALLEST eigenvalues of K, which take the longest to learn.
  • Physical analogy: A glass of water (large σ) vs. ice cubes
    (small σ). The smooth liquid = generalization; the sharp
    crystalline edges = memorization.
""")
