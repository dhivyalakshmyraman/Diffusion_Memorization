"""
Section 1 — Theory Overview
============================
Key theoretical insights from the paper:
"Why Diffusion Models Don't Memorize"
"""

THEORY = """
╔══════════════════════════════════════════════════════════════════════╗
║                        THEORY OVERVIEW                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. TWO TIMESCALES IN DIFFUSION TRAINING                          ║
║  ─────────────────────────────────────────                         ║
║  • τ_gen  ≈ constant   — time for good generative samples         ║
║  • τ_mem  ∝ N          — time for memorization to appear          ║
║  For large N the gap τ_mem − τ_gen grows, creating a wide         ║
║  "generalization window" where the model generates well            ║
║  WITHOUT memorizing.                                               ║
║                                                                    ║
║  2. KERNEL–SPECTRUM EXPLANATION                                    ║
║  ─────────────────────────────────                                 ║
║  Score matching can be linearized around initialization →          ║
║  dynamics governed by a Neural Tangent Kernel (NTK):               ║
║                                                                    ║
║      K = (1/N) Φ Φ^T                                              ║
║                                                                    ║
║  Eigenvalues of K determine learning speeds:                       ║
║    • Large λ_k  →  fast learning  →  smooth, population modes     ║
║    • Small λ_k  →  slow learning  →  memorization modes           ║
║                                                                    ║
║  The smallest eigenvalue λ_min ~ 1/N, so memorization time        ║
║  τ_mem ~ 1/λ_min ~ N.                                             ║
║                                                                    ║
║  3. MARCHENKO–PASTUR LAW                                          ║
║  ──────────────────────────                                        ║
║  For a random feature matrix Φ ∈ ℝ^{N×P}, the eigenvalue          ║
║  distribution of K follows the Marchenko–Pastur law:               ║
║                                                                    ║
║    ρ(λ) = (N/P) / (2π σ² λ) √((λ₊ − λ)(λ − λ₋))                ║
║                                                                    ║
║  where λ₊, λ₋ = σ²(1 ± √(N/P))². This gives a continuous        ║
║  spectrum with a pile-up of small eigenvalues that govern          ║
║  memorization.                                                     ║
║                                                                    ║
║  4. IMPLICIT REGULARIZATION                                       ║
║  ───────────────────────────                                       ║
║  Gradient descent acts as a spectral filter: at time t,            ║
║  mode k has learned a fraction  1 − exp(−λ_k t).                  ║
║  Early stopping therefore naturally suppresses modes with          ║
║  small λ_k — the memorization directions — providing              ║
║  IMPLICIT dynamic regularization without any explicit penalty.     ║
║                                                                    ║
║  5. PHYSICAL ANALOGY: GLASSY DYNAMICS                              ║
║  ────────────────────────────────────                               ║
║  The rugged loss landscape with many nearly–degenerate minima      ║
║  resembles a spin–glass energy surface. "Slow modes" in GD        ║
║  correspond to crossing tiny barriers between near-identical       ║
║  local minima — analogous to glassy relaxation.                    ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝

MATHEMATICAL SUMMARY
─────────────────────
  Score function:   s(x,t) = ∇_x log p_t(x)
  Empirical score:  s_emp(x,t) = −(1/σ_t²) Σ_i w_i(x,t)(x − x_i)
                    where w_i ∝ N(x; x_i, σ_t²)

  Kernel dynamics:  df/dt = −K(f − y)
  Solution:         f(t) = y − exp(−Kt) y
  Per eigenmode:    f_k(t) = (1 − exp(−λ_k t)) y_k

  Memorization time for mode k:  τ_k ~ 1/λ_k
  Smallest eigenvalue:           λ_min ~ 1/N
  Therefore:                     τ_mem ~ N  ✓
"""

print(THEORY)
