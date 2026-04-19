"""
=============================================================================
WHY DIFFUSION MODELS DON'T MEMORIZE
The Role of Implicit Dynamical Regularization in Training
=============================================================================
Complete Experimental Lab — Run All Sections
=============================================================================
"""
import os, sys

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 72)
print("WHY DIFFUSION MODELS DON'T MEMORIZE")
print("The Role of Implicit Dynamical Regularization in Training")
print("=" * 72)

sections = [
    ("sec1_theory_overview", "Section 1 — Theory Overview"),
    ("sec2_random_feature_model", "Section 2 — Random Feature Model & Kernel Spectrum"),
    ("sec3_marchenko_pastur", "Section 3 — Marchenko–Pastur Spectrum"),
    ("sec4_score_spikes", "Section 4 — Score Spikes Around Samples"),
    ("sec5_diffusion_training", "Section 5 — Diffusion Training Experiment"),
    ("sec6_dataset_scaling", "Section 6 — Dataset Scaling Experiment"),
    ("sec7_glassy_dynamics", "Section 7 — Glassy Dynamics Analogy"),
]

for module_name, title in sections:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")
    try:
        __import__(module_name)
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()

print("\n" + "=" * 72)
print("ALL SECTIONS COMPLETE — Figures saved to:", FIGURES_DIR)
print("=" * 72)
