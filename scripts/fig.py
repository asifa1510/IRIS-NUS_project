from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

FIG_DIR = ROOT / "results" / "module2_final" / "figures"
REGIME_DIR = ROOT / "results" / "module2_final" / "regimes"
OUT_DIR = ROOT / "results" / "module2_final" / "final_paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Copy the 3 important existing plots
important_files = {
    "05_latent_pca_by_regime.png": "Fig1_latent_space_by_discovered_regime.png",
    "06_latent_pca_by_recipe_stage.png": "Fig2_latent_space_by_recipe_stage.png",
    "09_regime_transition_matrix.png": "Fig3_regime_transition_matrix.png",
}

for src_name, dst_name in important_files.items():
    src = FIG_DIR / src_name
    dst = OUT_DIR / dst_name
    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied: {dst}")
    else:
        print(f"Missing: {src}")

# Create one compact physics fingerprint plot
summary = pd.read_csv(REGIME_DIR / "regime_summary.csv")

features = [
    "mean_U",
    "mean_viscosity",
    "mean_feed_rate",
    "mean_Q_reaction",
    "mean_Q_jacket",
    "mean_dTdt_K_min",
]

display_names = [
    "U",
    "Viscosity",
    "Feed rate",
    "Reaction heat",
    "Jacket heat",
    "dT/dt",
]

X = summary[features].values
X_scaled = StandardScaler().fit_transform(X)

fig, ax = plt.subplots(figsize=(9, 4.8))

im = ax.imshow(X_scaled, aspect="auto")

ax.set_xticks(range(len(display_names)))
ax.set_xticklabels(display_names, rotation=30, ha="right")
ax.set_yticks(range(len(summary)))
ax.set_yticklabels([f"R{int(r)}" for r in summary["regime_id"]])

ax.set_title("Physics Fingerprint of Discovered Regimes")
ax.set_xlabel("Physics feature")
ax.set_ylabel("Discovered regime")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Standardized mean value")

plt.tight_layout()

out_path = OUT_DIR / "Fig4_regime_physics_fingerprint.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
print(f"\nFinal paper figures saved in:\n{OUT_DIR}")