from pathlib import Path
import sys
from types import SimpleNamespace

# Allow importing run_module2_final.py from scripts folder
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import run_module2_final as m


def main():
    print("=" * 80)
    print("RESUMING MODULE 2 FROM SAVED BEST MODEL")
    print("=" * 80)

    model_path = m.MODEL_DIR / "pi_mamba_regime_final_best.pt"
    window_path = m.WINDOW_DIR / "X_windows.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")

    if not window_path.exists():
        raise FileNotFoundError(f"Window file not found: {window_path}")

    print(f"Found best model: {model_path}")
    print(f"Found windows: {window_path}")

    print("\n[1/3] Extracting embeddings from saved PI-MambaRegime model")
    m.extract_embeddings(batch_size=512)

    print("\n[2/3] Clustering embeddings and running Regime Reality Test")
    m.cluster_regimes(k_min=4, k_max=8, seed=42)

    print("\n[3/3] Writing manifest")
    args = SimpleNamespace(
        resumed_from_checkpoint=True,
        checkpoint=str(model_path),
        k_min=4,
        k_max=8,
        seed=42,
        embedding_batch_size=512,
    )
    m.write_manifest(args)

    print("\n" + "=" * 80)
    print("MODULE 2 VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Reality report: {m.REGIME_DIR / 'regime_reality_test_report.txt'}")
    print(f"Regime assignments: {m.REGIME_DIR / 'regime_assignments.csv'}")
    print(f"Regime summary: {m.REGIME_DIR / 'regime_summary.csv'}")
    print(f"Figures: {m.FIG_DIR}")


if __name__ == "__main__":
    main()