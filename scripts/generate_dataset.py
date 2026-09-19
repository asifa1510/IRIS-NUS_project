from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ch_regime_dataset.simulator import (
    random_config,
    simulate_batch,
    metadata_from_config,
)

from ch_regime_dataset.validation import (
    validate_dataset,
    save_validation_report,
    plot_dataset,
)


def generate_split(
    split: str,
    n: int,
    seed: int,
    start_id: int,
) -> tuple[list[pd.DataFrame], list[dict]]:
    frames: list[pd.DataFrame] = []
    metadata: list[dict] = []

    for i in range(n):
        tid = start_id + i

        cfg = random_config(
            trajectory_id=tid,
            split=split,
            seed=seed + tid,
        )

        print(
            f"[{split}] trajectory {i + 1}/{n} | "
            f"id={tid} | product={cfg.product} | "
            f"batch={cfg.batch_number} | season={cfg.season} | "
            f"disturbance={cfg.disturbance_type}",
            flush=True,
        )

        df = simulate_batch(cfg)

        frames.append(df)
        metadata.append(metadata_from_config(cfg))

    return frames, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate journal-level CH-Regime physics dataset."
    )

    parser.add_argument("--baseline", type=int, default=10)
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--ood", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "data" / "processed"),
    )

    parser.add_argument(
        "--fig",
        type=str,
        default=str(ROOT / "results" / "figures"),
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = Path(args.fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    all_meta: list[dict] = []

    start = 0

    for split, n in [
        ("baseline", args.baseline),
        ("train", args.train),
        ("val", args.val),
        ("test", args.test),
        ("ood", args.ood),
    ]:
        frames, meta = generate_split(
            split=split,
            n=n,
            seed=args.seed,
            start_id=start,
        )

        all_frames.extend(frames)
        all_meta.extend(meta)

        start += n

    df = pd.concat(all_frames, ignore_index=True)
    meta_df = pd.DataFrame(all_meta)

    data_path = out_dir / "ch_regime_physics_trajectories.csv"
    meta_path = out_dir / "ch_regime_physics_metadata.csv"
    summary_path = out_dir / "ch_regime_validation_summary.txt"

    df.to_csv(data_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    summary = validate_dataset(df)
    save_validation_report(summary, summary_path)
    plot_dataset(df, fig_dir)

    print("\nSaved dataset:", data_path)
    print("Saved metadata:", meta_path)
    print("Saved validation summary:", summary_path)
    print("Saved figures to:", fig_dir)

    print("\nVALIDATION SUMMARY")
    print("=" * 60)

    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()