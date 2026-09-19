from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "frozen" / "CH-Regime-220-v1" / "ch_regime_physics_trajectories.csv"
META_PATH = ROOT / "data" / "frozen" / "CH-Regime-220-v1" / "ch_regime_physics_metadata.csv"

OUT_DIR = ROOT / "results" / "frozen_dataset_understanding"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def load_data():
    df = pd.read_csv(DATA_PATH)
    meta = pd.read_csv(META_PATH)

    print("\nLoaded frozen dataset")
    print("=" * 70)
    print(f"Dataset: {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Trajectories: {df['trajectory_id'].nunique()}")
    print(f"Metadata: {META_PATH}")
    print(f"Metadata shape: {meta.shape}")

    return df, meta


def make_trajectory_summary(df: pd.DataFrame):
    controlled = df[df["recipe_stage"].isin(["feed", "hold"])].copy()
    controlled["abs_T_error_K"] = (controlled["T_K"] - controlled["T_set_K"]).abs()

    summary = (
        df.groupby("trajectory_id")
        .agg(
            dataset_split=("dataset_split", "first"),
            product=("product", "first"),
            season=("season", "first"),
            batch_number=("batch_number", "first"),
            disturbance_type=("disturbance_type", "first"),
            T_min_K=("T_K", "min"),
            T_max_K=("T_K", "max"),
            final_monomer_added_kg=("mM_added_kg", "max"),
            max_reaction_heat_kW=("Q_reaction_kW", "max"),
            min_U_kW_m2_K=("U_kW_m2_K", "min"),
            max_viscosity=("viscosity_proxy", "max"),
            max_valve_pct=("valve_actual_pct", "max"),
            min_valve_pct=("valve_actual_pct", "min"),
            max_feed_rate_kg_s=("feed_rate_kg_s", "max"),
        )
        .reset_index()
    )

    err = (
        controlled.groupby("trajectory_id")
        .agg(
            mean_abs_T_error_K=("abs_T_error_K", "mean"),
            p95_abs_T_error_K=("abs_T_error_K", lambda x: np.percentile(x, 95)),
            within_0p5K_pct=("abs_T_error_K", lambda x: 100 * np.mean(x <= 0.5)),
            within_1p0K_pct=("abs_T_error_K", lambda x: 100 * np.mean(x <= 1.0)),
        )
        .reset_index()
    )

    summary = summary.merge(err, on="trajectory_id", how="left")

    out_path = TABLE_DIR / "frozen_trajectory_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return summary


def plot_dataset_coverage(meta: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    meta["dataset_split"].value_counts().sort_index().plot(kind="bar")
    plt.title("Dataset Split Coverage")
    plt.xlabel("Split")
    plt.ylabel("Number of Trajectories")
    savefig("01_dataset_split_coverage.png")

    plt.figure(figsize=(8, 4))
    meta["product"].value_counts().sort_index().plot(kind="bar")
    plt.title("Product Coverage")
    plt.xlabel("Product")
    plt.ylabel("Number of Trajectories")
    savefig("02_product_coverage.png")

    plt.figure(figsize=(8, 4))
    meta["season"].value_counts().sort_index().plot(kind="bar")
    plt.title("Season Coverage")
    plt.xlabel("Season")
    plt.ylabel("Number of Trajectories")
    savefig("03_season_coverage.png")

    plt.figure(figsize=(8, 4))
    meta["batch_number"].value_counts().sort_index().plot(kind="bar")
    plt.title("Batch Number Coverage")
    plt.xlabel("Batch Number")
    plt.ylabel("Number of Trajectories")
    savefig("04_batch_number_coverage.png")

    plt.figure(figsize=(8, 4))
    meta["disturbance_type"].value_counts().sort_index().plot(kind="bar")
    plt.title("Disturbance Coverage")
    plt.xlabel("Disturbance Type")
    plt.ylabel("Number of Trajectories")
    plt.xticks(rotation=30, ha="right")
    savefig("05_disturbance_coverage.png")


def plot_stage_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    df["recipe_stage"].value_counts().sort_index().plot(kind="bar")
    plt.title("Recipe Stage Distribution")
    plt.xlabel("Stage")
    plt.ylabel("Number of Samples")
    savefig("06_recipe_stage_distribution.png")


def plot_temperature_samples(df: pd.DataFrame):
    sample_ids = []

    for split in ["baseline", "train", "val", "test", "ood"]:
        ids = df.loc[df["dataset_split"] == split, "trajectory_id"].drop_duplicates().head(3).tolist()
        sample_ids.extend(ids)

    plt.figure(figsize=(11, 5))

    for tid in sample_ids:
        d = df[df["trajectory_id"] == tid]
        label = f"id={tid}, {d['dataset_split'].iloc[0]}, P={d['product'].iloc[0]}, B={d['batch_number'].iloc[0]}"
        plt.plot(d["time_min"], d["T_K"], linewidth=1.0, alpha=0.85, label=label)

    plt.axhline(355.5, linestyle="--", linewidth=1.0, label="Setpoint 355.5 K")
    plt.title("Temperature Trajectories: Representative Samples")
    plt.xlabel("Time (min)")
    plt.ylabel("Reactor Temperature (K)")
    plt.legend(fontsize=7, ncol=2)
    savefig("07_temperature_trajectories_samples.png")


def plot_temperature_error(df: pd.DataFrame):
    controlled = df[df["recipe_stage"].isin(["feed", "hold"])].copy()
    controlled["T_error_K"] = controlled["T_K"] - controlled["T_set_K"]
    controlled["abs_T_error_K"] = controlled["T_error_K"].abs()

    plt.figure(figsize=(8, 4))
    plt.hist(controlled["T_error_K"], bins=80)
    plt.title("Controlled Temperature Error Distribution")
    plt.xlabel("T - T_set (K)")
    plt.ylabel("Frequency")
    savefig("08_temperature_error_histogram.png")

    order = ["baseline", "train", "val", "test", "ood"]
    data = [
        controlled.loc[controlled["dataset_split"] == split, "abs_T_error_K"].values
        for split in order
        if split in controlled["dataset_split"].unique()
    ]
    labels = [split for split in order if split in controlled["dataset_split"].unique()]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title("Absolute Temperature Error by Split")
    plt.xlabel("Split")
    plt.ylabel("|T - T_set| (K)")
    savefig("09_temperature_error_by_split.png")


def pick_interesting_trajectory(df: pd.DataFrame):
    summary = (
        df.groupby("trajectory_id")
        .agg(
            max_Q_reaction=("Q_reaction_kW", "max"),
            split=("dataset_split", "first"),
            disturbance=("disturbance_type", "first"),
        )
        .reset_index()
    )

    # Prefer a train/ood trajectory with high reaction heat.
    candidates = summary[summary["split"].isin(["train", "ood"])]
    tid = int(candidates.sort_values("max_Q_reaction", ascending=False).iloc[0]["trajectory_id"])
    return tid


def plot_single_trajectory_deep_dive(df: pd.DataFrame, tid: int):
    d = df[df["trajectory_id"] == tid].copy()

    title_suffix = (
        f"id={tid}, split={d['dataset_split'].iloc[0]}, "
        f"product={d['product'].iloc[0]}, batch={d['batch_number'].iloc[0]}, "
        f"season={d['season'].iloc[0]}, disturbance={d['disturbance_type'].iloc[0]}"
    )

    plt.figure(figsize=(11, 5))
    plt.plot(d["time_min"], d["T_K"], label="Reactor T")
    plt.plot(d["time_min"], d["T_set_K"], linestyle="--", label="Setpoint")
    plt.title(f"Single-Trajectory Temperature Profile\n{title_suffix}")
    plt.xlabel("Time (min)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    savefig("10_single_trajectory_temperature.png")

    plt.figure(figsize=(11, 5))
    plt.plot(d["time_min"], d["Q_feed_kW"], label="Q_feed")
    plt.plot(d["time_min"], d["Q_reaction_kW"], label="Q_reaction")
    plt.plot(d["time_min"], d["Q_jacket_kW"], label="Q_jacket")
    plt.plot(d["time_min"], d["Q_loss_kW"], label="Q_loss")
    plt.title(f"Heat-Flow Components\n{title_suffix}")
    plt.xlabel("Time (min)")
    plt.ylabel("Heat Flow (kW)")
    plt.legend()
    savefig("11_single_trajectory_heat_flows.png")

    plt.figure(figsize=(11, 5))
    plt.plot(d["time_min"], d["mM_kg"], label="Monomer in reactor")
    plt.plot(d["time_min"], d["mP_kg"], label="Polymer")
    plt.plot(d["time_min"], d["mM_added_kg"], label="Cumulative monomer added")
    plt.title(f"Mass Evolution\n{title_suffix}")
    plt.xlabel("Time (min)")
    plt.ylabel("Mass (kg)")
    plt.legend()
    savefig("12_single_trajectory_mass_evolution.png")

    plt.figure(figsize=(11, 5))
    plt.plot(d["time_min"], d["feed_rate_kg_s"], label="Actual feed rate")
    plt.plot(d["time_min"], d["feed_command_kg_s"], linestyle="--", label="Feed command")
    plt.title(f"Feed Dynamics\n{title_suffix}")
    plt.xlabel("Time (min)")
    plt.ylabel("Feed Rate (kg/s)")
    plt.legend()
    savefig("13_single_trajectory_feed_dynamics.png")

    plt.figure(figsize=(11, 5))
    plt.plot(d["time_min"], d["valve_actual_pct"], label="Actual valve")
    plt.plot(d["time_min"], d["valve_command_pct"], linestyle="--", label="Valve command")
    plt.title(f"Split-Range Valve Dynamics\n{title_suffix}")
    plt.xlabel("Time (min)")
    plt.ylabel("Valve Position (%)")
    plt.legend()
    savefig("14_single_trajectory_valve_dynamics.png")

    plt.figure(figsize=(11, 5))
    plt.plot(d["time_min"], d["reaction_rate_kg_s"], label="Reaction rate")
    plt.title(f"Reaction Rate\n{title_suffix}")
    plt.xlabel("Time (min)")
    plt.ylabel("Reaction Rate (kg/s)")
    plt.legend()
    savefig("15_single_trajectory_reaction_rate.png")

    plt.figure(figsize=(11, 5))
    plt.plot(d["time_min"], d["viscosity_proxy"], label="Viscosity proxy")
    plt.plot(d["time_min"], d["U_kW_m2_K"], label="Heat-transfer coefficient U")
    plt.title(f"Viscosity and Heat-Transfer Degradation\n{title_suffix}")
    plt.xlabel("Time (min)")
    plt.ylabel("Value")
    plt.legend()
    savefig("16_single_trajectory_viscosity_U.png")


def plot_physics_relationships(df: pd.DataFrame):
    sample = df.sample(n=min(60000, len(df)), random_state=42)

    plt.figure(figsize=(7, 5))
    plt.scatter(sample["viscosity_proxy"], sample["U_kW_m2_K"], s=4, alpha=0.25)
    plt.title("Physics Relation: U Decreases with Viscosity")
    plt.xlabel("Viscosity Proxy")
    plt.ylabel("U (kW/m²/K)")
    savefig("17_U_vs_viscosity_scatter.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(sample["reaction_rate_kg_s"], sample["Q_reaction_kW"], s=4, alpha=0.25)
    plt.title("Physics Relation: Reaction Heat Tracks Reaction Rate")
    plt.xlabel("Reaction Rate (kg/s)")
    plt.ylabel("Q_reaction (kW)")
    savefig("18_reaction_rate_vs_reaction_heat.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(sample["conversion_proxy"], sample["Q_reaction_kW"], s=4, alpha=0.25)
    plt.title("Conversion vs Reaction Heat")
    plt.xlabel("Conversion Proxy")
    plt.ylabel("Q_reaction (kW)")
    savefig("19_conversion_vs_reaction_heat.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(sample["T_K"], sample["dTdt_K_min"], s=4, alpha=0.25)
    plt.axvline(355.5, linestyle="--", linewidth=1.0)
    plt.title("Temperature Phase Portrait")
    plt.xlabel("Temperature (K)")
    plt.ylabel("dT/dt (K/min)")
    savefig("20_temperature_phase_portrait.png")


def plot_regime_understanding_maps(df: pd.DataFrame):
    sample = df.sample(n=min(70000, len(df)), random_state=7)

    stage_map = {"heatup": 0, "feed": 1, "hold": 2, "feed_pause": 3}
    sample["stage_code"] = sample["recipe_stage"].map(stage_map).fillna(-1)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        sample["conversion_proxy"],
        sample["Q_reaction_kW"],
        c=sample["stage_code"],
        s=4,
        alpha=0.35,
    )
    plt.title("Regime Map: Conversion vs Reaction Heat")
    plt.xlabel("Conversion Proxy")
    plt.ylabel("Q_reaction (kW)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Stage code: heatup=0, feed=1, hold=2")
    savefig("21_regime_map_conversion_reaction_heat.png")

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        sample["U_kW_m2_K"],
        sample["Q_jacket_kW"],
        c=sample["stage_code"],
        s=4,
        alpha=0.35,
    )
    plt.title("Regime Map: Heat Transfer vs Jacket Heat")
    plt.xlabel("U (kW/m²/K)")
    plt.ylabel("Q_jacket (kW)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Stage code: heatup=0, feed=1, hold=2")
    savefig("22_regime_map_U_jacket_heat.png")

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        sample["T_K"],
        sample["Q_jacket_kW"],
        c=sample["stage_code"],
        s=4,
        alpha=0.35,
    )
    plt.title("Regime Map: Temperature vs Jacket Heat")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Q_jacket (kW)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Stage code: heatup=0, feed=1, hold=2")
    savefig("23_regime_map_temperature_jacket_heat.png")


def plot_batch_and_disturbance_effects(summary: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    summary.boxplot(column="min_U_kW_m2_K", by="batch_number")
    plt.suptitle("")
    plt.title("Batch Fouling Effect: Batch Number vs Minimum U")
    plt.xlabel("Batch Number")
    plt.ylabel("Minimum U (kW/m²/K)")
    savefig("24_batch_number_vs_min_U.png")

    plt.figure(figsize=(9, 4))
    order = sorted(summary["disturbance_type"].dropna().unique())
    data = [
        summary.loc[summary["disturbance_type"] == disturbance, "p95_abs_T_error_K"].dropna().values
        for disturbance in order
    ]
    plt.boxplot(data, tick_labels=order, showfliers=True)
    plt.title("Disturbance Effect on Temperature Error")
    plt.xlabel("Disturbance Type")
    plt.ylabel("95th Percentile |T - T_set| (K)")
    plt.xticks(rotation=30, ha="right")
    savefig("25_disturbance_vs_temperature_error.png")


def plot_correlation_map(df: pd.DataFrame):
    cols = [
        "T_K",
        "dTdt_K_min",
        "mM_kg",
        "mP_kg",
        "mM_added_kg",
        "conversion_proxy",
        "solids_fraction",
        "viscosity_proxy",
        "U_kW_m2_K",
        "reaction_rate_kg_s",
        "Q_feed_kW",
        "Q_reaction_kW",
        "Q_jacket_kW",
        "Q_loss_kW",
        "valve_actual_pct",
        "feed_rate_kg_s",
    ]

    corr = df[cols].corr()
    out_path = TABLE_DIR / "physics_feature_correlation.csv"
    corr.to_csv(out_path)
    print(f"Saved: {out_path}")

    plt.figure(figsize=(11, 9))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Physics Feature Correlation Map")
    savefig("26_physics_feature_correlation_map.png")


def main():
    df, meta = load_data()

    summary = make_trajectory_summary(df)

    plot_dataset_coverage(meta)
    plot_stage_distribution(df)
    plot_temperature_samples(df)
    plot_temperature_error(df)

    tid = pick_interesting_trajectory(df)
    print(f"\nSelected trajectory for deep-dive plots: {tid}")
    plot_single_trajectory_deep_dive(df, tid)

    plot_physics_relationships(df)
    plot_regime_understanding_maps(df)
    plot_batch_and_disturbance_effects(summary)
    plot_correlation_map(df)

    print("\nDONE")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Tables saved to: {TABLE_DIR}")


if __name__ == "__main__":
    main()