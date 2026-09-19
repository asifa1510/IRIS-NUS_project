from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "ch_regime_physics_trajectories.csv"
META_PATH = ROOT / "data" / "processed" / "ch_regime_physics_metadata.csv"

OUT_DIR = ROOT / "results" / "journal_diagnostics"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


PHYSICS_COLS = [
    "T_K",
    "mM_kg",
    "mP_kg",
    "mM_added_kg",
    "conversion_proxy",
    "solids_fraction",
    "viscosity_proxy",
    "U_kW_m2_K",
    "reaction_rate_kg_s",
    "Q_reaction_kW",
    "Q_jacket_kW",
    "Q_feed_kW",
    "Q_loss_kW",
    "valve_actual_pct",
    "feed_rate_kg_s",
    "dTdt_energy_K_s",
]


def savefig(name: str) -> None:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print("Saved:", path)


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "First run: python scripts\\generate_dataset.py --baseline 2 --train 4 --val 1 --test 1 --ood 1"
        )

    df = pd.read_csv(DATA_PATH)

    if META_PATH.exists():
        meta = pd.read_csv(META_PATH)
    else:
        meta = pd.DataFrame()

    return df, meta


def trajectory_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tid, g in df.groupby("trajectory_id"):
        g = g.sort_values("time_s")

        controlled = g[g["recipe_stage"].isin(["feed", "hold", "feed_pause"])].copy()

        if len(controlled) > 0:
            temp_error = controlled["T_K"] - controlled["T_set_K"]
            mean_abs_error = float(temp_error.abs().mean())
            p95_abs_error = float(temp_error.abs().quantile(0.95))
            within_05 = float((temp_error.abs() <= 0.5).mean() * 100.0)
            within_10 = float((temp_error.abs() <= 1.0).mean() * 100.0)
        else:
            mean_abs_error = np.nan
            p95_abs_error = np.nan
            within_05 = np.nan
            within_10 = np.nan

        qsum = (
            g["Q_feed_kW"]
            + g["Q_reaction_kW"]
            + g["Q_jacket_kW"]
            + g["Q_loss_kW"]
        )

        energy_resid = g["C_total_kJ_K"] * g["dTdt_energy_K_s"] - qsum

        monomer_resid = g["dmMdt_num_kg_s"] - (
            g["feed_rate_kg_s"] - g["reaction_rate_kg_s"]
        )

        polymer_resid = g["dmPdt_num_kg_s"] - g["reaction_rate_kg_s"]

        dvalve = g["valve_actual_pct"].diff().abs()
        dfeed = g["feed_rate_kg_s"].diff().abs()

        rows.append(
            {
                "trajectory_id": tid,
                "dataset_split": g["dataset_split"].iloc[0],
                "product": g["product"].iloc[0],
                "batch_number": int(g["batch_number"].iloc[0]),
                "season": g["season"].iloc[0],
                "disturbance_type": g["disturbance_type"].iloc[0],
                "impurity": float(g["impurity"].iloc[0]),
                "fouling_resistance": float(g["fouling_resistance"].iloc[0]),
                "n_points": len(g),
                "final_monomer_added_kg": float(g["mM_added_kg"].max()),
                "final_monomer_mass_kg": float(g["mM_kg"].iloc[-1]),
                "final_polymer_mass_kg": float(g["mP_kg"].iloc[-1]),
                "final_conversion_proxy": float(g["conversion_proxy"].iloc[-1]),
                "T_min_K": float(g["T_K"].min()),
                "T_max_K": float(g["T_K"].max()),
                "T_final_K": float(g["T_K"].iloc[-1]),
                "controlled_mean_abs_error_K": mean_abs_error,
                "controlled_95_abs_error_K": p95_abs_error,
                "controlled_within_0p5K_pct": within_05,
                "controlled_within_1p0K_pct": within_10,
                "max_reaction_heat_kW": float(g["Q_reaction_kW"].max()),
                "min_jacket_heat_kW": float(g["Q_jacket_kW"].min()),
                "max_jacket_heat_kW": float(g["Q_jacket_kW"].max()),
                "max_reaction_rate_kg_s": float(g["reaction_rate_kg_s"].max()),
                "min_U_kW_m2_K": float(g["U_kW_m2_K"].min()),
                "max_U_kW_m2_K": float(g["U_kW_m2_K"].max()),
                "max_viscosity_proxy": float(g["viscosity_proxy"].max()),
                "energy_residual_median_abs_kW": float(energy_resid.abs().median()),
                "energy_residual_max_abs_kW": float(energy_resid.abs().max()),
                "monomer_residual_median_abs": float(monomer_resid.abs().median()),
                "polymer_residual_median_abs": float(polymer_resid.abs().median()),
                "max_valve_step_pct": float(dvalve.max()),
                "max_feed_step_kg_s": float(dfeed.max()),
                "stage_sequence": "->".join(g["recipe_stage"].drop_duplicates().astype(str).tolist()),
            }
        )

    summary = pd.DataFrame(rows)
    path = TABLE_DIR / "trajectory_summary.csv"
    summary.to_csv(path, index=False)
    print("Saved:", path)

    return summary


def acceptance_report(df: pd.DataFrame, summary: pd.DataFrame) -> Dict[str, object]:
    controlled = df[df["recipe_stage"].isin(["feed", "hold", "feed_pause"])].copy()
    controlled["temp_error_K"] = controlled["T_K"] - controlled["T_set_K"]

    qsum = (
        df["Q_feed_kW"]
        + df["Q_reaction_kW"]
        + df["Q_jacket_kW"]
        + df["Q_loss_kW"]
    )

    energy_resid = df["C_total_kJ_K"] * df["dTdt_energy_K_s"] - qsum

    monomer_resid = df["dmMdt_num_kg_s"] - (
        df["feed_rate_kg_s"] - df["reaction_rate_kg_s"]
    )

    polymer_resid = df["dmPdt_num_kg_s"] - df["reaction_rate_kg_s"]

    split_counts = (
        summary["dataset_split"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    product_counts = (
        summary["product"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    season_counts = (
        summary["season"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    batch_counts = (
        summary["batch_number"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    stage_counts = df["recipe_stage"].value_counts().to_dict()

    report = {
        "n_rows": int(len(df)),
        "n_trajectories": int(df["trajectory_id"].nunique()),
        "missing_values": int(df.isna().sum().sum()),
        "split_counts": split_counts,
        "product_counts": product_counts,
        "season_counts": season_counts,
        "batch_counts": batch_counts,
        "stage_counts": stage_counts,
        "T_range_K": (float(df["T_K"].min()), float(df["T_K"].max())),
        "controlled_mean_abs_error_K": float(controlled["temp_error_K"].abs().mean()),
        "controlled_95_abs_error_K": float(controlled["temp_error_K"].abs().quantile(0.95)),
        "controlled_within_0p5K_pct": float((controlled["temp_error_K"].abs() <= 0.5).mean() * 100.0),
        "controlled_within_1p0K_pct": float((controlled["temp_error_K"].abs() <= 1.0).mean() * 100.0),
        "energy_residual_median_abs_kW": float(energy_resid.abs().median()),
        "energy_residual_max_abs_kW": float(energy_resid.abs().max()),
        "monomer_residual_median_abs": float(monomer_resid.abs().median()),
        "polymer_residual_median_abs": float(polymer_resid.abs().median()),
        "max_valve_step_pct": float(summary["max_valve_step_pct"].max()),
        "max_feed_step_kg_s": float(summary["max_feed_step_kg_s"].max()),
        "corr_viscosity_U": float(df[["viscosity_proxy", "U_kW_m2_K"]].corr().iloc[0, 1]),
        "corr_reaction_rate_reaction_heat": float(df[["reaction_rate_kg_s", "Q_reaction_kW"]].corr().iloc[0, 1]),
    }

    gates = {
        "missing_values_is_zero": report["missing_values"] == 0,
        "has_all_splits": {"baseline", "train", "val", "test", "ood"}.issubset(set(summary["dataset_split"].unique())),
        "has_products_A_B": {"A", "B"}.issubset(set(summary["product"].unique())),
        "has_seasons": {"summer", "winter"}.issubset(set(summary["season"].unique())),
        "has_batches_1_to_5": {1, 2, 3, 4, 5}.issubset(set(summary["batch_number"].unique())),
        "has_heatup_feed_hold": {"heatup", "feed", "hold"}.issubset(set(df["recipe_stage"].unique())),
        "energy_balance_good": report["energy_residual_median_abs_kW"] < 1e-8,
        "mass_balance_good": (
            report["monomer_residual_median_abs"] < 1e-5
            and report["polymer_residual_median_abs"] < 1e-5
        ),
        "valve_smooth": report["max_valve_step_pct"] <= 3.0,
        "U_decreases_with_viscosity": report["corr_viscosity_U"] < -0.50,
        "reaction_heat_tracks_reaction_rate": report["corr_reaction_rate_reaction_heat"] > 0.98,
        "temperature_control_reasonable": report["controlled_within_1p0K_pct"] >= 90.0,
    }

    report["gates"] = gates
    report["overall_status"] = "PASS" if all(gates.values()) else "NEEDS_ATTENTION"

    path = OUT_DIR / "journal_acceptance_report.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("CH-REGIME JOURNAL DATASET ACCEPTANCE REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 70 + "\n")

        for k, v in report.items():
            if k != "gates":
                f.write(f"{k}: {v}\n")

        f.write("\nACCEPTANCE GATES\n")
        f.write("-" * 70 + "\n")

        for k, v in gates.items():
            label = "PASS" if v else "CHECK"
            f.write(f"{label}: {k}\n")

        f.write("\nFINAL STATUS\n")
        f.write("-" * 70 + "\n")
        f.write(str(report["overall_status"]) + "\n")

    print("Saved:", path)

    return report


def plot_split_coverage(summary: pd.DataFrame) -> None:
    counts = summary["dataset_split"].value_counts().reindex(
        ["baseline", "train", "val", "test", "ood"]
    ).fillna(0)

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Dataset split")
    plt.ylabel("Number of trajectories")
    plt.title("Trajectory coverage by dataset split")
    savefig("01_split_coverage.png")


def plot_product_season_batch_coverage(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    product_counts = summary["product"].value_counts().sort_index()
    season_counts = summary["season"].value_counts().sort_index()
    batch_counts = summary["batch_number"].value_counts().sort_index()

    axes[0].bar(product_counts.index.astype(str), product_counts.values)
    axes[0].set_title("Product coverage")
    axes[0].set_xlabel("Product")
    axes[0].set_ylabel("Trajectories")

    axes[1].bar(season_counts.index.astype(str), season_counts.values)
    axes[1].set_title("Season coverage")
    axes[1].set_xlabel("Season")

    axes[2].bar(batch_counts.index.astype(str), batch_counts.values)
    axes[2].set_title("Batch-number coverage")
    axes[2].set_xlabel("Batch number")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_product_season_batch_coverage.png", dpi=220)
    plt.close()
    print("Saved:", FIG_DIR / "02_product_season_batch_coverage.png")


def plot_stage_distribution(df: pd.DataFrame) -> None:
    counts = df["recipe_stage"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Recipe stage")
    plt.ylabel("Number of samples")
    plt.title("Recipe-stage sample distribution")
    savefig("03_stage_distribution.png")


def plot_temperature_trajectories(df: pd.DataFrame, max_per_split: int = 12) -> None:
    plt.figure(figsize=(11, 6))

    for split, gsplit in df.groupby("dataset_split"):
        tids = gsplit["trajectory_id"].drop_duplicates().head(max_per_split).tolist()

        for tid in tids:
            g = gsplit[gsplit["trajectory_id"] == tid]
            plt.plot(g["time_min"], g["T_K"], alpha=0.35, linewidth=1)

    plt.axhline(df["T_set_K"].median(), linestyle="--", linewidth=1.5, label="Nominal setpoint")
    plt.xlabel("Time [min]")
    plt.ylabel("Reactor temperature [K]")
    plt.title("Reactor temperature trajectories across dataset splits")
    plt.legend()
    savefig("04_temperature_trajectories_all_splits.png")


def plot_temperature_error(df: pd.DataFrame) -> None:
    controlled = df[df["recipe_stage"].isin(["feed", "hold", "feed_pause"])].copy()
    controlled["temp_error_K"] = controlled["T_K"] - controlled["T_set_K"]

    plt.figure(figsize=(9, 5))
    plt.hist(controlled["temp_error_K"], bins=80, alpha=0.85)
    plt.axvline(-0.5, linestyle="--", label="-0.5 K")
    plt.axvline(0.5, linestyle="--", label="+0.5 K")
    plt.axvline(-1.0, linestyle=":", label="-1.0 K")
    plt.axvline(1.0, linestyle=":", label="+1.0 K")
    plt.xlabel("Temperature error, T - Tset [K]")
    plt.ylabel("Samples")
    plt.title("Controlled-stage temperature error distribution")
    plt.legend()
    savefig("05_temperature_error_histogram.png")


def plot_error_by_split(summary: pd.DataFrame) -> None:
    order = ["baseline", "train", "val", "test", "ood"]
    data = [
        summary.loc[summary["dataset_split"] == split, "controlled_95_abs_error_K"].dropna().values
        for split in order
    ]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=order, showfliers=True)
    plt.axhline(0.5, linestyle="--", label="0.5 K")
    plt.axhline(1.0, linestyle=":", label="1.0 K")
    plt.xlabel("Dataset split")
    plt.ylabel("Trajectory 95% absolute error [K]")
    plt.title("Temperature-control difficulty by split")
    plt.legend()
    savefig("06_temperature_error_by_split.png")


def plot_heat_flows_sample(df: pd.DataFrame) -> None:
    tid = int(df["trajectory_id"].iloc[0])
    g = df[df["trajectory_id"] == tid].sort_values("time_s")

    plt.figure(figsize=(11, 6))
    plt.plot(g["time_min"], g["Q_reaction_kW"], label="Reaction heat")
    plt.plot(g["time_min"], g["Q_jacket_kW"], label="Jacket heat")
    plt.plot(g["time_min"], g["Q_feed_kW"], label="Feed heat")
    plt.plot(g["time_min"], g["Q_loss_kW"], label="Ambient loss")
    plt.xlabel("Time [min]")
    plt.ylabel("Heat flow [kW]")
    plt.title(f"Energy-balance heat-flow components, trajectory {tid}")
    plt.legend()
    savefig("07_heat_flow_components_sample.png")


def plot_mass_conversion_sample(df: pd.DataFrame) -> None:
    tid = int(df["trajectory_id"].iloc[0])
    g = df[df["trajectory_id"] == tid].sort_values("time_s")

    plt.figure(figsize=(11, 6))
    plt.plot(g["time_min"], g["mM_kg"], label="Monomer mass")
    plt.plot(g["time_min"], g["mP_kg"], label="Polymer mass")
    plt.plot(g["time_min"], g["mM_added_kg"], label="Cumulative monomer added")
    plt.xlabel("Time [min]")
    plt.ylabel("Mass [kg]")
    plt.title(f"Material-balance variables, trajectory {tid}")
    plt.legend()
    savefig("08_mass_balance_sample.png")

    plt.figure(figsize=(11, 5))
    plt.plot(g["time_min"], g["conversion_proxy"])
    plt.xlabel("Time [min]")
    plt.ylabel("Conversion proxy")
    plt.title(f"Conversion evolution, trajectory {tid}")
    savefig("09_conversion_sample.png")


def plot_valve_feed_sample(df: pd.DataFrame) -> None:
    tid = int(df["trajectory_id"].iloc[0])
    g = df[df["trajectory_id"] == tid].sort_values("time_s")

    plt.figure(figsize=(11, 6))
    plt.plot(g["time_min"], g["valve_command_pct"], label="Valve command", alpha=0.65)
    plt.plot(g["time_min"], g["valve_actual_pct"], label="Rate-limited actual valve")
    plt.xlabel("Time [min]")
    plt.ylabel("Valve [%]")
    plt.title(f"Split-range valve command vs actuator response, trajectory {tid}")
    plt.legend()
    savefig("10_valve_dynamics_sample.png")

    plt.figure(figsize=(11, 5))
    plt.plot(g["time_min"], g["feed_command_kg_s"], label="Feed command", alpha=0.65)
    plt.plot(g["time_min"], g["feed_rate_kg_s"], label="Actual feed")
    plt.xlabel("Time [min]")
    plt.ylabel("Feed rate [kg/s]")
    plt.title(f"Feed actuator dynamics, trajectory {tid}")
    plt.legend()
    savefig("11_feed_dynamics_sample.png")


def plot_U_viscosity(df: pd.DataFrame) -> None:
    sample = df.sample(min(len(df), 60000), random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(sample["viscosity_proxy"], sample["U_kW_m2_K"], s=4, alpha=0.25)
    plt.xscale("log")
    plt.xlabel("Viscosity proxy [log scale]")
    plt.ylabel("U [kW m$^{-2}$ K$^{-1}$]")
    plt.title("Heat-transfer degradation with viscosity and fouling")
    savefig("12_U_vs_viscosity.png")


def plot_reaction_heat_relation(df: pd.DataFrame) -> None:
    sample = df.sample(min(len(df), 60000), random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(sample["reaction_rate_kg_s"], sample["Q_reaction_kW"], s=4, alpha=0.25)
    plt.xlabel("Reaction rate [kg/s]")
    plt.ylabel("Reaction heat [kW]")
    plt.title("Reaction heat consistency with reaction rate")
    savefig("13_reaction_rate_vs_reaction_heat.png")


def plot_temperature_phase(df: pd.DataFrame) -> None:
    sample = df.sample(min(len(df), 60000), random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(sample["T_K"], sample["dTdt_K_min"], s=4, alpha=0.25)
    plt.xlabel("Reactor temperature [K]")
    plt.ylabel("dT/dt [K/min]")
    plt.title("Temperature phase portrait")
    savefig("14_temperature_phase_portrait.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = [c for c in PHYSICS_COLS if c in df.columns]
    corr = df[cols].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(np.arange(len(cols)), cols, rotation=90)
    plt.yticks(np.arange(len(cols)), cols)
    plt.title("Physics-feature correlation map")
    savefig("15_physics_feature_correlation_map.png")

    path = TABLE_DIR / "physics_feature_correlation.csv"
    corr.to_csv(path)
    print("Saved:", path)


def plot_residuals(df: pd.DataFrame) -> None:
    qsum = (
        df["Q_feed_kW"]
        + df["Q_reaction_kW"]
        + df["Q_jacket_kW"]
        + df["Q_loss_kW"]
    )

    energy_resid = df["C_total_kJ_K"] * df["dTdt_energy_K_s"] - qsum

    monomer_resid = df["dmMdt_num_kg_s"] - (
        df["feed_rate_kg_s"] - df["reaction_rate_kg_s"]
    )

    polymer_resid = df["dmPdt_num_kg_s"] - df["reaction_rate_kg_s"]

    plt.figure(figsize=(9, 5))
    plt.hist(energy_resid, bins=80)
    plt.xlabel("Energy balance residual [kW]")
    plt.ylabel("Samples")
    plt.title("Energy-balance residual distribution")
    savefig("16_energy_residual_histogram.png")

    plt.figure(figsize=(9, 5))
    plt.hist(monomer_resid, bins=80, alpha=0.75, label="Monomer residual")
    plt.hist(polymer_resid, bins=80, alpha=0.75, label="Polymer residual")
    plt.xlabel("Mass-balance residual [kg/s]")
    plt.ylabel("Samples")
    plt.title("Mass-balance residual distribution")
    plt.legend()
    savefig("17_mass_residual_histogram.png")


def plot_regime_proxy_map(df: pd.DataFrame) -> None:
    sample = df.sample(min(len(df), 80000), random_state=42)

    stage_to_num = {
        "heatup": 0,
        "feed": 1,
        "feed_pause": 2,
        "hold": 3,
    }

    stage_num = sample["recipe_stage"].map(stage_to_num).fillna(-1)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        sample["conversion_proxy"],
        sample["Q_reaction_kW"],
        c=stage_num,
        s=4,
        alpha=0.35,
    )
    plt.colorbar(sc, label="Stage index")
    plt.xlabel("Conversion proxy")
    plt.ylabel("Reaction heat [kW]")
    plt.title("Physical regime proxy map: conversion vs reaction heat")
    savefig("18_conversion_reaction_heat_stage_map.png")


def plot_batch_fouling_effect(summary: pd.DataFrame) -> None:
    grouped = summary.groupby("batch_number").agg(
        min_U_mean=("min_U_kW_m2_K", "mean"),
        max_error_mean=("controlled_95_abs_error_K", "mean"),
        n=("trajectory_id", "count"),
    ).reset_index()

    grouped.to_csv(TABLE_DIR / "batch_fouling_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["batch_number"], grouped["min_U_mean"], marker="o")
    plt.xlabel("Batch number")
    plt.ylabel("Mean minimum U [kW m$^{-2}$ K$^{-1}$]")
    plt.title("Effect of repeated batching / fouling on heat transfer")
    savefig("19_batch_number_vs_min_U.png")

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["batch_number"], grouped["max_error_mean"], marker="o")
    plt.xlabel("Batch number")
    plt.ylabel("Mean trajectory 95% absolute temperature error [K]")
    plt.title("Control difficulty across batch number")
    savefig("20_batch_number_vs_temperature_error.png")


def main() -> None:
    df, meta = load_dataset()

    print("\nLoaded dataset:")
    print(DATA_PATH)
    print("Shape:", df.shape)
    print("Trajectories:", df["trajectory_id"].nunique())

    if len(meta) > 0:
        print("Metadata shape:", meta.shape)

    summary = trajectory_summary(df)
    report = acceptance_report(df, summary)

    plot_split_coverage(summary)
    plot_product_season_batch_coverage(summary)
    plot_stage_distribution(df)
    plot_temperature_trajectories(df)
    plot_temperature_error(df)
    plot_error_by_split(summary)
    plot_heat_flows_sample(df)
    plot_mass_conversion_sample(df)
    plot_valve_feed_sample(df)
    plot_U_viscosity(df)
    plot_reaction_heat_relation(df)
    plot_temperature_phase(df)
    plot_correlation_heatmap(df)
    plot_residuals(df)
    plot_regime_proxy_map(df)
    plot_batch_fouling_effect(summary)

    print("\nJOURNAL DATASET DIAGNOSTICS COMPLETE")
    print("Status:", report["overall_status"])
    print("Report:", OUT_DIR / "journal_acceptance_report.txt")
    print("Figures:", FIG_DIR)
    print("Tables:", TABLE_DIR)


if __name__ == "__main__":
    main()