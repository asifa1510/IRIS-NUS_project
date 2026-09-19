from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def validate_dataset(df: pd.DataFrame) -> Dict[str, float | int | str]:
    out: Dict[str, float | int | str] = {}

    out["n_rows"] = int(len(df))
    out["n_trajectories"] = int(df["trajectory_id"].nunique())
    out["missing_values"] = int(df.isna().sum().sum())

    out["T_min_K"] = float(df["T_K"].min())
    out["T_max_K"] = float(df["T_K"].max())

    out["valve_min_pct"] = float(df["valve_actual_pct"].min())
    out["valve_max_pct"] = float(df["valve_actual_pct"].max())

    out["feed_max_kg_s"] = float(df["feed_rate_kg_s"].max())

    out["median_final_monomer_added_kg"] = float(
        df.groupby("trajectory_id")["mM_added_kg"].max().median()
    )

    controlled = df[df["recipe_stage"].isin(["feed", "hold", "feed_pause"])].copy()

    if len(controlled) > 0:
        controlled["temp_error_K"] = controlled["T_K"] - controlled["T_set_K"]

        out["controlled_abs_error_mean_K"] = float(
            controlled["temp_error_K"].abs().mean()
        )

        out["controlled_abs_error_95pct_K"] = float(
            controlled["temp_error_K"].abs().quantile(0.95)
        )

        out["controlled_within_0p5K_pct"] = float(
            (controlled["temp_error_K"].abs() <= 0.5).mean() * 100.0
        )

        out["controlled_within_1p0K_pct"] = float(
            (controlled["temp_error_K"].abs() <= 1.0).mean() * 100.0
        )

    qsum = (
        df["Q_feed_kW"]
        + df["Q_reaction_kW"]
        + df["Q_jacket_kW"]
        + df["Q_loss_kW"]
    )

    energy_residual = df["C_total_kJ_K"] * df["dTdt_energy_K_s"] - qsum

    out["energy_balance_residual_median_abs_kW"] = float(
        energy_residual.abs().median()
    )

    out["energy_balance_residual_max_abs_kW"] = float(
        energy_residual.abs().max()
    )

    monomer_resid = df["dmMdt_num_kg_s"] - (
        df["feed_rate_kg_s"] - df["reaction_rate_kg_s"]
    )

    polymer_resid = df["dmPdt_num_kg_s"] - df["reaction_rate_kg_s"]

    out["monomer_balance_residual_median_abs"] = float(
        monomer_resid.abs().median()
    )

    out["polymer_balance_residual_median_abs"] = float(
        polymer_resid.abs().median()
    )

    sorted_df = df.sort_values(["trajectory_id", "time_s"]).copy()

    sorted_df["dvalve_step"] = (
        sorted_df.groupby("trajectory_id")["valve_actual_pct"].diff().abs()
    )

    sorted_df["dfeed_step"] = (
        sorted_df.groupby("trajectory_id")["feed_rate_kg_s"].diff().abs()
    )

    out["max_valve_step_per_saved_sample_pct"] = float(
        sorted_df["dvalve_step"].max()
    )

    out["max_feed_step_per_saved_sample_kg_s"] = float(
        sorted_df["dfeed_step"].max()
    )

    if df["viscosity_proxy"].nunique() > 3 and df["U_kW_m2_K"].nunique() > 3:
        out["corr_viscosity_U"] = float(
            df[["viscosity_proxy", "U_kW_m2_K"]].corr().iloc[0, 1]
        )

    out["stage_counts"] = str(df["recipe_stage"].value_counts().to_dict())

    out["split_counts"] = str(
        df[["trajectory_id", "dataset_split"]]
        .drop_duplicates()["dataset_split"]
        .value_counts()
        .to_dict()
    )

    return out


def save_validation_report(summary: Dict[str, float | int | str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("CH-REGIME PHYSICS DATASET VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


def plot_dataset(df: pd.DataFrame, out_dir: Path, max_traj: int = 50) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for i, (_, g) in enumerate(df.groupby("trajectory_id")):
        if i >= max_traj:
            break
        plt.plot(g["time_min"], g["T_K"], alpha=0.35)

    plt.axhline(df["T_set_K"].median(), linestyle="--", label="Nominal setpoint")
    plt.xlabel("Time [min]")
    plt.ylabel("Reactor temperature [K]")
    plt.title("Physics-consistent reactor temperature trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "temperature_trajectories.png", dpi=200)
    plt.close()

    first_id = df["trajectory_id"].min()
    g = df[df["trajectory_id"] == first_id]

    plt.figure(figsize=(10, 5))
    plt.plot(g["time_min"], g["Q_reaction_kW"], label="Reaction heat")
    plt.plot(g["time_min"], g["Q_feed_kW"], label="Feed heat")
    plt.plot(g["time_min"], g["Q_jacket_kW"], label="Jacket heat")
    plt.plot(g["time_min"], g["Q_loss_kW"], label="Ambient loss")
    plt.xlabel("Time [min]")
    plt.ylabel("Heat flow [kW]")
    plt.title("Energy-balance heat-flow components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "heat_flow_components.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(g["time_min"], g["valve_command_pct"], label="Valve command", alpha=0.65)
    plt.plot(g["time_min"], g["valve_actual_pct"], label="Actual valve")
    plt.xlabel("Time [min]")
    plt.ylabel("Valve [%]")
    plt.title("Rate-limited split-range valve movement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "valve_smoothness.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(g["time_min"], g["mM_kg"], label="Monomer mass")
    plt.plot(g["time_min"], g["mP_kg"], label="Polymer mass")
    plt.plot(g["time_min"], g["mM_added_kg"], label="Cumulative monomer fed")
    plt.xlabel("Time [min]")
    plt.ylabel("Mass [kg]")
    plt.title("Material balance behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mass_balance.png", dpi=200)
    plt.close()

    sample = df.sample(min(len(df), 40000), random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(sample["viscosity_proxy"], sample["U_kW_m2_K"], s=3, alpha=0.25)
    plt.xscale("log")
    plt.xlabel("Viscosity proxy [log scale]")
    plt.ylabel("U [kW m$^{-2}$ K$^{-1}$]")
    plt.title("Heat-transfer degradation with viscosity and fouling")
    plt.tight_layout()
    plt.savefig(out_dir / "U_vs_viscosity.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(sample["T_K"], sample["dTdt_K_min"], s=3, alpha=0.25)
    plt.xlabel("Reactor temperature [K]")
    plt.ylabel("dT/dt from energy balance [K/min]")
    plt.title("Temperature phase plot")
    plt.tight_layout()
    plt.savefig(out_dir / "temperature_phase_plot.png", dpi=200)
    plt.close()