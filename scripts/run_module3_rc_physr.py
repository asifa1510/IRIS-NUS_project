"""
Module 3 Final Pipeline
Safe RC-PhySR: Reliability-Gated Regime-Conditioned Physics-Guided Symbolic Regression

Input:
    results/module2_final/regimes/regime_assignments.csv

Output:
    results/module3_rc_physr/
"""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "module2_final" / "regimes" / "regime_assignments.csv"
OUT_ROOT = ROOT / "results" / "module3_rc_physr"
MODEL_DIR = OUT_ROOT / "models"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"
REPORT_DIR = OUT_ROOT / "reports"

for d in [OUT_ROOT, MODEL_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE_FEATURES_DTDT = [
    "Q_reaction_kW",
    "Q_jacket_kW",
    "Q_feed_kW",
    "Q_loss_kW",
    "feed_rate_kg_s",
    "U_kW_m2_K",
    "viscosity_proxy",
    "solids_fraction",
    "conversion_proxy",
    "valve_actual_pct",
]

BASE_FEATURES_U = [
    "viscosity_proxy",
    "solids_fraction",
    "feed_rate_kg_s",
    "Q_reaction_kW",
    "Q_jacket_kW",
    "conversion_proxy",
    "valve_actual_pct",
]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def safe_r2(y_true, y_pred):
    if len(np.unique(np.asarray(y_true))) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def clean_array(x):
    x = np.asarray(x, dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x


def build_library(df, target):
    """
    Compact physics-guided symbolic term library.
    The final law is sparse over standardized terms: z(term).
    """

    if target == "U_kW_m2_K":
        base = [c for c in BASE_FEATURES_U if c in df.columns]
    else:
        base = [c for c in BASE_FEATURES_DTDT if c in df.columns]

    lib = pd.DataFrame(index=df.index)

    for c in base:
        lib[c] = df[c].astype(float)

    heat_cols = [c for c in ["Q_reaction_kW", "Q_jacket_kW", "Q_feed_kW", "Q_loss_kW"] if c in df.columns]
    if len(heat_cols) >= 2:
        lib["Q_total_kW"] = df[heat_cols].sum(axis=1)

    if "Q_jacket_kW" in df.columns:
        qj = df["Q_jacket_kW"].astype(float)
        lib["cooling_load_kW"] = np.maximum(-qj, 0.0)
        lib["heating_load_kW"] = np.maximum(qj, 0.0)

    if "viscosity_proxy" in df.columns:
        visc = df["viscosity_proxy"].astype(float)
        lib["inv_viscosity"] = 1.0 / (visc + 1e-6)
        lib["log_viscosity"] = np.log1p(np.maximum(visc, 0.0))

    if "U_kW_m2_K" in df.columns and "viscosity_proxy" in df.columns:
        lib["U_times_viscosity"] = df["U_kW_m2_K"] * df["viscosity_proxy"]
        lib["U_over_viscosity"] = df["U_kW_m2_K"] / (df["viscosity_proxy"] + 1e-6)

    if "U_kW_m2_K" in df.columns and "solids_fraction" in df.columns:
        lib["U_times_solids"] = df["U_kW_m2_K"] * df["solids_fraction"]

    if "feed_rate_kg_s" in df.columns and "viscosity_proxy" in df.columns:
        lib["feed_times_viscosity"] = df["feed_rate_kg_s"] * df["viscosity_proxy"]

    if "Q_reaction_kW" in df.columns and "feed_rate_kg_s" in df.columns:
        lib["reaction_times_feed"] = df["Q_reaction_kW"] * df["feed_rate_kg_s"]

    if "Q_jacket_kW" in df.columns and "U_kW_m2_K" in df.columns:
        lib["jacket_times_U"] = df["Q_jacket_kW"] * df["U_kW_m2_K"]

    if "Q_total_kW" in lib.columns and "U_kW_m2_K" in df.columns:
        lib["Q_total_times_U"] = lib["Q_total_kW"] * df["U_kW_m2_K"]

    for c in ["Q_total_kW", "Q_reaction_kW", "Q_jacket_kW", "U_kW_m2_K", "viscosity_proxy", "feed_rate_kg_s"]:
        if c in lib.columns:
            lib[f"{c}^2"] = lib[c] ** 2

    lib = lib.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keep_cols = []
    for c in lib.columns:
        if float(lib[c].std()) > 1e-12:
            keep_cols.append(c)

    return lib[keep_cols]


class SparseSymbolicLaw:
    def __init__(self, target, alpha_grid=None, max_terms=8, random_state=42):
        self.target = target
        self.alpha_grid = alpha_grid
        self.max_terms = max_terms
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.term_names = None
        self.intercept_ = None
        self.coef_ = None
        self.selected_terms_ = None
        self.equation_ = None

    def fit(self, df_train):
        lib = build_library(df_train, self.target)
        y = df_train[self.target].astype(float).values

        self.term_names = list(lib.columns)
        X = clean_array(lib.values)
        y = clean_array(y)

        # Reliability range used by Safe RC-PhySR fallback.
        # If a local regime law predicts far outside the target range it saw
        # during training, we fallback to the global law for that sample.
        self.y_min_ = float(np.min(y))
        self.y_max_ = float(np.max(y))
        self.y_range_ = float(self.y_max_ - self.y_min_)
        self.y_margin_ = max(0.50 * self.y_range_, 1e-4)

        Xz = self.scaler.fit_transform(X)

        if self.alpha_grid is None:
            self.alpha_grid = np.logspace(-5, -1, 35)

        model = LassoCV(
            alphas=self.alpha_grid,
            cv=3,
            max_iter=20000,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(Xz, y)

        coef = model.coef_.copy()
        nonzero = np.where(np.abs(coef) > 1e-10)[0]
        if len(nonzero) > self.max_terms:
            top = nonzero[np.argsort(np.abs(coef[nonzero]))[::-1][: self.max_terms]]
            mask = np.zeros_like(coef, dtype=bool)
            mask[top] = True
            coef[~mask] = 0.0

        self.model = model
        self.intercept_ = float(model.intercept_)
        self.coef_ = coef
        self.selected_terms_ = [(self.term_names[i], float(coef[i])) for i in np.where(np.abs(coef) > 1e-10)[0]]
        self.equation_ = self.make_equation()
        return self

    def predict(self, df):
        lib = build_library(df, self.target)
        for c in self.term_names:
            if c not in lib.columns:
                lib[c] = 0.0
        lib = lib[self.term_names]
        X = clean_array(lib.values)
        Xz = self.scaler.transform(X)
        return self.intercept_ + Xz @ self.coef_

    def make_equation(self):
        pieces = [f"{self.target} = {self.intercept_:.6g}"]
        for name, coef in self.selected_terms_:
            sign = "+" if coef >= 0 else "-"
            pieces.append(f" {sign} {abs(coef):.6g}*z({name})")
        return "".join(pieces)

    def complexity(self):
        return int(len(self.selected_terms_))


def evaluate_model(name, law, df_eval, split_name, regime_id="global"):
    if len(df_eval) == 0:
        return None
    y = df_eval[law.target].astype(float).values
    pred = law.predict(df_eval)
    return {
        "model": name,
        "regime_id": regime_id,
        "split": split_name,
        "n_samples": int(len(df_eval)),
        "rmse": rmse(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": safe_r2(y, pred),
        "complexity": law.complexity(),
        "equation": law.equation_,
    }


def predict_regime_conditioned(df, regime_laws, global_law):
    """
    Safe RC-PhySR prediction.

    Use the regime-specific law when its prediction remains inside the
    reliability range learned from that regime's training data. If the local
    symbolic law extrapolates outside that range, fallback to the global law.

    This prevents near-steady-state regimes from producing unstable symbolic
    extrapolations on baseline/OOD-like data.
    """
    preds = np.zeros(len(df), dtype=float)

    for rid, idx in df.groupby("regime_id").groups.items():
        rid_int = int(rid)
        part = df.loc[idx]
        positions = df.index.get_indexer(idx)

        global_pred = global_law.predict(part)

        if rid_int not in regime_laws:
            final_pred = global_pred
        else:
            law = regime_laws[rid_int]
            local_pred = law.predict(part)

            lo = law.y_min_ - law.y_margin_
            hi = law.y_max_ + law.y_margin_

            unsafe = (
                ~np.isfinite(local_pred)
                | (local_pred < lo)
                | (local_pred > hi)
            )

            final_pred = local_pred.copy()
            final_pred[unsafe] = global_pred[unsafe]

        preds[positions] = final_pred

    return preds


def evaluate_regime_conditioned(df_eval, target, regime_laws, global_law, split_name):
    if len(df_eval) == 0:
        return None
    y = df_eval[target].astype(float).values
    pred = predict_regime_conditioned(df_eval, regime_laws, global_law)
    return {
        "model": "regime_conditioned",
        "regime_id": "piecewise",
        "split": split_name,
        "n_samples": int(len(df_eval)),
        "rmse": rmse(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": safe_r2(y, pred),
        "complexity": int(sum(law.complexity() for law in regime_laws.values())),
        "equation": "piecewise law selected by PI-MambaRegime regime_id",
    }


def physics_sign_score(law):
    if law.target != "dTdt_K_min":
        return np.nan, "not_applicable"

    expected_positive_terms = ["Q_reaction_kW", "Q_jacket_kW", "Q_feed_kW", "Q_loss_kW", "Q_total_kW"]
    checked = 0
    passed = 0
    for term, coef in law.selected_terms_:
        if term in expected_positive_terms:
            checked += 1
            if coef > 0:
                passed += 1
    if checked == 0:
        return np.nan, "no_direct_heat_terms_selected"
    score = passed / checked
    return float(score), "pass" if score >= 0.75 else "check"


def plot_global_vs_piecewise(metrics_df, target):
    plot_df = metrics_df[(metrics_df["model"].isin(["global", "regime_conditioned"])) & (metrics_df["split"].isin(["val", "test", "ood"]))].copy()
    if len(plot_df) == 0:
        return
    splits = ["val", "test", "ood"]
    global_vals = []
    piecewise_vals = []
    for s in splits:
        g = plot_df[(plot_df["split"] == s) & (plot_df["model"] == "global")]
        p = plot_df[(plot_df["split"] == s) & (plot_df["model"] == "regime_conditioned")]
        global_vals.append(float(g["rmse"].iloc[0]) if len(g) else np.nan)
        piecewise_vals.append(float(p["rmse"].iloc[0]) if len(p) else np.nan)

    x = np.arange(len(splits))
    width = 0.35
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width / 2, global_vals, width, label="Global symbolic law")
    plt.bar(x + width / 2, piecewise_vals, width, label="Regime-conditioned laws")
    plt.xticks(x, [s.upper() for s in splits])
    plt.ylabel("RMSE")
    plt.title(f"Global vs Regime-Conditioned Symbolic Laws ({target})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig1_global_vs_regime_conditioned_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(df_eval, target, global_law, regime_laws, split_name):
    if len(df_eval) == 0:
        return
    df_plot = df_eval.sample(5000, random_state=42).copy() if len(df_eval) > 5000 else df_eval.copy()
    y = df_plot[target].astype(float).values
    pred_global = global_law.predict(df_plot)
    pred_piece = predict_regime_conditioned(df_plot, regime_laws, global_law)
    plt.figure(figsize=(6, 6))
    plt.scatter(y, pred_global, s=6, alpha=0.25, label="Global")
    plt.scatter(y, pred_piece, s=6, alpha=0.25, label="Regime-conditioned")
    lo = min(np.min(y), np.min(pred_global), np.min(pred_piece))
    hi = max(np.max(y), np.max(pred_global), np.max(pred_piece))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual on {split_name.upper()} Split ({target})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"Fig2_predicted_vs_actual_{split_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_law_complexity(law_rows):
    rows = [r for r in law_rows if r["model"] == "regime_specific"]
    if len(rows) == 0:
        return
    df = pd.DataFrame(rows).sort_values("regime_id")
    plt.figure(figsize=(7.5, 4.2))
    plt.bar([f"R{int(r)}" for r in df["regime_id"]], df["complexity"])
    plt.xlabel("Regime")
    plt.ylabel("Number of symbolic terms")
    plt.title("Complexity of Regime-Specific Symbolic Laws")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig3_regime_law_complexity.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_report(target, global_law, regime_laws, metrics_df, law_table, args):
    report_path = REPORT_DIR / "law_reality_test_report.txt"

    def metric(model, split, col):
        m = metrics_df[(metrics_df["model"] == model) & (metrics_df["split"] == split)]
        if len(m) == 0:
            return np.nan
        return float(m[col].iloc[0])

    test_global = metric("global", "test", "rmse")
    test_piece = metric("regime_conditioned", "test", "rmse")
    ood_global = metric("global", "ood", "rmse")
    ood_piece = metric("regime_conditioned", "ood", "rmse")

    improvement_test = 100.0 * (test_global - test_piece) / (test_global + 1e-12)
    improvement_ood = 100.0 * (ood_global - ood_piece) / (ood_global + 1e-12)

    sign_scores = []
    for _, law in regime_laws.items():
        score, _ = physics_sign_score(law)
        if np.isfinite(score):
            sign_scores.append(score)
    avg_sign_score = float(np.mean(sign_scores)) if sign_scores else np.nan

    gates = {
        "global_law_fitted": global_law is not None,
        "all_regime_laws_fitted": len(regime_laws) >= args.min_regimes,
        "test_piecewise_not_worse_than_global": test_piece <= test_global * 1.05,
        "test_piecewise_improves_global": test_piece < test_global,
        "ood_piecewise_not_worse_than_global": ood_piece <= ood_global * 1.10,
        "baseline_piecewise_not_worse_than_global": metric("regime_conditioned", "baseline", "rmse") <= metric("global", "baseline", "rmse") * 1.10,
        "average_law_complexity_reasonable": law_table["complexity"].max() <= args.max_terms,
    }
    if np.isfinite(avg_sign_score):
        gates["physics_sign_score_reasonable"] = avg_sign_score >= 0.70

    final_status = "PASS" if all(gates.values()) else "CHECK"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MODULE 3 LAW REALITY TEST REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write("METHOD\n")
        f.write("-" * 72 + "\n")
        f.write("Safe RC-PhySR: Reliability-Gated Regime-Conditioned Physics-Guided Symbolic Regression\n")
        f.write("Regimes used: PI-MambaRegime discovered regime_id from Module 2\n")
        f.write(f"Target variable: {target}\n")
        f.write("Equation format: sparse symbolic law over standardized physics terms z(term)\n\n")
        f.write("GLOBAL LAW\n")
        f.write("-" * 72 + "\n")
        f.write(global_law.equation_ + "\n\n")
        f.write("REGIME-SPECIFIC LAWS\n")
        f.write("-" * 72 + "\n")
        for rid in sorted(regime_laws.keys()):
            f.write(f"R{rid}: {regime_laws[rid].equation_}\n")
        f.write("\nGLOBAL VS REGIME-CONDITIONED PERFORMANCE\n")
        f.write("-" * 72 + "\n")
        show = metrics_df[metrics_df["model"].isin(["global", "regime_conditioned"])][["model", "split", "n_samples", "rmse", "mae", "r2", "complexity"]]
        f.write(show.to_string(index=False))
        f.write("\n\nIMPROVEMENT SUMMARY\n")
        f.write("-" * 72 + "\n")
        f.write(f"Test RMSE improvement of regime-conditioned over global: {improvement_test:.3f}%\n")
        f.write(f"OOD RMSE improvement of regime-conditioned over global: {improvement_ood:.3f}%\n")
        if np.isfinite(avg_sign_score):
            f.write(f"Average physics sign score: {avg_sign_score:.3f}\n")
        f.write("\nACCEPTANCE GATES\n")
        f.write("-" * 72 + "\n")
        for k, v in gates.items():
            f.write(f"{'PASS' if v else 'CHECK'}: {k}\n")
        f.write("\nFINAL STATUS\n")
        f.write("-" * 72 + "\n")
        f.write(final_status + "\n")

    print(f"Saved report: {report_path}")
    return final_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--target", type=str, default="dTdt_K_min")
    parser.add_argument("--max-terms", type=int, default=8)
    parser.add_argument("--min-train-per-regime", type=int, default=300)
    parser.add_argument("--min-regimes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    input_path = Path(args.input)

    print("=" * 80)
    print("MODULE 3 FINAL PIPELINE: RC-PhySR")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {OUT_ROOT}")
    print(f"Target: {args.target}")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    required = ["regime_id", "dataset_split", args.target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in regime assignments: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[args.target, "regime_id", "dataset_split"]).copy()
    df["regime_id"] = df["regime_id"].astype(int)

    print(f"Loaded samples: {len(df)}")
    print("Split counts:")
    print(df["dataset_split"].value_counts())
    print("Regime counts:")
    print(df["regime_id"].value_counts().sort_index())

    train_df = df[df["dataset_split"] == "train"].copy()
    if len(train_df) < 1000:
        raise ValueError("Too few train samples for symbolic law learning.")

    print("\n[1/5] Fitting global symbolic law")
    global_law = SparseSymbolicLaw(target=args.target, max_terms=args.max_terms, random_state=args.seed).fit(train_df)
    print("Global law:")
    print(global_law.equation_)

    print("\n[2/5] Fitting regime-specific symbolic laws")
    regime_laws = {}
    law_rows = []

    global_sign_score, global_sign_status = physics_sign_score(global_law)
    law_rows.append({
        "model": "global",
        "regime_id": "global",
        "n_train": int(len(train_df)),
        "complexity": global_law.complexity(),
        "physics_sign_score": global_sign_score,
        "physics_sign_status": global_sign_status,
        "equation": global_law.equation_,
    })

    for rid in sorted(train_df["regime_id"].unique()):
        part = train_df[train_df["regime_id"] == rid].copy()
        if len(part) < args.min_train_per_regime:
            print(f"Skipping R{rid}: only {len(part)} train samples")
            continue
        law = SparseSymbolicLaw(target=args.target, max_terms=args.max_terms, random_state=args.seed).fit(part)
        regime_laws[int(rid)] = law
        sign_score, sign_status = physics_sign_score(law)
        law_rows.append({
            "model": "regime_specific",
            "regime_id": int(rid),
            "n_train": int(len(part)),
            "complexity": law.complexity(),
            "physics_sign_score": sign_score,
            "physics_sign_status": sign_status,
            "equation": law.equation_,
        })
        print(f"R{rid}: {law.equation_}")

    law_table = pd.DataFrame(law_rows)
    law_table.to_csv(TABLE_DIR / "symbolic_law_table.csv", index=False)

    print("\n[3/5] Evaluating global and regime-conditioned laws")
    metric_rows = []
    for split in ["train", "val", "test", "ood", "baseline"]:
        split_df = df[df["dataset_split"] == split].copy()
        if len(split_df) == 0:
            continue
        row = evaluate_model("global", global_law, split_df, split, "global")
        if row:
            metric_rows.append(row)
        row = evaluate_regime_conditioned(split_df, args.target, regime_laws, global_law, split)
        if row:
            metric_rows.append(row)

    for rid, law in regime_laws.items():
        for split in ["train", "val", "test", "ood"]:
            part = df[(df["dataset_split"] == split) & (df["regime_id"] == rid)].copy()
            row = evaluate_model("regime_specific", law, part, split, rid)
            if row:
                metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(TABLE_DIR / "global_vs_regime_symbolic_metrics.csv", index=False)

    print(metrics_df[metrics_df["model"].isin(["global", "regime_conditioned"])][["model", "split", "n_samples", "rmse", "mae", "r2", "complexity"]])

    print("\n[4/5] Creating final Module 3 figures")
    plot_global_vs_piecewise(metrics_df, args.target)
    test_df = df[df["dataset_split"] == "test"].copy()
    if len(test_df) > 0:
        plot_predicted_vs_actual(test_df, args.target, global_law, regime_laws, "test")
    plot_law_complexity(law_rows)
    print(f"Saved figures to: {FIG_DIR}")

    print("\n[5/5] Writing Law Reality Test report")
    final_status = write_report(args.target, global_law, regime_laws, metrics_df, law_table, args)

    manifest = {
        "module": "Module 3",
        "method": "RC-PhySR",
        "target": args.target,
        "input": str(input_path),
        "output_root": str(OUT_ROOT),
        "n_samples": int(len(df)),
        "n_regime_laws": int(len(regime_laws)),
        "final_status": final_status,
        "artifacts": {
            "law_table": str(TABLE_DIR / "symbolic_law_table.csv"),
            "metrics": str(TABLE_DIR / "global_vs_regime_symbolic_metrics.csv"),
            "report": str(REPORT_DIR / "law_reality_test_report.txt"),
            "figures": str(FIG_DIR),
        },
    }

    with open(OUT_ROOT / "MODULE3_RC_PHYSR_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print("\n" + "=" * 80)
    print("MODULE 3 COMPLETE")
    print("=" * 80)
    print(f"Final status: {final_status}")
    print(f"Report: {REPORT_DIR / 'law_reality_test_report.txt'}")
    print(f"Law table: {TABLE_DIR / 'symbolic_law_table.csv'}")
    print(f"Metrics: {TABLE_DIR / 'global_vs_regime_symbolic_metrics.csv'}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
