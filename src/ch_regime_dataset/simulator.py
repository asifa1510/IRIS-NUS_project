from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ReactorParams:
    """
    Physics-consistent Chylla-Haase-style semibatch polymerization reactor.

    Important:
    - Reactor temperature is solved from the plant energy balance.
    - Controller only manipulates split-range valve position.
    - Valve and feed actuator movements are rate-limited.
    - Temperature is never directly imposed.
    """

    mM0: float = 0.0
    mP0: float = 11.227
    mW: float = 42.75

    cpM: float = 1.675
    cpP: float = 3.140
    cpW: float = 4.187
    cpC: float = 4.187

    MW_M: float = 104.0
    k0: float = 55.0
    E: float = 29560.89
    R: float = 8.314
    deltaH_poly: float = -70152.16

    area: float = 1.93
    h0: float = 0.9814
    h_lower: float = 0.2
    UA_loss: float = 5.67567e-3

    mC: float = 21.455
    F_C: float = 0.9412
    tau_jacket_in: float = 25.0

    T_steam: float = 449.82
    T_inlet_summer: float = 294.26
    T_inlet_winter: float = 277.26

    T_set_nominal: float = 355.5
    total_monomer_target: float = 45.0


@dataclass
class ReactorConfig:
    trajectory_id: int = 0
    dataset_split: str = "train"

    product: str = "A"
    batch_number: int = 1
    season: str = "summer"
    ambient_T: Optional[float] = None

    impurity: float = 1.0
    kinetic_multiplier: float = 1.0
    heat_transfer_multiplier: float = 1.0
    utility_temp_bias_K: float = 0.0
    fouling_multiplier: float = 1.0

    t_final: float = 36000.0
    dt: float = 20.0

    T_set: float = 355.5
    feed_start: float = 1800.0
    feed_rate_nominal: float = 0.0028
    feed_start_tolerance_K: float = 1.0

    feed_tau: float = 120.0
    feed_rate_limit: float = 3.0e-5

    controller_Kp: float = 4.5
    controller_Ki: float = 0.0010
    controller_Kd: float = 360.0
    integral_limit: float = 4000.0

    valve_tau: float = 55.0
    valve_rate_limit: float = 0.08
    valve_noise_std: float = 0.0

    feed_temp_bias_K: float = 0.0
    cooling_water_disturbance_K: float = 0.0
    disturbance_start_s: float = 1e12
    disturbance_duration_s: float = 0.0
    disturbance_type: str = "none"
    disturbance_magnitude: float = 0.0

    seed: Optional[int] = None


def ambient_temperature(cfg: ReactorConfig) -> float:
    if cfg.ambient_T is not None:
        return float(cfg.ambient_T)
    return 277.26 if cfg.season.lower() == "winter" else 294.26


def inlet_water_temperature(cfg: ReactorConfig, p: ReactorParams) -> float:
    base = p.T_inlet_winter if cfg.season.lower() == "winter" else p.T_inlet_summer
    return float(base + cfg.utility_temp_bias_K + cfg.cooling_water_disturbance_K)


def disturbance_active(t: float, cfg: ReactorConfig) -> bool:
    return cfg.disturbance_start_s <= t <= cfg.disturbance_start_s + cfg.disturbance_duration_s


def feed_temperature(t: float, cfg: ReactorConfig) -> float:
    T = ambient_temperature(cfg) + cfg.feed_temp_bias_K
    if disturbance_active(t, cfg) and cfg.disturbance_type == "feed_temp":
        T += cfg.disturbance_magnitude
    return float(T)


def fouling_resistance(cfg: ReactorConfig) -> float:
    table = {
        1: 0.0000,
        2: 0.0005,
        3: 0.0010,
        4: 0.0015,
        5: 0.0020,
    }
    return float(table.get(int(cfg.batch_number), 0.0) * cfg.fouling_multiplier)


def solids_fraction(mP: float, mM: float, p: ReactorParams) -> float:
    total = p.mW + max(mP, 0.0) + max(mM, 0.0)
    if total <= 0.0:
        return 0.0
    return float(np.clip(max(mP, 0.0) / total, 0.0, 0.95))


def viscosity(mP: float, mM: float, cfg: ReactorConfig, p: ReactorParams) -> float:
    f = solids_fraction(mP, mM, p)

    if cfg.product.upper() == "B":
        base = 0.032
        exponent = 19.7 * f
    else:
        base = 0.052
        exponent = 16.4 * f

    mu = base * np.exp(exponent)
    return float(np.clip(mu, 0.01, 1.0e4))


def heat_transfer_coefficient(mu: float, cfg: ReactorConfig, p: ReactorParams) -> float:
    h = p.h0 * np.exp(-0.00513 * mu)
    h = max(p.h_lower, h)
    h *= cfg.heat_transfer_multiplier
    h = max(p.h_lower, h)

    rf = fouling_resistance(cfg)

    if rf > 0.0:
        U = 1.0 / ((1.0 / h) + rf)
    else:
        U = h

    return float(max(p.h_lower, U))


def reaction_rate_kg_s(
    t: float,
    T: float,
    mM: float,
    mP: float,
    cfg: ReactorConfig,
    p: ReactorParams,
) -> float:
    if mM <= 1e-12:
        return 0.0

    nM = max(mM, 0.0) / p.MW_M
    k = p.k0 * np.exp(-p.E / (p.R * max(T, 1.0)))

    f = solids_fraction(mP, mM, p)
    gel = 1.0 + 8.0 * (f**2 / (f**2 + 0.22**2))

    impurity = cfg.impurity
    if disturbance_active(t, cfg) and cfg.disturbance_type == "impurity":
        impurity *= 1.0 + cfg.disturbance_magnitude

    Rp_kmol_s = cfg.kinetic_multiplier * impurity * k * nM * gel
    r = Rp_kmol_s * p.MW_M

    return float(np.clip(r, 0.0, 0.08))


def feed_command_kg_s(
    t: float,
    T: float,
    mM_added: float,
    cfg: ReactorConfig,
    p: ReactorParams,
) -> float:
    if mM_added >= p.total_monomer_target:
        return 0.0

    if t < cfg.feed_start:
        return 0.0

    feed_has_started = mM_added > 1e-6
    reactor_ready = abs(T - cfg.T_set) <= cfg.feed_start_tolerance_K

    if not feed_has_started and not reactor_ready:
        return 0.0

    error_hot = T - cfg.T_set
    factor = 1.0

    if error_hot > 1.2:
        factor = 0.0
    elif error_hot > 0.8:
        factor = 0.25
    elif error_hot > 0.4:
        factor = 0.60
    elif error_hot < -1.0:
        factor = 1.10

    remaining = max(p.total_monomer_target - mM_added, 0.0)
    taper = float(np.clip(remaining / 1.0, 0.0, 1.0))
    factor *= taper

    if disturbance_active(t, cfg) and cfg.disturbance_type == "pulse_feed":
        factor *= 1.0 + cfg.disturbance_magnitude

    return float(np.clip(cfg.feed_rate_nominal * factor, 0.0, 1.20 * cfg.feed_rate_nominal))


def split_range_source_temperature(
    t: float,
    valve_actual: float,
    Tj_out: float,
    cfg: ReactorConfig,
    p: ReactorParams,
) -> float:
    c = float(np.clip(valve_actual, 0.0, 100.0))
    Tinlet = inlet_water_temperature(cfg, p)

    if disturbance_active(t, cfg) and cfg.disturbance_type == "cooling_water":
        Tinlet += cfg.disturbance_magnitude

    if c < 50.0:
        x = (50.0 - c) / 50.0
        alpha = 1.0 - np.exp(-3.0 * x)
        return float(Tj_out + alpha * (Tinlet - Tj_out))

    if c > 50.0:
        x = (c - 50.0) / 50.0
        alpha = 1.0 - np.exp(-2.5 * x)
        return float(Tj_out + alpha * (p.T_steam - Tj_out))

    return float(Tj_out)


def inverse_split_range_valve(
    t: float,
    Tj_target: float,
    Tj_out: float,
    cfg: ReactorConfig,
    p: ReactorParams,
) -> float:
    Tinlet = inlet_water_temperature(cfg, p)

    if disturbance_active(t, cfg) and cfg.disturbance_type == "cooling_water":
        Tinlet += cfg.disturbance_magnitude

    if Tj_target < Tj_out - 0.05:
        denom = Tinlet - Tj_out
        if abs(denom) < 1e-9:
            return 50.0

        alpha = (Tj_target - Tj_out) / denom
        alpha = float(np.clip(alpha, 0.0, 0.95))
        x = -np.log(max(1.0 - alpha, 1e-6)) / 3.0

        return float(np.clip(50.0 * (1.0 - x), 0.0, 50.0))

    if Tj_target > Tj_out + 0.05:
        denom = p.T_steam - Tj_out
        if abs(denom) < 1e-9:
            return 50.0

        alpha = (Tj_target - Tj_out) / denom
        alpha = float(np.clip(alpha, 0.0, 0.95))
        x = -np.log(max(1.0 - alpha, 1e-6)) / 2.5

        return float(np.clip(50.0 * (1.0 + x), 50.0, 100.0))

    return 50.0


def controller_command(
    t: float,
    T: float,
    Tj_out: float,
    Ierr: float,
    mM_added: float,
    cfg: ReactorConfig,
    p: ReactorParams,
    dTdt: float = 0.0,
) -> float:
    """
    Two-mode master controller.

    Heat-up mode:
    - Conservative heating to avoid overshoot.

    Reaction mode:
    - Stronger tracking during feed/hold so temperature remains near setpoint.

    The controller does not impose dT/dt. It only manipulates the split-range valve.
    """

    error = cfg.T_set - T
    I = float(np.clip(Ierr, -cfg.integral_limit, cfg.integral_limit))

    # -------------------------
    # MODE 1: SAFE HEAT-UP
    # -------------------------
    if mM_added <= 1e-6:
        raw_delta_Tj = 3.2 * error - 520.0 * dTdt

        if error > 50.0:
            low, high = -2.0, 9.0
        elif error > 30.0:
            low, high = -5.0, 8.0
        elif error > 15.0:
            low, high = -10.0, 6.0
        elif error > 8.0:
            low, high = -18.0, 4.0
        elif error > 3.0:
            low, high = -30.0, 2.0
        elif error > 0.0:
            low, high = -45.0, 0.8
        else:
            low, high = -70.0, 0.0

        raw_delta_Tj = float(np.clip(raw_delta_Tj, low, high))

        predicted_T_5min = T + dTdt * 300.0
        predicted_T_3min = T + dTdt * 180.0
        predicted_T_1min = T + dTdt * 60.0

        if T > cfg.T_set - 25.0 and predicted_T_5min > cfg.T_set + 0.5:
            raw_delta_Tj = min(raw_delta_Tj, -10.0)

        if T > cfg.T_set - 10.0 and predicted_T_3min > cfg.T_set + 0.3:
            raw_delta_Tj = min(raw_delta_Tj, -25.0)

        if T > cfg.T_set - 2.0 and predicted_T_1min > cfg.T_set + 0.15:
            raw_delta_Tj = min(raw_delta_Tj, -45.0)

        if T > cfg.T_set + 0.2:
            raw_delta_Tj = min(raw_delta_Tj, -65.0)

    # -------------------------
    # MODE 2: FEED/HOLD TRACKING
    # -------------------------
    else:
        raw_delta_Tj = (
            cfg.controller_Kp * error
            + cfg.controller_Ki * I
            - cfg.controller_Kd * dTdt
        )

        raw_delta_Tj = float(np.clip(raw_delta_Tj, -70.0, 45.0))

        # If reactor is cold during feed/hold, allow heating authority.
        if error > 2.0:
            raw_delta_Tj = max(raw_delta_Tj, 12.0)
        elif error > 1.0:
            raw_delta_Tj = max(raw_delta_Tj, 7.0)
        elif error > 0.5:
            raw_delta_Tj = max(raw_delta_Tj, 3.0)

        # If reactor is hot, force cooling.
        if error < -0.3:
            raw_delta_Tj = min(raw_delta_Tj, -15.0)
        if error < -0.8:
            raw_delta_Tj = min(raw_delta_Tj, -35.0)
        if error < -1.2:
            raw_delta_Tj = min(raw_delta_Tj, -55.0)

        # Prevent overheating trend.
        predicted_T_2min = T + dTdt * 120.0
        if predicted_T_2min > cfg.T_set + 0.5:
            raw_delta_Tj = min(raw_delta_Tj, -25.0)

    Tj_target = float(np.clip(T + raw_delta_Tj, 270.0, p.T_steam))
    return inverse_split_range_valve(t, Tj_target, Tj_out, cfg, p)


def heat_capacity_total(mM: float, mP: float, p: ReactorParams) -> float:
    C = max(mM, 0.0) * p.cpM + max(mP, 0.0) * p.cpP + p.mW * p.cpW
    return float(max(C, 1e-9))


def plant_terms(
    t: float,
    mM: float,
    mP: float,
    T: float,
    Tj_out: float,
    Tj_in: float,
    F_actual: float,
    cfg: ReactorConfig,
    p: ReactorParams,
) -> Dict[str, float]:
    mu = viscosity(mP, mM, cfg, p)
    U = heat_transfer_coefficient(mu, cfg, p)
    UA = U * p.area

    r = reaction_rate_kg_s(t, T, mM, mP, cfg, p)

    C_total = heat_capacity_total(mM, mP, p)
    Tj_avg = 0.5 * (Tj_in + Tj_out)

    Q_feed = F_actual * p.cpM * (feed_temperature(t, cfg) - T)
    Q_reaction = (r / p.MW_M) * (-p.deltaH_poly)
    Q_jacket = UA * (Tj_avg - T)
    Q_loss = -p.UA_loss * (T - ambient_temperature(cfg))

    dTdt = (Q_feed + Q_reaction + Q_jacket + Q_loss) / C_total

    return {
        "mu": mu,
        "U": U,
        "UA": UA,
        "reaction_rate_kg_s": r,
        "C_total_kJ_K": C_total,
        "Tj_avg_K": Tj_avg,
        "Q_feed_kW": Q_feed,
        "Q_reaction_kW": Q_reaction,
        "Q_jacket_kW": Q_jacket,
        "Q_loss_kW": Q_loss,
        "dTdt_energy_K_s": dTdt,
    }


def rhs(
    t: float,
    y: np.ndarray,
    cfg: ReactorConfig,
    p: ReactorParams,
    rng: np.random.Generator,
) -> np.ndarray:
    mM, mP, T, Tj_out, Tj_in, valve, mM_added, F_actual, Ierr = y

    mM = max(float(mM), 0.0)
    mP = max(float(mP), 0.0)
    T = float(np.clip(T, 250.0, 470.0))
    Tj_out = float(np.clip(Tj_out, 250.0, 470.0))
    Tj_in = float(np.clip(Tj_in, 250.0, 470.0))
    valve = float(np.clip(valve, 0.0, 100.0))
    mM_added = max(float(mM_added), 0.0)
    F_actual = max(float(F_actual), 0.0)
    Ierr = float(np.clip(Ierr, -cfg.integral_limit, cfg.integral_limit))

    F_cmd = feed_command_kg_s(t, T, mM_added, cfg, p)

    dF = (F_cmd - F_actual) / cfg.feed_tau
    dF = float(np.clip(dF, -cfg.feed_rate_limit, cfg.feed_rate_limit))

    if mM_added >= p.total_monomer_target:
        F_cmd = 0.0
        dF = min(dF, 0.0)

    F_to_reactor = F_actual if mM_added < p.total_monomer_target else 0.0

    terms = plant_terms(
        t=t,
        mM=mM,
        mP=mP,
        T=T,
        Tj_out=Tj_out,
        Tj_in=Tj_in,
        F_actual=F_to_reactor,
        cfg=cfg,
        p=p,
    )

    r = terms["reaction_rate_kg_s"]

    dmM = F_to_reactor - r
    dmP = r
    dmM_added = F_to_reactor if mM_added < p.total_monomer_target else 0.0

    dT = terms["dTdt_energy_K_s"]

    UA = terms["UA"]
    Tj_avg = terms["Tj_avg_K"]

    dTj_out = (
        p.F_C * p.cpC * (Tj_in - Tj_out)
        + UA * (T - Tj_avg)
    ) / (p.mC * p.cpC)

    Tj_source = split_range_source_temperature(t, valve, Tj_out, cfg, p)
    dTj_in = (Tj_source - Tj_in) / p.tau_jacket_in

    valve_cmd = controller_command(
        t=t,
        T=T,
        Tj_out=Tj_out,
        Ierr=Ierr,
        mM_added=mM_added,
        cfg=cfg,
        p=p,
        dTdt=dT,
    )

    if cfg.valve_noise_std > 0.0:
        valve_cmd += rng.normal(0.0, cfg.valve_noise_std)

    valve_cmd = float(np.clip(valve_cmd, 0.0, 100.0))

    dvalve = (valve_cmd - valve) / cfg.valve_tau
    dvalve = float(np.clip(dvalve, -cfg.valve_rate_limit, cfg.valve_rate_limit))

    error = cfg.T_set - T

    if mM_added <= 1e-6 or abs(error) > 5.0:
        dIerr = 0.0
    elif (valve_cmd >= 99.9 and error > 0.0) or (valve_cmd <= 0.1 and error < 0.0):
        dIerr = 0.0
    else:
        dIerr = error

    return np.array(
        [
            dmM,
            dmP,
            dT,
            dTj_out,
            dTj_in,
            dvalve,
            dmM_added,
            dF,
            dIerr,
        ],
        dtype=float,
    )


def simulate_batch(
    cfg: ReactorConfig,
    p: Optional[ReactorParams] = None,
) -> pd.DataFrame:
    if p is None:
        p = ReactorParams()

    rng = np.random.default_rng(cfg.seed)
    Tamb = ambient_temperature(cfg)

    y0 = np.array(
        [
            p.mM0,
            p.mP0,
            Tamb,
            Tamb,
            Tamb,
            50.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    t_eval = np.arange(0.0, cfg.t_final + cfg.dt, cfg.dt)

    Y = np.zeros((len(y0), len(t_eval)), dtype=float)
    Y[:, 0] = y0

    for k in range(1, len(t_eval)):
        t0 = float(t_eval[k - 1])
        h = float(t_eval[k] - t_eval[k - 1])

        y = Y[:, k - 1].copy()

        k1 = rhs(t0, y, cfg, p, rng)
        k2 = rhs(t0 + 0.5 * h, y + 0.5 * h * k1, cfg, p, rng)

        y_next = y + h * k2

        y_next[0] = max(y_next[0], 0.0)
        y_next[1] = max(y_next[1], 0.0)
        y_next[2] = float(np.clip(y_next[2], 250.0, 470.0))
        y_next[3] = float(np.clip(y_next[3], 250.0, 470.0))
        y_next[4] = float(np.clip(y_next[4], 250.0, 470.0))
        y_next[5] = float(np.clip(y_next[5], 0.0, 100.0))
        y_next[6] = max(y_next[6], 0.0)
        y_next[7] = max(y_next[7], 0.0)
        y_next[8] = float(np.clip(y_next[8], -cfg.integral_limit, cfg.integral_limit))

        Y[:, k] = y_next

    df = pd.DataFrame(
        {
            "trajectory_id": cfg.trajectory_id,
            "time_s": t_eval,
            "time_min": t_eval / 60.0,
            "mM_kg": np.maximum(Y[0], 0.0),
            "mP_kg": np.maximum(Y[1], 0.0),
            "T_K": Y[2],
            "Tj_out_K": Y[3],
            "Tj_in_K": Y[4],
            "valve_actual_pct": np.clip(Y[5], 0.0, 100.0),
            "mM_added_kg": np.maximum(Y[6], 0.0),
            "feed_actuator_kg_s": np.maximum(Y[7], 0.0),
            "controller_integral_K_s": Y[8],
        }
    )

    df["feed_rate_kg_s"] = df["feed_actuator_kg_s"]
    df.loc[df["mM_added_kg"] >= p.total_monomer_target, "feed_rate_kg_s"] = 0.0

    rows = []

    for _, row in df.iterrows():
        t = float(row["time_s"])

        terms = plant_terms(
            t=t,
            mM=float(row["mM_kg"]),
            mP=float(row["mP_kg"]),
            T=float(row["T_K"]),
            Tj_out=float(row["Tj_out_K"]),
            Tj_in=float(row["Tj_in_K"]),
            F_actual=float(row["feed_rate_kg_s"]),
            cfg=cfg,
            p=p,
        )

        F_cmd = feed_command_kg_s(
            t=t,
            T=float(row["T_K"]),
            mM_added=float(row["mM_added_kg"]),
            cfg=cfg,
            p=p,
        )

        valve_cmd = controller_command(
            t=t,
            T=float(row["T_K"]),
            Tj_out=float(row["Tj_out_K"]),
            Ierr=float(row["controller_integral_K_s"]),
            mM_added=float(row["mM_added_kg"]),
            cfg=cfg,
            p=p,
            dTdt=terms["dTdt_energy_K_s"],
        )

        Tj_source = split_range_source_temperature(
            t=t,
            valve_actual=float(row["valve_actual_pct"]),
            Tj_out=float(row["Tj_out_K"]),
            cfg=cfg,
            p=p,
        )

        rows.append(
            {
                **terms,
                "viscosity_proxy": terms["mu"],
                "U_kW_m2_K": terms["U"],
                "UA_kW_K": terms["UA"],
                "feed_command_kg_s": F_cmd,
                "valve_command_pct": valve_cmd,
                "Tj_source_K": Tj_source,
                "feed_temperature_K": feed_temperature(t, cfg),
                "ambient_T_K": ambient_temperature(cfg),
                "inlet_water_T_K": inlet_water_temperature(cfg, p),
            }
        )

    extra = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)

    df["dTdt_num_K_s"] = np.gradient(df["T_K"].to_numpy(), df["time_s"].to_numpy())
    df["dTdt_K_min"] = df["dTdt_energy_K_s"] * 60.0

    df["dmMdt_num_kg_s"] = np.gradient(df["mM_kg"].to_numpy(), df["time_s"].to_numpy())
    df["dmPdt_num_kg_s"] = np.gradient(df["mP_kg"].to_numpy(), df["time_s"].to_numpy())

    df["conversion_proxy"] = df["mP_kg"] / (df["mP_kg"] + df["mM_kg"] + 1e-9)

    df["solids_fraction"] = [
        solids_fraction(mp, mm, p)
        for mp, mm in zip(df["mP_kg"], df["mM_kg"])
    ]

    df["recipe_stage"] = "heatup"

    df.loc[
        (df["mM_added_kg"] > 1e-6)
        & (df["mM_added_kg"] < p.total_monomer_target - 1e-3),
        "recipe_stage",
    ] = "feed"

    df.loc[
        (df["mM_added_kg"] >= p.total_monomer_target - 1e-3)
        & (df["feed_rate_kg_s"] <= 1e-6),
        "recipe_stage",
    ] = "hold"

    df.loc[
        (df["mM_added_kg"] > 1e-6)
        & (df["feed_rate_kg_s"] <= 1e-8)
        & (df["mM_added_kg"] < p.total_monomer_target - 1e-3),
        "recipe_stage",
    ] = "feed_pause"

    df["T_set_K"] = cfg.T_set
    df["product"] = cfg.product
    df["batch_number"] = cfg.batch_number
    df["season"] = cfg.season
    df["impurity"] = cfg.impurity
    df["kinetic_multiplier"] = cfg.kinetic_multiplier
    df["heat_transfer_multiplier"] = cfg.heat_transfer_multiplier
    df["fouling_resistance"] = fouling_resistance(cfg)
    df["dataset_split"] = cfg.dataset_split
    df["disturbance_type"] = cfg.disturbance_type
    df["disturbance_magnitude"] = cfg.disturbance_magnitude

    return df


def random_config(
    trajectory_id: int,
    split: str = "train",
    seed: Optional[int] = None,
) -> ReactorConfig:
    rng = np.random.default_rng(seed)

    if split == "baseline":
        product = "A" if trajectory_id % 2 == 0 else "B"
        batch_number = (trajectory_id % 5) + 1
        season = "summer" if (trajectory_id // 5) % 2 == 0 else "winter"

        return ReactorConfig(
            trajectory_id=trajectory_id,
            dataset_split=split,
            product=product,
            batch_number=batch_number,
            season=season,
            impurity=1.0,
            kinetic_multiplier=1.0,
            heat_transfer_multiplier=1.0,
            fouling_multiplier=1.0,
            feed_start=1800.0,
            feed_rate_nominal=0.0028,
            controller_Kp=4.5,
            controller_Ki=0.0010,
            controller_Kd=360.0,
            valve_tau=55.0,
            valve_rate_limit=0.08,
            seed=seed,
        )

    if split in {"train", "val", "test"}:
        season = "winter" if rng.random() < 0.5 else "summer"

        disturbance_type = "none"
        disturbance_mag = 0.0
        disturbance_start = 1e12
        disturbance_duration = 0.0

        if rng.random() < 0.25:
            disturbance_type = str(rng.choice(["feed_temp", "cooling_water", "pulse_feed"]))
            disturbance_start = float(rng.uniform(9000.0, 21000.0))
            disturbance_duration = float(rng.uniform(1200.0, 3600.0))

            if disturbance_type in {"feed_temp", "cooling_water"}:
                disturbance_mag = float(rng.uniform(-6.0, 6.0))
            else:
                disturbance_mag = float(rng.uniform(-0.35, 0.35))

        return ReactorConfig(
            trajectory_id=trajectory_id,
            dataset_split=split,
            product="A" if rng.random() < 0.65 else "B",
            batch_number=int(rng.integers(1, 6)),
            season=season,
            impurity=float(rng.uniform(0.80, 1.20)),
            kinetic_multiplier=float(rng.uniform(0.90, 1.10)),
            heat_transfer_multiplier=float(rng.uniform(0.85, 1.15)),
            utility_temp_bias_K=float(rng.uniform(-3.0, 3.0)),
            fouling_multiplier=float(rng.uniform(0.80, 1.20)),
            feed_start=float(rng.uniform(1500.0, 2400.0)),
            feed_rate_nominal=float(rng.uniform(0.0024, 0.0033)),
            feed_tau=float(rng.uniform(90.0, 150.0)),
            feed_rate_limit=float(rng.uniform(2.0e-5, 4.0e-5)),
            controller_Kp=float(rng.uniform(3.8, 5.2)),
            controller_Ki=float(rng.uniform(0.0007, 0.0014)),
            controller_Kd=float(rng.uniform(280.0, 420.0)),
            valve_tau=float(rng.uniform(45.0, 75.0)),
            valve_rate_limit=float(rng.uniform(0.05, 0.10)),
            valve_noise_std=float(rng.uniform(0.0, 0.03)),
            feed_temp_bias_K=float(rng.uniform(-2.0, 2.0)),
            disturbance_type=disturbance_type,
            disturbance_magnitude=disturbance_mag,
            disturbance_start_s=disturbance_start,
            disturbance_duration_s=disturbance_duration,
            seed=seed,
        )

    season = "summer" if rng.random() < 0.5 else "winter"
    disturbance_type = str(rng.choice(["feed_temp", "cooling_water", "impurity", "pulse_feed"]))

    if disturbance_type in {"feed_temp", "cooling_water"}:
        disturbance_mag = float(rng.choice([-1, 1]) * rng.uniform(6.0, 12.0))
    elif disturbance_type == "impurity":
        disturbance_mag = float(rng.choice([-1, 1]) * rng.uniform(0.10, 0.25))
    else:
        disturbance_mag = float(rng.choice([-1, 1]) * rng.uniform(0.35, 0.60))

    return ReactorConfig(
        trajectory_id=trajectory_id,
        dataset_split=split,
        product="B" if rng.random() < 0.65 else "A",
        batch_number=int(rng.integers(4, 6)),
        season=season,
        impurity=float(rng.uniform(0.75, 1.25)),
        kinetic_multiplier=float(rng.uniform(0.82, 1.18)),
        heat_transfer_multiplier=float(rng.uniform(0.75, 1.05)),
        utility_temp_bias_K=float(rng.uniform(-5.0, 5.0)),
        fouling_multiplier=float(rng.uniform(1.10, 1.60)),
        feed_start=float(rng.uniform(1400.0, 2600.0)),
        feed_rate_nominal=float(rng.uniform(0.0022, 0.0035)),
        controller_Kp=float(rng.uniform(3.5, 5.5)),
        controller_Ki=float(rng.uniform(0.0006, 0.0015)),
        controller_Kd=float(rng.uniform(260.0, 440.0)),
        valve_tau=float(rng.uniform(50.0, 90.0)),
        valve_rate_limit=float(rng.uniform(0.04, 0.09)),
        valve_noise_std=float(rng.uniform(0.0, 0.05)),
        feed_temp_bias_K=float(rng.uniform(-3.0, 3.0)),
        disturbance_type=disturbance_type,
        disturbance_magnitude=disturbance_mag,
        disturbance_start_s=float(rng.uniform(8000.0, 22000.0)),
        disturbance_duration_s=float(rng.uniform(1800.0, 5400.0)),
        seed=seed,
    )


def metadata_from_config(cfg: ReactorConfig) -> Dict[str, object]:
    return asdict(cfg)