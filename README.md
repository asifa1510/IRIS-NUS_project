# Symbolic-Regime-Discovery-for-Decision-Critical-Chemical-Process-Optimization

An interpretable AI framework that discovers symbolic decision-critical operating regimes in chemical processes, identifying when optimal control actions must change and validating these regimes using optimization performance, safety, and regret.
---

## Contribution 1 (Keep)

### Decision-Critical Regime Definition

A regime is defined by a change in optimal operating decision rather than only a change in process dynamics. This remains your primary novelty.

---

## Contribution 2 (Add)

### Counterfactual Decision Explanation Layer

For every discovered symbolic regime, generate:

- Current optimal action
- Alternative action
- Expected future consequence

Example:

```text
IF T > 369 K
AND coolant < 0.4

Optimal action:
Increase coolant

Counterfactual:
If coolant is not increased

Expected regret = 24.6
Violation probability = 81%
```

No title change needed.

---

## Contribution 3 (Add)

### Multi-Horizon Decision Regret

Instead of:

- 1-step regret

evaluate:

- 1-step
- 10-step
- 50-step

This is much more publishable.

Most papers stop at immediate regret.

---

## Contribution 4 (Add)

### Regime Stability Analysis

A new metric:

#### Regime Stability Index (RSI)

Measures:

- How long a discovered regime remains decision-optimal before another policy switch becomes necessary.

Example:

```text
Production regime:
RSI = 180 sec

Cooling regime:
RSI = 45 sec

Safety regime:
RSI = 12 sec
```

This is relatively unexplored in symbolic regime discovery.

---

## Contribution 5 (Add)

### Robust Symbolic Regimes

Test discovered rules under:

- Sensor noise
- Feed disturbances
- Cooling degradation
- Actuator delay

and define:

#### Regime Robustness Score

---

## Contribution 6 (Most Important)

### Add a new optimization criterion.

Current DART-Opt:

- RMSE
- + Regret
- + Safety
- + Complexity

Upgrade it to:

- RMSE
- + Multi-Horizon Regret
- + Safety Risk
- + Counterfactual Risk
- + Stability
- + Complexity

This becomes your paper's unique framework.
