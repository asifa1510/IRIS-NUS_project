# Physics-Informed Hidden Regime Discovery & Local Symbolic Law Learning

**Scientific Machine Learning for Nonlinear Dynamical Systems**

Research conducted at the **National University of Singapore (NUS)** under the
supervision of **Assoc. Prof. Lakshminarayanan Samavedham**, Department of
Chemical & Biomolecular Engineering.

**Asifa S** · NUS IRIS Research  
`Scientific ML` · `Mamba` · `Time-Series` · `Regime Discovery` · `Symbolic Regression`

---

## Overview

Nonlinear reactors exhibit different dynamics across operating stages, making a
single global model difficult to interpret and potentially inaccurate.

This project develops a **physics-informed AI framework** that:

1. generates physically consistent semi-batch reactor trajectories,
2. learns temporal representations using a **Mamba-based sequence model**,
3. discovers hidden operating regimes using **Gaussian Mixture Models (GMM)**,
4. learns **interpretable symbolic laws** separately within each regime.

> **Core idea:** Discover when the system dynamics change, then learn the
> governing behaviour locally.

---

## Framework

<img width="1200" height="420" alt="image" src="https://github.com/user-attachments/assets/90bacce9-6dc9-4666-ab3f-e604d949981f" />

```text
Physics-Based Reactor Simulation
              ↓
       64-Step Windows
              ↓
   Physics-Informed Mamba
              ↓
     Temporal Embeddings
              ↓
             GMM
              ↓
   Hidden Regimes R0–R7
              ↓
     Symbolic Regression
              ↓
 Interpretable Local Laws
```

The reactor simulation incorporates **mass and energy balances, reaction heat,
cooling, viscosity, and valve dynamics**. Temporal representations capture the
evolution of coupled reactor variables before unsupervised regime discovery.

---

## Hidden Regime Discovery

<img width="625" height="347" alt="image" src="https://github.com/user-attachments/assets/137d41c0-5cfa-4ceb-84d1-d6467b7f5dec" />

The framework discovers **8 latent regimes (R0–R7)** without manually supplied
regime labels.

The regimes exhibit distinct physical fingerprints across:

- viscosity
- heat-transfer coefficient \(U\)
- feed rate
- reaction heat
- jacket heat
- temperature-change rate

This suggests that the learned clusters correspond to physically distinguishable
reactor behaviours rather than arbitrary latent groups.

---

## Symbolic Law Learning

Instead of fitting one equation across the complete reactor trajectory, symbolic
regression learns a separate interpretable law for each discovered regime:

```text
Global approach:     Entire trajectory → One symbolic law

Proposed approach:   R0 → Local law
                     R1 → Local law
                     ...
                     R7 → Local law
```

This combines **temporal representation learning** with **interpretable
mathematical modelling**.

---

## Results

<img width="581" height="420" alt="image" src="https://github.com/user-attachments/assets/3f780bc2-0d9d-4da7-8cfb-10780b895574" />

Regime-conditioned symbolic laws outperform a single global symbolic model:

| Evaluation | Improvement |
|---|---:|
| **Test RMSE** | **43.4% lower** |
| **OOD RMSE** | **17.6% lower** |

The results indicate that the discovered regimes are not only physically
meaningful but also improve prediction of reactor temperature-change dynamics.

---

## Key Contributions

- **Physics-consistent reactor simulation** for nonlinear semi-batch polymerization.
- **Physics-informed Mamba** for temporal representation learning.
- **Unsupervised discovery of 8 hidden operating regimes**.
- **Physical interpretation** of discovered regime fingerprints.
- **Regime-wise symbolic regression** for interpretable local dynamics.
- Improved **test and out-of-distribution generalization** over a global symbolic law.

---

## Research Pipeline

```text
Reactor Physics
      ↓
Simulation
      ↓
Temporal Modelling
      ↓
Mamba Representations
      ↓
GMM Regime Discovery
      ↓
Physical Interpretation
      ↓
Symbolic Regression
      ↓
Local Dynamical Laws
      ↓
Test + OOD Evaluation
```

---

## Tech Stack

`Python` · `PyTorch` · `Mamba` · `Gaussian Mixture Models` ·
`Symbolic Regression` · `Scientific Machine Learning` · `Time-Series Analysis`

---

## Research Context

**Researcher:** Asifa S  
**Project Advisor:** Assoc. Prof. Lakshminarayanan Samavedham  
**Institution:** National University of Singapore  
**Department:** Chemical & Biomolecular Engineering  
**Programme:** IRIS Research

---

## Repository

```text
IRIS-NUS_project/
├── README.md
├── assets/
│   ├── framework.png
│   ├── hidden_regimes.png
│   └── symbolic_law_results.png
├── src/
├── notebooks/
├── results/
└── requirements.txt
```

Large datasets, checkpoints, or research materials subject to institutional
restrictions are not included in the public repository.

---

## Author

**Asifa S**

[GitHub](https://github.com/asifa1510) ·
[Portfolio](https://asifa1510.github.io/portfolio/) ·
[LinkedIn](https://www.linkedin.com/in/s-asifa-896741250/)

---

*Research conducted at the National University of Singapore (NUS).*
