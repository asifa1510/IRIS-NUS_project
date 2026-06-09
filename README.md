# 🧬 **Regime Aware Symbolic Discovery of Hidden Physical Laws in Semibatch Polymerization Reactors**

> An interpretable AI-for-Science framework that discovers, validates, and explains hidden physical regimes and their governing symbolic laws in nonlinear dynamical systems, demonstrated on the Chylla-Haase semibatch polymerization reactor.

---

# 🚀 **Executive Summary**

Most scientific machine learning methods assume that a physical system is governed by a single mathematical law throughout its entire operating range.

Whether using:

* Symbolic Regression
* System Identification
* Neural Networks
* Physics-Informed Machine Learning
* Neural ODEs
* Model Predictive Control

the fundamental assumption remains the same:

```text
One System
      ↓
One Governing Law
```

This project challenges that assumption.

We hypothesize that many nonlinear dynamical systems are actually composed of multiple hidden operating regimes, each governed by different physical mechanisms and therefore different symbolic laws.

Instead of forcing one equation to explain the entire system, this project develops an AI framework capable of:

1. Discovering hidden operating regimes directly from data.
2. Validating whether those regimes are physically real.
3. Learning symbolic laws governing each regime.
4. Explaining transitions between regimes.
5. Demonstrating that piecewise symbolic laws provide a better scientific description than a single global law.

The Chylla-Haase semibatch polymerization reactor serves as the primary benchmark system.

---

# ❓ **The Fundamental Scientific Question**

Traditional scientific machine learning asks:

> What equation best fits the data?

This project asks a deeper question:

> Is there really only one equation?

Many nonlinear systems exhibit changing dominant mechanisms throughout operation.

As operating conditions evolve:

* Different variables become important.
* Different physical processes dominate.
* Different dynamics emerge.

This suggests that a single global equation may be an oversimplification.

The goal of this project is to determine whether hidden operating regimes exist and whether each regime obeys its own symbolic physical law.

---

# 🧪 **Why the Chylla-Haase Reactor?**

The Chylla-Haase semibatch polymerization reactor is one of the most widely studied benchmark systems in process systems engineering.

It is ideal for this research because it naturally exhibits:

### **Nonlinear Reaction Kinetics**

Reaction rates change dramatically with temperature and concentration.

### **Dynamic Feed Conditions**

Monomer is continuously added during operation.

### **Strong Thermal Coupling**

Heat generation and heat removal continuously interact.

### **Viscosity Growth**

As polymerization progresses:

```text
Viscosity ↑
```

which affects mixing and heat transfer.

### **Heat Transfer Degradation**

Increasing viscosity and fouling reduce cooling effectiveness.

### **Batch-to-Batch Variability**

Small disturbances can produce significantly different trajectories.

These characteristics create a rich nonlinear system where hidden regimes are likely to emerge.

---

# 🧠 **Core Hypothesis**

## **Traditional View**

```text
One Reactor
      ↓
One Governing Law
      ↓
One Symbolic Equation
```

---

## **Proposed View**

```text
One Reactor
      ↓
Multiple Hidden Regimes
      ↓
Multiple Governing Laws
      ↓
Piecewise Symbolic Dynamics
```

The central claim of this project is:

> A nonlinear dynamical system may be governed by multiple hidden symbolic laws rather than a single global law.

---

# 📌 **Formal Definition of a Regime**

A regime is defined as:

> A coherent region of state-control space in which the local system dynamics are explained by a low-complexity symbolic model significantly better than by a single global symbolic model, and whose structure is recurrent, physically interpretable, and generalizable across trajectories.

This definition is critical because it distinguishes true physical regimes from arbitrary machine-learning clusters.

---

# 🎯 **Research Objectives**

## **Objective 1**

Develop a validated simulation environment for the Chylla-Haase reactor.

---

## **Objective 2**

Generate a large-scale benchmark dataset spanning diverse operating conditions.

---

## **Objective 3**

Automatically discover hidden operating regimes.

---

## **Objective 4**

Validate whether discovered regimes represent genuine physical behavior.

---

## **Objective 5**

Identify symbolic laws governing each regime.

---

## **Objective 6**

Discover interpretable transition rules explaining regime changes.

---

## **Objective 7**

Demonstrate that piecewise symbolic laws outperform global symbolic laws.

---

# 🏗️ **Complete Framework**

```text
Validated Chylla-Haase Simulator
                ↓
Large-Scale Trajectory Generation
                ↓
CH-Regime Dataset
                ↓
Latent Representation Learning
                ↓
Hidden Regime Discovery
                ↓
Regime Reality Test
                ↓
Regime Characterization
                ↓
Piecewise Symbolic Law Discovery
                ↓
Transition Rule Discovery
                ↓
Global vs Piecewise Validation
```

---

# 🧫 **Module 1: Validated Chylla-Haase Simulator**

The first component reproduces a known Chylla-Haase benchmark model from literature.

The simulator captures:

### **Material Dynamics**

* Monomer concentration
* Polymer concentration
* Conversion

### **Energy Dynamics**

* Reactor temperature
* Jacket temperature
* Heat generation
* Heat removal

### **Reaction Kinetics**

* Polymerization rates
* Temperature dependence
* Impurity effects

### **Heat Transfer**

* Cooling effectiveness
* Heat-transfer coefficient degradation
* Fouling

### **Rheology**

* Viscosity growth
* Gel-effect behavior

The simulator must first reproduce published benchmark trajectories before any novel modifications are introduced.

---

# 📚 **Module 2: CH-Regime Dataset**

The simulator is used to generate thousands of trajectories under varying:

* Feed profiles
* Cooling strategies
* Ambient temperatures
* Fouling levels
* Impurity factors
* Initial conditions

Each trajectory records:

```text
Time
Temperature
Conversion
Monomer Mass
Polymer Mass
Reaction Rate
Viscosity
Heat Generation
Heat Removal
Heat Transfer Coefficient
Feed Rate
Coolant Temperature
Ambient Temperature
```

This dataset becomes the foundation of the entire framework.

---

# 🔍 **Module 3: Hidden Regime Discovery**

The framework analyzes trajectories and searches for latent dynamical regimes.

Possible methods:

* Hidden Markov Models
* Bayesian Change Point Detection
* Gaussian Mixture Models
* Switching Dynamical Systems
* Autoencoder-Based Clustering

The goal is not to discover obvious process phases.

The goal is to discover previously unknown dynamical structure.

Potential examples:

```text
Regime A
Reaction-Limited

Regime B
Reaction Acceleration

Regime C
Viscosity-Dominated

Regime D
Heat-Transfer-Limited

Regime E
Fouling-Sensitive
```

These regimes emerge automatically from data.

---

# ✅ **Module 4: Regime Reality Test**

This is one of the most important contributions.

A discovered cluster is not automatically accepted as a regime.

It must pass five validation tests.

---

## **Test 1: Recurrence Test**

The regime must appear across many independent trajectories.

Otherwise it may be noise.

---

## **Test 2: Distinct-Law Test**

The regime must possess a distinct symbolic law.

If two regimes share the same law, they are not truly separate.

---

## **Test 3: Physical-Characteristic Test**

The regime must exhibit distinct physical behavior.

Examples:

* Temperature
* Conversion
* Viscosity
* Reaction rate
* Heat transfer

must differ significantly.

---

## **Test 4: Generalization Test**

The discovered regime must remain valid on unseen operating conditions.

---

## **Test 5: Random Segmentation Test**

The discovered regime must outperform random trajectory splitting.

This prevents arbitrary clustering from appearing meaningful.

---

# 📊 **Module 5: Regime Characterization**

Each validated regime is interpreted physically.

For every regime, the framework computes:

* Mean temperature
* Mean conversion
* Mean reaction rate
* Mean viscosity
* Mean heat-transfer coefficient

The goal is to connect machine learning outputs to real physical phenomena.

---

# 🧩 **Module 6: Piecewise Symbolic Law Discovery**

This is the central scientific contribution.

Traditional symbolic regression:

```text
Entire Dataset
      ↓
One Equation
```

Proposed approach:

```text
Regime A
      ↓
Equation A

Regime B
      ↓
Equation B

Regime C
      ↓
Equation C
```

Each regime receives its own symbolic model.

Potential tools:

* PySR
* SINDy
* Sparse Symbolic Regression

The resulting equations are:

* Human-readable
* Physically interpretable
* Regime-specific

---

# 🔁 **Module 7: Transition Rule Discovery**

The framework then investigates:

> Why do regime transitions occur?

Interpretable rule-learning methods identify transition signatures.

Example:

```text
IF

Conversion > Threshold

AND

Viscosity > Threshold

AND

Heat Transfer Coefficient < Threshold

THEN

Regime C → Regime D
```

These rules provide interpretable explanations for changes in behavior.

---

# ⚖️ **Module 8: Global vs Piecewise Validation**

The central hypothesis is tested through rigorous comparison.

## **Baselines**

### **Global Symbolic Regression**

One equation for the entire dataset.

### **Global Neural Network**

Black-box model.

### **SINDy**

Sparse global dynamics.

### **Manual Phase Segmentation**

Human-defined phases plus symbolic regression.

### **Random Segmentation**

Arbitrary trajectory splitting.

---

## **Proposed Method**

```text
Hidden Regime Discovery
        ↓
Regime Validation
        ↓
Piecewise Symbolic Laws
```

---

## **Evaluation Metrics**

### **Prediction Metrics**

* RMSE
* MAE
* R²

### **Complexity Metrics**

* Equation Length
* Number of Terms

### **Generalization Metrics**

* Unseen Operating Conditions
* Unseen Trajectories

### **Scientific Metrics**

* Regime Recurrence
* Physical Distinctiveness
* Symbolic Separability

---

# 💡 **Novelty**

## **Novelty 1: Challenges the Single-Law Assumption**

Most symbolic modeling assumes one governing equation.

This project explicitly investigates whether multiple hidden laws exist.

---

## **Novelty 2: Discovers and Validates Regimes**

Existing clustering discovers groups.

This project discovers and validates scientific regimes.

---

## **Novelty 3: Regime-Specific Symbolic Physics**

Instead of one equation, multiple localized physical laws are discovered.

---

## **Novelty 4: Discovery of Transition Signatures**

The framework identifies interpretable rules explaining regime changes.

---

## **Novelty 5: Scientific Discovery Rather Than Prediction**

The primary goal is not prediction accuracy.

The primary goal is discovering hidden scientific structure.

---

## **Novelty 6: First Regime-Validation Framework**

The Regime Reality Test introduces a systematic methodology for determining whether discovered regimes correspond to genuine physical behavior.

---

## **Novelty 7: AI-for-Science Contribution**

The framework moves beyond modeling toward automated scientific discovery.

---

# 🏆 **Expected Contributions**

* Hidden Regime Discovery Framework
* Regime Reality Test
* Piecewise Symbolic Law Discovery Framework
* Transition Rule Discovery Framework
* CH-Regime Benchmark Dataset
* Evidence Against Single-Law Modeling Assumptions
* Interpretable AI-for-Science Methodology

---

# 🌍 **Impact**

This work lies at the intersection of:

* AI for Science
* Scientific Machine Learning
* Symbolic AI
* Dynamical Systems
* Explainable AI
* Process Systems Engineering

The ultimate contribution is a framework that does not merely predict system behavior, but discovers, validates, and explains hidden physical structure within nonlinear dynamical systems.
