# Hybrid ML-Based Predictive Auto-Scaling System for Cloud Computing Resources

> **Final-Year Project** | University  
> **Author:** Anthony  
> **Dataset:** NASA HTTP Web Server Log (August 1995) + Google Cluster Trace  
> **Stack:** Python · pandas · scikit-learn · matplotlib · (AWS Free Tier – optional)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Datasets](#4-datasets)
5. [Module 1 — Data Preprocessing](#5-module-1--data-preprocessing)
6. [Module 2 — Reactive Auto-Scaler (Baseline)](#6-module-2--reactive-auto-scaler-baseline)
7. [Simulator Parameters Reference](#7-simulator-parameters-reference)
8. [Output Columns Reference](#8-output-columns-reference)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Key Concepts Explained](#10-key-concepts-explained)
11. [Current Results (Reactive Baseline)](#11-current-results-reactive-baseline)
12. [Roadmap — Next Modules](#12-roadmap--next-modules)
13. [How to Run](#13-how-to-run)
14. [Dependencies](#14-dependencies)

---

## 1. Project Overview

This project builds and evaluates a **hybrid cloud auto-scaling system** that combines three approaches:

| Layer | Approach | Description |
|---|---|---|
| **Baseline** | Reactive Scaling | Responds to demand *after* it is observed |
| **Layer 2** | Predictive Scaling (ML) | Forecasts future demand and scales *ahead of time* |
| **Layer 3** | Hybrid Decision Engine | Combines reactive + predictive to get the best of both |

**Why this matters in real cloud systems:**

Reactive scaling always reacts too late — by the time an alarm fires and a new instance boots, the traffic spike has already violated your SLA. Predictive scaling solves this but requires an accurate forecast. The hybrid engine uses ML predictions when confidence is high and falls back to reactive logic when uncertainty is high.

---

## 2. System Architecture

```
Raw Log Data
     │
     ▼
┌─────────────────────┐
│  Data Preprocessing  │  ← preprocess_nasa.ipynb
│  (aggregation,       │
│   normalization)     │
└────────┬────────────┘
         │ nasa_workload_clean_5min.csv
         │ google_demand_5min_clean.csv
         ▼
┌─────────────────────┐
│  Reactive Scaler     │  ← reactive_autoscaler.ipynb  [DONE]
│  (Baseline)          │
└────────┬────────────┘
         │ results_df (metrics)
         ▼
┌─────────────────────┐
│  Predictive Scaler   │  ← predictive_autoscaler.ipynb  [TODO]
│  (ML Model)          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Hybrid Engine       │  ← hybrid_engine.ipynb  [TODO]
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Evaluation &        │  ← evaluation.ipynb  [TODO]
│  Comparison          │
└─────────────────────┘
```

---

## 3. Repository Structure

```
project-root/
│
├── Data/
│   ├── raw/                          # Original log files (not committed)
│   └── processed/
│       ├── nasa_workload_clean_5min.csv
│       └── google_demand_5min_clean.csv
│
├── Notebooks/
│   ├── preprocess_nasa.ipynb         # Data cleaning & feature engineering
│   ├── reactive_autoscaler.ipynb     # Baseline simulator [DONE]
│   ├── predictive_autoscaler.ipynb   # ML-based scaler [TODO]
│   ├── hybrid_engine.ipynb           # Combined decision engine [TODO]
│   └── evaluation.ipynb             # Cross-system comparison [TODO]
│
├── Results/
│   └── reactive_baseline_results.csv # Saved simulation outputs
│
├── README.md
└── requirements.txt
```

---

## 4. Datasets

### NASA HTTP Web Server Log

| Property | Value |
|---|---|
| File | `Data/processed/nasa_workload_clean_5min.csv` |
| Source | NASA Kennedy Space Center web server, August 1995 |
| Time range | 1995-08-01 → 1995-08-31 |
| Granularity | 5-minute intervals |
| Rows | 8,928 |

**Columns:**

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Start of 5-minute window |
| `requests` | float | Total HTTP requests in the window |
| `demand` | float | Normalised load score (0–100 scale) |

> `demand` is derived from `requests` by scaling against the peak request count in the dataset. A value of 100 means the system is at peak historical load.

---

### Google Cluster Trace

| Property | Value |
|---|---|
| File | `Data/processed/google_demand_5min_clean.csv` |
| Source | Google cluster CPU usage trace (2011) |
| Granularity | 5-minute intervals |

**Columns:**

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Start of 5-minute window |
| `demand` | float | Normalised CPU demand score (0–100 scale) |

---

## 5. Module 1 — Data Preprocessing

**File:** `preprocess_nasa.ipynb`

### What it does

Takes raw NASA HTTP log data and produces a clean, regularly-spaced time series at 5-minute intervals.

### Processing Steps

| Step | Logic | Problem it solves | What breaks if removed | Real cloud equivalent |
|---|---|---|---|---|
| **Load raw logs** | Read raw access log, parse timestamps | Unstructured log → structured table | Nothing to work with | CloudWatch log ingestion |
| **Aggregate to 5-min bins** | Group requests by 5-minute window, count per bin | Log has per-request rows; simulator needs intervals | Simulator would have millions of rows | CloudWatch metric resolution |
| **Handle missing intervals** | Forward-fill or zero-fill gaps | Gaps in time series break time-based ML models | Model sees irregular time steps, features misalign | CloudWatch gap-filling |
| **Normalise demand (0–100)** | `demand = (requests / max_requests) * 100` | Raw request counts are dataset-specific; normalisation enables cross-dataset comparison | Metrics are not comparable across NASA vs Google datasets | Percentage-of-capacity metric |
| **Save to CSV** | Export cleaned data | Downstream notebooks need a stable input | Every notebook would need to re-run preprocessing | S3 / data lake storage |

---

## 6. Module 2 — Reactive Auto-Scaler (Baseline)

**File:** `reactive_autoscaler.ipynb`

### Concept

The reactive scaler is the **"naïve" baseline** — it observes the current demand at each timestep and immediately adjusts the number of running instances to match. It has **zero look-ahead**: it never anticipates what demand will be in the next step.

This is how most basic cloud auto-scaling works (e.g. AWS Auto Scaling with a simple CPU alarm). It serves as the benchmark that the ML-based predictive system must beat.

### Step-by-Step Logic

#### Step 1 — Configuration (`params` dict)

```python
params = {
    "capacity_per_instance": 100,
    "target_utilization": 0.60,
    "provisioning_delay_steps": 2,
    "cooldown_steps": 2,
    "min_instances": 1,
    "max_instances": 20,
    "initial_instances": 1,
    "cost_per_instance_step": 1.0
}
```

| Parameter | Meaning | Real Cloud Equivalent |
|---|---|---|
| `capacity_per_instance` | Max demand units one instance can handle | EC2 instance vCPU / memory capacity |
| `target_utilization` | Keep utilisation at or below this fraction (e.g. 60%) | AWS Target Tracking policy target (e.g. 60% CPU) |
| `provisioning_delay_steps` | How many timesteps it takes for a new instance to become active | EC2 boot time (~2–5 minutes) |
| `cooldown_steps` | Minimum steps between two scale-out actions | AWS ASG cooldown period |
| `min_instances` | Floor on running instances | ASG minimum capacity |
| `max_instances` | Ceiling on running instances | ASG maximum capacity |
| `initial_instances` | Instances active at time zero | Initial desired capacity |
| `cost_per_instance_step` | Cost unit per instance per timestep | EC2 per-hour billing |

---

#### Step 2 — Per-Timestep Simulation Loop

For each row in the dataset, the simulator runs this decision sequence:

```
Read demand → Compute capacity → Compute utilization
→ Compute desired instances → Clamp to bounds
→ Apply scaling decision immediately
→ Recalculate capacity & utilization post-scale
→ Check SLA → Compute cost → Append to results
```

---

#### Step 3 — Key Calculations

| Code / Logic | Problem it solves | What breaks if removed | Real cloud equivalent |
|---|---|---|---|
| `capacity = instances * capacity_per_instance` | Translates instance count into load units | No way to know if demand exceeds supply | Total available throughput (e.g. total vCPUs) |
| `utilization = demand / capacity` | Measures how loaded the system is (0.0–1.0+) | Cannot tell if over/under-provisioned | CloudWatch CPUUtilization metric |
| `desired = ceil(demand / (capacity_per_instance * target_utilization))` | Computes exact instance count needed to keep utilisation at target | No scaling target — instances never change | AWS Target Tracking desired count formula |
| `desired = max(min_instances, min(desired, max_instances))` | Clamps desired count within safe bounds | Could scale to 0 (outage) or to infinity (cost explosion) | ASG min/max capacity constraints |
| `instances = desired` (immediate) | Applies the scale decision for this timestep | Instances would never change | "Ideal" reactive scaling with zero boot delay |
| `sla_violation = 1 if demand > capacity else 0` | Flags timesteps where demand exceeded what we could serve | No way to measure service quality | SLA breach / 5xx error rate |
| `unserved_demand = max(0.0, demand - capacity)` | Quantifies how much demand was dropped | Can't calculate error rate or impact | Dropped requests / throttled traffic |
| `step_cost = instances * cost_per_instance_step` | Running cost for this timestep | No cost tracking for comparison | AWS Cost Explorer per-hour EC2 cost |
| `cumulative_cost += step_cost` | Running total cost over simulation | Cannot compare total cost across strategies | Monthly bill |

---

#### Step 4 — Results DataFrame

Each timestep produces one row in `results_df`:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Time of this step |
| `demand` | float | Normalised demand (0–100) |
| `instances` | int | Active instances this step |
| `capacity` | int | Total capacity this step |
| `utilization` | float | `demand / capacity` (can exceed 1.0 if overloaded) |
| `desired_instances` | int | What the scaler *wanted* |
| `sla_violation` | int | 1 = SLA breached, 0 = OK |
| `unserved_demand` | float | Demand that could not be served |
| `step_cost` | float | Cost this timestep |
| `cumulative_cost` | float | Running total cost |

---

## 7. Simulator Parameters Reference

> **How to tune these for your experiments:**

| Parameter | Lower value effect | Higher value effect |
|---|---|---|
| `target_utilization` | More instances provisioned (safer, more expensive) | Fewer instances (cheaper, more SLA risk) |
| `provisioning_delay_steps` | Faster reaction | More SLA violations during spikes (realistic) |
| `cooldown_steps` | More responsive to rapid changes | Less thrashing / over-scaling |
| `capacity_per_instance` | More instances needed (granular scaling) | Fewer, larger instances needed |

---

## 8. Output Columns Reference

| Column | Formula | Interpretation |
|---|---|---|
| `utilization` | `demand / capacity` | < 1.0 = healthy; > 1.0 = overloaded |
| `sla_violation` | `1 if demand > capacity` | Rate = fraction of timesteps with service degradation |
| `unserved_demand` | `max(0, demand - capacity)` | Total "dropped load" — lower is better |
| `step_cost` | `instances × cost_per_instance_step` | Infrastructure spend per timestep |

---

## 9. Evaluation Metrics

These metrics are used to compare **Reactive vs Predictive vs Hybrid**:

| Metric | Formula | Goal |
|---|---|---|
| **SLA Violation Rate** | `mean(sla_violation)` | Minimise |
| **Average Utilization** | `mean(utilization)` | Maximise (closer to `target_utilization`) |
| **Total Cost** | `cumulative_cost[-1]` | Minimise |
| **Scaling Actions** | Count of timesteps where `instances` changed | Minimise (stability) |
| **Prediction Error (MAE)** | `mean(|predicted_demand - actual_demand|)` | Minimise (predictive module only) |

---

## 10. Key Concepts Explained

### Why `target_utilization = 0.60`?

If you set target utilisation to 100%, you have zero headroom. A sudden spike will immediately breach SLA before a new instance can launch. 60% is a common production default (used by AWS) — it gives a 40% buffer to absorb brief demand spikes without SLA violation.

### Why `ceil()` for desired instances?

```python
desired = math.ceil(demand / (capacity_per_instance * target_utilization))
```

You always round **up** — you'd rather have one extra instance (slightly over-provisioned) than one fewer (SLA breach). In production, the cost of a missed SLA almost always exceeds the cost of one extra instance.

### What is "provisioning delay"?

In real cloud systems, a new instance doesn't serve traffic the moment you click "launch." It needs 1–5 minutes to boot, initialise, and pass health checks. The `provisioning_delay_steps` parameter models this — a scale-out action triggered now won't help until 2 timesteps (10 minutes) later. This is **the core weakness of reactive scaling** and why predictive scaling is valuable.

### What is "cooldown"?

Rapid scaling decisions (scaling out, then immediately back in) waste cost and can destabilise systems. A cooldown period prevents a new scale action until enough time has passed since the last one. AWS Auto Scaling enforces this by default (300s).

---

## 11. Current Results (Reactive Baseline)

> Simulated on NASA dataset (August 1995, 8,928 timesteps × 5-min intervals = 31 days)

| Metric | Value |
|---|---|
| SLA Violation Rate | **0.0** (0%) |
| Average Utilization | **~25%** |
| Total Cost | **9,395 cost-units** |
| Unique Instance Counts | **2** |

**Observation:** The reactive scaler achieves 0% SLA violations but at the cost of very low average utilisation (~25%). This means the system is heavily over-provisioned — it is spending money on idle capacity. The predictive and hybrid modules aim to reduce cost while maintaining or improving the SLA rate.

---

## 12. Roadmap — Next Modules

| Module | Notebook | Status | Description |
|---|---|---|---|
| Preprocessing | `preprocess_nasa.ipynb` | ✅ Done | Clean datasets ready |
| Reactive Baseline | `reactive_autoscaler.ipynb` | ✅ Done | Baseline metrics established |
| Predictive Scaler | `predictive_autoscaler.ipynb` | 🔲 Next | Train ML model (e.g. Linear Regression / LSTM) on demand; generate `predicted_demand` per timestep; use prediction to scale ahead of time |
| Hybrid Engine | `hybrid_engine.ipynb` | 🔲 Pending | Decision logic: use ML prediction if confidence > threshold, else fall back to reactive |
| Evaluation | `evaluation.ipynb` | 🔲 Pending | Side-by-side comparison of all three strategies on both datasets |

---

## 13. How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter notebook Notebooks/preprocess_nasa.ipynb
jupyter notebook Notebooks/reactive_autoscaler.ipynb
```

> **Note:** The datasets are already processed and stored in `Data/processed/`. You can skip the preprocessing step and run the autoscaler notebooks directly.

---

## 14. Dependencies

```
pandas
numpy
matplotlib
scikit-learn
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

---

## References

- NASA Kennedy Space Center HTTP Log Dataset — [clarknet.edu mirror](http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html)
- Google Cluster Usage Traces — [Google Research](https://research.google/tools/datasets/google-cluster-workload-traces/)
- AWS Auto Scaling Target Tracking Policies — [AWS Documentation](https://docs.aws.amazon.com/autoscaling/ec2/userguide/as-scaling-target-tracking.html)

---

*This project is part of a final-year undergraduate computer science dissertation.*
