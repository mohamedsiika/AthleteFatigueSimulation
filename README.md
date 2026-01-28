# AthleteFatigueSimulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MPI](https://img.shields.io/badge/MPI-mpi4py-green.svg)](https://mpi4py.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Stochastic Analysis of Athlete Fatigue using Parallelized Monte Carlo Simulations of the Banister Impulse-Response Model**

A high-performance computational framework for prospective risk assessment of athletic training protocols using population-level Monte Carlo simulations. This tool enables coaches and sports scientists to evaluate training schedules **before** implementation, quantifying overtraining risk across athletes with diverse recovery kinetics.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Scientific Background](#scientific-background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Output Files](#output-files)
- [Understanding the Results](#understanding-the-results)
- [System Requirements](#system-requirements)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Traditional Banister model applications use **retrospective parameter estimation** (analyzing historical data from individual athletes). This project introduces a **prospective risk analysis** approach: simulating training outcomes across 1,000 synthetic athletes to quantify the probability that a proposed training schedule will cause overtraining.

**Key Question Answered:** *"What percentage of athletes will this training plan overtrain?"*

### Who Benefits?

- **Coaches & Sports Scientists**: Stress-test training protocols before athlete implementation
- **National Sports Organizations**: Evaluate standardized programs across diverse populations
- **Individual Athletes**: Assess whether published training plans match their recovery characteristics

## ✨ Key Features

- **MPI-Based Parallelization**: 2,677 simulations/second on 4 cores (0.37 seconds for 1,000 iterations)
- **Monte Carlo Simulation**: 1,000 synthetic athletes with stochastic recovery parameters
- **4th-Order Runge-Kutta Integration**: High-precision numerical ODE solving
- **Comprehensive Visualization**: 4 publication-quality plots (training schedule, CDF, risk analysis, performance distribution)
- **Flexible Configuration**: Command-line interface for rapid protocol evaluation
- **No Data Required**: Operates on population-level parameter distributions

## 🔬 Scientific Background

### The Banister Impulse-Response Model

The model conceptualizes performance as two competing exponential processes:

```
dg/dt = -g/τ₁ + w(t)    [Fitness accumulation]
dh/dt = -h/τ₂ + w(t)    [Fatigue accumulation]
p(t) = k₁·g(t) - k₂·h(t) [Performance]
```

Where:
- `τ₁`: Fitness decay time constant (~45 days)
- `τ₂`: Fatigue decay time constant (~12 days)
- `k₁, k₂`: Fitness/fatigue coefficients (1.0, 2.0)
- `w(t)`: Training load impulse

### Stochastic Framework

We sample recovery kinetics from normal distributions:
- `τ₁ ~ N(μ=45.0, σ=3.0)` days
- `τ₂ ~ N(μ=12.0, σ=2.0)` days

This captures biological variability: identical training produces outcomes ranging from severe overtraining to exceptional performance.

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **MPI Implementation**: MPICH, OpenMPI, or MS-MPI
- **Operating System**: Linux, macOS, or Windows

### Step 1: Clone the Repository

```bash
git clone https://github.com/mohamedsiika/AthleteFatigueSimulation.git
cd AthleteFatigueSimulation
```

### Step 2: Install MPI (if not already installed)

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y mpich
```

#### macOS (using Homebrew)
```bash
brew install mpich
```

#### Windows
Download and install [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)

### Step 3: Install Python Dependencies

```bash
pip install numpy scipy matplotlib mpi4py
```

**Or using requirements.txt:**

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
mpiexec --version
python -c "import mpi4py; print(mpi4py.__version__)"
```

## ⚡ Quick Start

### Option 1: Automated Quick Start (Recommended)

**Linux/macOS:**
```bash
chmod +x quickstart-script.sh
./quickstart-script.sh
```

**Windows:**
```bash
quickstart-script.bat
```

This will run a quick test simulation with 100 iterations and generate plots in `plots/quicktest/`.

### Option 2: Manual Execution

**Run the simulation:**
```bash
mpiexec -n 4 python mpi-training-sim.py \
  --n-sims 1000 \
  --training-days 60 \
  --race-day 59 \
  --threshold 10.0 \
  --output-dir results \
  --schedule periodized
```

**Generate visualizations:**
```bash
python visualize-results.py --input results/results.json --output plots
```

## 📖 Detailed Usage

### Main Simulation Script: `mpi-training-sim.py`

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-sims` | int | 1000 | Number of Monte Carlo simulations |
| `--training-days` | int | 60 | Total training duration (days) |
| `--race-day` | int | 59 | Target performance evaluation day |
| `--threshold` | float | 10.0 | Success threshold for performance |
| `--tau1-mean` | float | 45.0 | Fitness decay mean (days) |
| `--tau1-std` | float | 3.0 | Fitness decay standard deviation |
| `--tau2-mean` | float | 12.0 | Fatigue decay mean (days) |
| `--tau2-std` | float | 2.0 | Fatigue decay standard deviation |
| `--k1` | float | 1.0 | Fitness coefficient |
| `--k2` | float | 2.0 | Fatigue coefficient |
| `--output-dir` | str | results | Output directory for results |
| `--schedule` | str | periodized | Training schedule type |

#### Example: Custom Parameter Sweep

Test different taper durations:
```bash
for taper in 0.05 0.10 0.15 0.20; do
  mpiexec -n 4 python mpi-training-sim.py \
    --n-sims 1000 \
    --taper-fraction $taper \
    --output-dir results/taper_$taper
done
```

### Visualization Script: `visualize-results.py`

```bash
python visualize-results.py --input results/results.json --output plots
```

Generates:
1. **training_schedule.png**: 60-day periodized protocol
2. **performance_distribution.png**: Histogram with success/failure split
3. **cdf.png**: Cumulative distribution function
4. **risk_analysis.png**: Success probability vs. threshold curve

### Training Schedule Types

**Periodized (Default):**
- Build Phase (70%): Sinusoidal load variation (20-100 units)
- Peak Phase (20%): Constant high intensity (120 units)
- Taper Phase (10%): Exponential load reduction

## 📊 Output Files

### Results Directory Structure

```
results/
├── results.json          # Simulation results and metadata
├── summary_report.txt    # Human-readable summary
└── raw_data.csv          # Per-simulation performance data
```

### Results JSON Schema

```json
{
  "config": {
    "n_simulations": 1000,
    "training_days": 60,
    "tau1_distribution": {"mean": 45.0, "std": 3.0},
    "tau2_distribution": {"mean": 12.0, "std": 2.0}
  },
  "statistics": {
    "mean_performance": 414.55,
    "std_performance": 329.73,
    "median_performance": 402.61,
    "percentile_5": -111.69,
    "percentile_95": 984.96
  },
  "success_analysis": {
    "success_count": 895,
    "failure_count": 105,
    "success_rate": 0.895
  },
  "computational_performance": {
    "elapsed_time_seconds": 0.37,
    "mpi_ranks": 4,
    "throughput_sims_per_second": 2677
  }
}
```

## 📈 Understanding the Results

### Key Metrics

**Success Rate (89.5%)**: Proportion of athletes achieving performance > threshold

**Standard Deviation (329.73)**: Indicates extreme variability—79.5% of mean performance

**90% Confidence Interval [-111.69, 984.96]**: Range spanning from overtraining to exceptional performance

### Interpretation Guidelines

- **Success Rate < 85%**: High-risk protocol, consider reducing peak phase intensity or extending taper
- **Std Dev > 50% of Mean**: Expect bimodal responder distribution (fast vs. slow recoverers)
- **Negative Performance Values**: Indicate overtraining (fatigue exceeds fitness)




