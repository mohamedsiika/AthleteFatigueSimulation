# Athletic Training Robustness Analysis - Complete Setup Guide

## 📋 Prerequisites

### Software Requirements
- **Python 3.8+**
- **MPI Implementation**: OpenMPI or MPICH
- **Python Packages**:
  - `mpi4py` (MPI interface for Python)
  - `numpy` (numerical computing)
  - `matplotlib` (visualization)

### Hardware
- Multi-core workstation (minimum: 4 cores)
- HPC cluster (recommended for N ≥ 100,000 simulations)

---

## 🚀 Installation

### Step 1: Install MPI

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

**macOS (with Homebrew):**
```bash
brew install open-mpi
```

**HPC Cluster:**
```bash
module load openmpi  # or mpich, intel-mpi
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv training_env
source training_env/bin/activate  # On Windows: training_env\Scripts\activate

# Install packages
pip install mpi4py numpy matplotlib
```

**Verify installation:**
```bash
mpirun -np 4 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.rank} ready')"
```

You should see output from 4 processes.

---

## 📁 Project Structure

```
training_robustness/
├── mpi_training_sim.py       # Main MPI simulation code
├── visualize_results.py      # Visualization script
├── submit_job.sh             # SLURM job submission script
├── README.md                 # This file
├── results/                  # Output directory (created automatically)
│   ├── results.json
│   └── plots/
└── environments/             # Environment files for HPC
    └── requirements.txt
```

---

## 🏃 Running the Simulation

### Local Machine (Quick Test)

Run a small test with 4 MPI processes:

```bash
mpirun -np 4 python mpi_training_sim.py \
    --n-sims 10000 \
    --training-days 90 \
    --race-day 89 \
    --tau1-mean 45.0 \
    --tau1-std 3.0 \
    --tau2-mean 12.0 \
    --tau2-std 3.0 \
    --threshold 0.8 \
    --schedule periodized \
    --output results/test_run.json
```

**Expected output:**
```
================================================================================
ATHLETIC TRAINING ROBUSTNESS ANALYSIS - MPI MONTE CARLO
================================================================================
Total simulations: 10,000
MPI processes: 4
Training days: 90
Race day: 89
τ₁ ~ N(45.0, 3.0²)
τ₂ ~ N(12.0, 3.0²)
Success threshold: p > 0.80
================================================================================
Starting simulation on 4 MPI ranks...
```

### Full-Scale Run (Workstation)

For N = 100,000 simulations on an 8-core machine:

```bash
mpirun -np 8 python mpi_training_sim.py \
    --n-sims 100000 \
    --output results/full_run.json
```

### HPC Cluster (SLURM)

Create `submit_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=training_robust
#SBATCH --nodes=4                  # Number of compute nodes
#SBATCH --ntasks-per-node=16       # MPI tasks per node (= cores per node)
#SBATCH --time=02:00:00            # Max runtime (HH:MM:SS)
#SBATCH --partition=compute        # Partition/queue name
#SBATCH --output=logs/job_%j.out   # Standard output log
#SBATCH --error=logs/job_%j.err    # Standard error log

# Load required modules
module purge
module load python/3.9
module load openmpi/4.1.1

# Activate virtual environment
source ~/training_env/bin/activate

# Run simulation
# Total MPI ranks = nodes × ntasks-per-node = 4 × 16 = 64
mpirun python mpi_training_sim.py \
    --n-sims 1000000 \
    --training-days 120 \
    --race-day 119 \
    --tau1-mean 45.0 \
    --tau1-std 5.0 \
    --tau2-mean 12.0 \
    --tau2-std 2.0 \
    --threshold 1.0 \
    --schedule periodized \
    --output results/hpc_run_${SLURM_JOB_ID}.json

echo "Job completed at $(date)"
```

Submit the job:
```bash
mkdir -p logs results
sbatch submit_job.sh
```

Monitor job:
```bash
squeue -u $USER           # Check job status
tail -f logs/job_*.out    # Watch output in real-time
```

---

## 📊 Visualizing Results

After the simulation completes:

```bash
python visualize_results.py results/full_run.json --output-dir results/plots
```

**Generated outputs:**
- `performance_distribution.png` - Histogram with success threshold
- `training_schedule.png` - Daily training load visualization
- `cumulative_distribution.png` - CDF of performance
- `risk_analysis.png` - Success rate vs. threshold curve
- `summary_report.txt` - Text summary of results

---

## 🔧 Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-sims` | 100000 | Total number of Monte Carlo simulations |
| `--training-days` | 90 | Number of training days in the plan |
| `--race-day` | 89 | Day to evaluate performance (0-indexed) |
| `--tau1-mean` | 45.0 | Mean fitness decay time constant (days) |
| `--tau1-std` | 3.0 | Std dev of fitness decay time |
| `--tau2-mean` | 12.0 | Mean fatigue decay time constant (days) |
| `--tau2-std` | 3.0 | Std dev of fatigue decay time |
| `--k1` | 1.0 | Fitness gain coefficient |
| `--k2` | 2.0 | Fatigue penalty coefficient |
| `--threshold` | 0.8 | Performance threshold for "success" |
| `--schedule` | periodized | Training schedule type (`periodized`, `constant`, `high_low`) |
| `--output` | results.json | Output JSON file path |

---

## 📈 Example Analysis Workflow

### 1. Compare Different Training Schedules

```bash
# Periodized training
mpirun -np 8 python mpi_training_sim.py --n-sims 50000 \
    --schedule periodized --output results/periodized.json

# Constant load
mpirun -np 8 python mpi_training_sim.py --n-sims 50000 \
    --schedule constant --output results/constant.json

# High-low alternating
mpirun -np 8 python mpi_training_sim.py --n-sims 50000 \
    --schedule high_low --output results/high_low.json
```

### 2. Sensitivity Analysis on Noise Level

```bash
for sigma in 1 2 3 5 8; do
    mpirun -np 8 python mpi_training_sim.py --n-sims 50000 \
        --tau1-std $sigma --tau2-std $sigma \
        --output results/sigma_${sigma}.json
done
```

### 3. Find Optimal Threshold

```bash
# Run simulation once
mpirun -np 8 python mpi_training_sim.py --n-sims 100000 \
    --output results/threshold_analysis.json

# Then analyze with visualization script to see success rate vs threshold curve
python visualize_results.py results/threshold_analysis.json
```

---

## 🐛 Troubleshooting

### Issue: "ImportError: No module named mpi4py"
**Solution:** Install in the correct Python environment
```bash
pip install --upgrade mpi4py
# Or for cluster: pip install --user mpi4py
```

### Issue: Slow performance / No speedup
**Diagnosis:** Check if MPI is actually using multiple processes
```bash
# Add this debug line to your script temporarily:
if rank == 0:
    print(f"Running on {size} MPI ranks")
```

**Solution:** Ensure you're using `mpirun -np N` where N > 1

### Issue: "SLURM job fails to start"
**Check:**
```bash
scontrol show job <job_id>  # Detailed job info
sacct -j <job_id>           # Accounting info
```

---

## 📊 Performance Benchmarks

Approximate throughput on typical hardware:

| Hardware | MPI Ranks | Simulations/Second |
|----------|-----------|-------------------|
| Laptop (4 cores) | 4 | ~2,000 |
| Workstation (16 cores) | 16 | ~10,000 |
| HPC Node (64 cores) | 64 | ~40,000 |
| HPC Cluster (256 cores) | 256 | ~150,000 |

**Example:** 1 million simulations on 256 cores takes ~7 seconds

---

## 🎯 Next Steps

1. **Validate with Real Data:** Fit the model to actual athlete training logs
2. **Add More Features:**
   - Time-varying noise (stress periods)
   - Correlated τ₁ and τ₂ parameters
   - Multi-objective optimization
3. **Scale to GPU:** Port to CUDA for 100x speedup
4. **Build Web Interface:** Create dashboard for coaches

---

## 📚 References

- Banister, E.W. (1991). "Modeling elite athletic performance"
- Busso, T. (2003). "Variable dose-response relationship between exercise training and performance"

---

## 📧 Support

For questions or issues, create a GitHub issue or contact the development team.

**Happy Training! 🏃‍♂️💪**
