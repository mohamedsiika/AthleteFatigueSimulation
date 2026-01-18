#!/bin/bash
################################################################################
# QUICK START SCRIPT - Athletic Training Robustness Analysis
################################################################################
# This script runs through a complete example workflow from setup to visualization
# Author: Training Analytics Team
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "ATHLETIC TRAINING ROBUSTNESS ANALYSIS - QUICK START"
echo "========================================================================"
echo ""

# Step 1: Check dependencies
echo "Step 1: Checking dependencies..."
echo "----------------------------------------"

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ ERROR: Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check for MPI
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version | head -n 1)
    echo "✓ MPI found: $MPI_VERSION"
else
    echo "✗ ERROR: MPI not found. Please install OpenMPI or MPICH."
    echo "  Ubuntu/Debian: sudo apt-get install openmpi-bin"
    echo "  macOS: brew install open-mpi"
    exit 1
fi

echo ""

# Step 2: Install Python packages
echo "Step 2: Installing Python packages..."
echo "----------------------------------------"

pip3 install --quiet --upgrade mpi4py numpy matplotlib 2>&1 | grep -v "Requirement already satisfied" || true
echo "✓ Python packages installed"
echo ""

# Step 3: Create directory structure
echo "Step 3: Creating project directories..."
echo "----------------------------------------"

mkdir -p results logs plots
echo "✓ Created directories: results/, logs/, plots/"
echo ""

# Step 4: Run a quick test with 4 processes
echo "Step 4: Running quick test (N=1,000 simulations, 4 MPI ranks)..."
echo "----------------------------------------"

mpirun -np 4 python3 mpi_training_sim.py \
    --n-sims 1000 \
    --training-days 60 \
    --race-day 59 \
    --tau1-mean 45.0 \
    --tau1-std 3.0 \
    --tau2-mean 12.0 \
    --tau2-std 2.0 \
    --threshold 0.8 \
    --schedule periodized \
    --output results/quicktest.json

echo ""
echo "✓ Quick test completed successfully!"
echo ""

# Step 5: Generate visualizations
echo "Step 5: Generating visualization plots..."
echo "----------------------------------------"

python3 visualize_results.py results/quicktest.json --output-dir plots/quicktest

echo ""
echo "✓ Plots generated in plots/quicktest/"
echo ""

# Step 6: Display summary
echo "Step 6: Displaying results summary..."
echo "----------------------------------------"
echo ""

cat plots/quicktest/summary_report.txt

echo ""
echo "========================================================================"
echo "QUICK START COMPLETE! ✓"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - results/quicktest.json           (simulation data)"
echo "  - plots/quicktest/*.png            (visualization plots)"
echo "  - plots/quicktest/summary_report.txt  (text summary)"
echo ""
echo "Next steps:"
echo "  1. Run larger simulation:"
echo "     mpirun -np 8 python3 mpi_training_sim.py --n-sims 100000"
echo ""
echo "  2. Compare training schedules:"
echo "     ./compare_schedules.sh"
echo ""
echo "  3. Submit to HPC cluster:"
echo "     sbatch submit_job.sh"
echo ""
echo "  4. Read the full documentation:"
echo "     cat SETUP_GUIDE.md"
echo ""
echo "========================================================================"
