@echo off
@chcp 65001 >nul
REM ################################################################################
REM # QUICK START SCRIPT - Athletic Training Robustness Analysis (Windows)
REM ################################################################################
REM # This script runs through a complete example workflow from setup to visualization
REM # Author: Training Analytics Team
REM ################################################################################

setlocal enabledelayedexpansion

echo ========================================================================
echo ATHLETIC TRAINING ROBUSTNESS ANALYSIS - QUICK START
echo ========================================================================
echo.

REM Step 1: Check dependencies
echo Step 1: Checking dependencies...
echo ----------------------------------------

REM Check for Python
python --version >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo [OK] Python found: !PYTHON_VERSION!
) else (
    echo [ERROR] Python not found. Please install Python 3.8 or later.
    exit /b 1
)

REM Check for MPI
mpiexec -n 1 hostname >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] MPI found and appears to be working.
    goto mpi_found
) else (
    echo [ERROR] MPI not found or not working. Please install Microsoft MPI.
    echo   Download: https://www.microsoft.com/en-us/download/details.aspx?id=57467
    exit /b 1
)

:mpi_found
echo.

REM Step 2: Install Python packages
echo Step 2: Installing Python packages...
echo ----------------------------------------

python -m pip install --quiet --upgrade mpi4py numpy matplotlib
echo [OK] Python packages installed
echo.

REM Step 3: Create directory structure
echo Step 3: Creating project directories...
echo ----------------------------------------

if not exist "results" mkdir results
if not exist "logs" mkdir logs
if not exist "plots" mkdir plots
echo [OK] Created directories: results\, logs\, plots\
echo.

REM Step 4: Run a quick test with 4 processes
echo Step 4: Running quick test (N=1,000 simulations, 4 MPI ranks)...
echo ----------------------------------------

mpiexec -n 4 python "%~dp0mpi-training-sim.py" ^
    --n-sims 1000 ^
    --training-days 60 ^
    --race-day 59 ^
    --tau1-mean 45.0 ^
    --tau1-std 3.0 ^
    --tau2-mean 12.0 ^
    --tau2-std 2.0 ^
    --threshold 0.8 ^
    --schedule periodized ^
    --output results/quicktest.json

if !errorlevel! neq 0 (
    echo [ERROR] Quick test failed with exit code !errorlevel!
    exit /b !errorlevel!
)

echo.
echo [OK] Quick test completed successfully!
echo.

REM Step 5: Generate visualizations
echo Step 5: Generating visualization plots...
echo ----------------------------------------

python "%~dp0visualize-results.py" results/quicktest.json --output-dir plots/quicktest

if !errorlevel! neq 0 (
    echo [WARNING] Visualization generation may have failed
)

echo.
echo [OK] Plots generated in plots\quicktest\
echo.

REM Step 6: Display summary
echo Step 6: Displaying results summary...
echo ----------------------------------------
echo.

if exist "plots\quicktest\summary_report.txt" (
    type plots\quicktest\summary_report.txt
) else (
    echo [INFO] Summary report not yet available
)

echo.
echo ========================================================================
echo QUICK START COMPLETE!
echo ========================================================================
echo.
echo Generated files:
echo   - results\quicktest.json              (simulation data)
echo   - plots\quicktest\*.png               (visualization plots)
echo   - plots\quicktest\summary_report.txt  (text summary)
echo.

echo ========================================================================

endlocal
