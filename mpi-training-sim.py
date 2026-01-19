#!/usr/bin/env python3
"""
Robustness Analysis of Athletic Training Loads: MPI Implementation
Author: Scientific Computing Team
Description: Massively parallel Monte Carlo simulation of the Banister
             Fitness-Fatigue model with stochastic recovery parameters.
"""

import numpy as np
from mpi4py import MPI
import time
import json
import argparse


class BanisterModel:
    """
    Banister Fitness-Fatigue Model with RK4 numerical integration.
    
    Model equations:
        dg/dt = -g/tau1 + w(t)  [Fitness dynamics]
        dh/dt = -h/tau2 + w(t)  [Fatigue dynamics]
        p(t) = k1*g(t) - k2*h(t) [Performance]
    """
    
    def __init__(self, tau1, tau2, k1=1.0, k2=2.0):
        self.tau1 = tau1  # Fitness decay time constant
        self.tau2 = tau2  # Fatigue decay time constant
        self.k1 = k1      # Fitness gain coefficient
        self.k2 = k2      # Fatigue penalty coefficient
    
    def rk4_step(self, g, h, w, dt=1.0):
        """
        4th-order Runge-Kutta integration step.
        
        Args:
            g: Current fitness level
            h: Current fatigue level
            w: Training load (input)
            dt: Time step (default: 1 day)
        
        Returns:
            (g_new, h_new): Updated state
        """
        # Define derivative functions
        def dg_dt(g_val, w_val):
            return -g_val / self.tau1 + w_val
        
        def dh_dt(h_val, w_val):
            return -h_val / self.tau2 + w_val
        
        # RK4 coefficients for fitness (g)
        k1_g = dg_dt(g, w)
        k2_g = dg_dt(g + 0.5 * dt * k1_g, w)
        k3_g = dg_dt(g + 0.5 * dt * k2_g, w)
        k4_g = dg_dt(g + dt * k3_g, w)
        
        # RK4 coefficients for fatigue (h)
        k1_h = dh_dt(h, w)
        k2_h = dh_dt(h + 0.5 * dt * k1_h, w)
        k3_h = dh_dt(h + 0.5 * dt * k2_h, w)
        k4_h = dh_dt(h + dt * k3_h, w)
        
        # Update state
        g_new = g + (dt / 6.0) * (k1_g + 2*k2_g + 2*k3_g + k4_g)
        h_new = h + (dt / 6.0) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
        
        return g_new, h_new
    
    def simulate(self, training_schedule, g0=0.0, h0=0.0, dt=1.0):
        """
        Simulate the full training period.
        
        Args:
            training_schedule: Array of daily training loads
            g0: Initial fitness (default: 0)
            h0: Initial fatigue (default: 0)
            dt: Time step in days
        
        Returns:
            trajectory: Dictionary with daily g, h, performance values
        """
        g, h = g0, h0
        trajectory = {
            'fitness': [g],
            'fatigue': [h],
            'performance': [self.k1 * g - self.k2 * h],
            'load': [0.0]  # Day 0 has no load
        }
        
        for w in training_schedule:
            g, h = self.rk4_step(g, h, w, dt)
            performance = self.k1 * g - self.k2 * h
            
            trajectory['fitness'].append(g)
            trajectory['fatigue'].append(h)
            trajectory['performance'].append(performance)
            trajectory['load'].append(w)
        
        return trajectory


def generate_training_schedule(n_days, schedule_type='periodized'):
    """
    Generate a training schedule.
    
    Args:
        n_days: Total number of training days
        schedule_type: 'periodized', 'constant', or 'high_low'
    
    Returns:
        schedule: Array of daily training loads
    """
    if schedule_type == 'periodized':
        # Build phase (70%) -> Peak phase (20%) -> Taper (10%)
        schedule = []
        build_days = int(n_days * 0.7)
        peak_days = int(n_days * 0.2)
        taper_days = n_days - build_days - peak_days
        
        # Build phase: weekly periodization
        for day in range(build_days):
            week_progress = (day % 7) / 7.0
            load = 100 * (0.6 + 0.4 * np.sin(week_progress * 2 * np.pi))
            schedule.append(load)
        
        # Peak phase: sustained high load
        schedule.extend([120] * peak_days)
        
        # Taper phase: exponential reduction
        for day in range(taper_days):
            taper_progress = day / taper_days
            load = 120 * (1 - 0.7 * taper_progress)
            schedule.append(load)
        
        return np.array(schedule)
    
    elif schedule_type == 'constant':
        return np.full(n_days, 100.0)
    
    elif schedule_type == 'high_low':
        # Alternating high/low days
        return np.array([120 if i % 2 == 0 else 60 for i in range(n_days)])
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def run_monte_carlo_worker(training_schedule, n_local_sims, tau1_mean, tau1_std,
                           tau2_mean, tau2_std, k1, k2, race_day, threshold, rank):
    """
    Worker function to run Monte Carlo simulations on a single MPI rank.
    
    Args:
        training_schedule: Training load schedule
        n_local_sims: Number of simulations for this worker
        tau1_mean, tau1_std: Fitness decay parameters (mean, std dev)
        tau2_mean, tau2_std: Fatigue decay parameters (mean, std dev)
        k1, k2: Performance coefficients
        race_day: Day to evaluate performance
        threshold: Success threshold for performance
        rank: MPI rank ID
    
    Returns:
        local_results: Dictionary with simulation results
    """
    np.random.seed(rank * 12345)  # Ensure different random streams per rank
    
    success_count = 0
    final_performances = []
    
    # Store sample trajectories for visualization (every 100th simulation)
    sample_trajectories = []
    
    for sim in range(n_local_sims):
        # Sample stochastic parameters from Gaussian distributions
        tau1 = np.random.normal(tau1_mean, tau1_std)
        tau2 = np.random.normal(tau2_mean, tau2_std)
        
        # Enforce physical constraints (decay times must be positive)
        tau1 = max(tau1, 20.0)  # Minimum 20 days
        tau2 = max(tau2, 5.0)   # Minimum 5 days
        
        # Create model and simulate
        model = BanisterModel(tau1, tau2, k1, k2)
        trajectory = model.simulate(training_schedule)
        
        # Evaluate performance on race day
        race_performance = trajectory['performance'][race_day]
        final_performances.append(race_performance)
        
        if race_performance > threshold:
            success_count += 1
        
        # Store sample trajectory
        if sim % 100 == 0:
            sample_trajectories.append({
                'tau1': tau1,
                'tau2': tau2,
                'performance': trajectory['performance']
            })
    
    return {
        'success_count': success_count,
        'final_performances': final_performances,
        'sample_trajectories': sample_trajectories,
        'n_local_sims': n_local_sims
    }


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parse command-line arguments (only on rank 0)
    if rank == 0:
        parser = argparse.ArgumentParser(description='MPI Monte Carlo Training Simulator')
        parser.add_argument('--n-sims', type=int, default=100000,
                          help='Total number of Monte Carlo simulations')
        parser.add_argument('--training-days', type=int, default=90,
                          help='Number of training days')
        parser.add_argument('--race-day', type=int, default=89,
                          help='Race day (0-indexed)')
        parser.add_argument('--tau1-mean', type=float, default=45.0,
                          help='Mean fitness decay time (days)')
        parser.add_argument('--tau1-std', type=float, default=3.0,
                          help='Std dev of fitness decay time')
        parser.add_argument('--tau2-mean', type=float, default=12.0,
                          help='Mean fatigue decay time (days)')
        parser.add_argument('--tau2-std', type=float, default=3.0,
                          help='Std dev of fatigue decay time')
        parser.add_argument('--k1', type=float, default=1.0,
                          help='Fitness gain coefficient')
        parser.add_argument('--k2', type=float, default=2.0,
                          help='Fatigue penalty coefficient')
        parser.add_argument('--threshold', type=float, default=0.8,
                          help='Performance threshold for success')
        parser.add_argument('--schedule', type=str, default='periodized',
                          choices=['periodized', 'constant', 'high_low'],
                          help='Training schedule type')
        parser.add_argument('--output', type=str, default='results.json',
                          help='Output file for results')
        
        args = parser.parse_args()
        config = vars(args)
    else:
        config = None
    
    # Broadcast configuration to all ranks
    config = comm.bcast(config, root=0)
    
    # Generate training schedule (same on all ranks)
    training_schedule = generate_training_schedule(
        config['training_days'],
        config['schedule']
    )
    
    # Distribute work across MPI ranks
    n_total_sims = config['n_sims']
    n_local_sims = n_total_sims // size
    remainder = n_total_sims % size
    
    # Give extra simulations to first few ranks
    if rank < remainder:
        n_local_sims += 1
    
    if rank == 0:
        print("=" * 80)
        print("ATHLETIC TRAINING ROBUSTNESS ANALYSIS - MPI MONTE CARLO")
        print("=" * 80)
        print(f"Total simulations: {n_total_sims:,}")
        print(f"MPI processes: {size}")
        print(f"Training days: {config['training_days']}")
        print(f"Race day: {config['race_day']}")
        print(f"tau1 ~ N({config['tau1_mean']:.1f}, {config['tau1_std']:.1f})")
        print(f"tau2 ~ N({config['tau2_mean']:.1f}, {config['tau2_std']:.1f})")
        print(f"Success threshold: p > {config['threshold']:.2f}")
        print("=" * 80)
        print(f"Starting simulation on {size} MPI ranks...")
        start_time = time.time()
    
    # Each rank runs its portion of simulations
    local_results = run_monte_carlo_worker(
        training_schedule,
        n_local_sims,
        config['tau1_mean'],
        config['tau1_std'],
        config['tau2_mean'],
        config['tau2_std'],
        config['k1'],
        config['k2'],
        config['race_day'],
        config['threshold'],
        rank
    )
    
    # Gather results from all ranks to rank 0
    all_results = comm.gather(local_results, root=0)
    
    # Rank 0 aggregates and analyzes results
    if rank == 0:
        elapsed_time = time.time() - start_time
        
        # Aggregate success counts
        total_success = sum(r['success_count'] for r in all_results)
        total_sims = sum(r['n_local_sims'] for r in all_results)
        probability_success = total_success / total_sims
        
        # Aggregate all final performances
        all_performances = []
        for r in all_results:
            all_performances.extend(r['final_performances'])
        all_performances = np.array(all_performances)
        
        # Calculate statistics
        perf_mean = np.mean(all_performances)
        perf_std = np.std(all_performances)
        perf_median = np.median(all_performances)
        perf_p5 = np.percentile(all_performances, 5)
        perf_p95 = np.percentile(all_performances, 95)
        
        # Print results
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Throughput: {total_sims / elapsed_time:,.0f} simulations/second")
        print(f"Speedup per rank: {total_sims / elapsed_time / size:,.0f} sims/sec/rank")
        print("\n" + "-" * 80)
        print("PERFORMANCE STATISTICS (Race Day)")
        print("-" * 80)
        print(f"Mean:               {perf_mean:>10.4f}")
        print(f"Std Dev:            {perf_std:>10.4f}")
        print(f"Median:             {perf_median:>10.4f}")
        print(f"5th percentile:     {perf_p5:>10.4f}")
        print(f"95th percentile:    {perf_p95:>10.4f}")
        print("\n" + "-" * 80)
        print("ROBUSTNESS ANALYSIS")
        print("-" * 80)
        print(f"Success count:      {total_success:>10,} / {total_sims:,}")
        print(f"Probability(Success): {probability_success:>8.2%}")
        print(f"Expected failures:  {(1-probability_success):>8.2%}")
        print("=" * 80)
        
        # Save results to JSON
        output_data = {
            'config': config,
            'results': {
                'total_simulations': int(total_sims),
                'success_count': int(total_success),
                'probability_success': float(probability_success),
                'performance_stats': {
                    'mean': float(perf_mean),
                    'std': float(perf_std),
                    'median': float(perf_median),
                    'p5': float(perf_p5),
                    'p95': float(perf_p95)
                },
                'all_performances': all_performances.tolist(),
                'training_schedule': training_schedule.tolist()
            },
            'computational_info': {
                'mpi_ranks': size,
                'elapsed_time_seconds': elapsed_time,
                'throughput_sims_per_sec': total_sims / elapsed_time
            }
        }
        
        with open(config['output'], 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {config['output']}")
        print("\nTo visualize results, run: python visualize_results.py {config['output']}")


if __name__ == '__main__':
    main()
