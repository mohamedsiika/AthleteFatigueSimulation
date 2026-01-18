#!/usr/bin/env python3
"""
Visualization script for MPI Monte Carlo training simulation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_results(filename):
    """Load simulation results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def plot_performance_distribution(performances, threshold, stats, output_dir):
    """Plot histogram of final performance distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(performances, bins=50, density=True, 
                                alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Success Threshold ({threshold:.2f})')
    
    # Add mean line
    ax.axvline(stats['mean'], color='green', linestyle='-', linewidth=2,
               label=f"Mean ({stats['mean']:.2f})")
    
    # Add percentile lines
    ax.axvline(stats['p5'], color='orange', linestyle=':', linewidth=1.5,
               label=f"5th %ile ({stats['p5']:.2f})")
    ax.axvline(stats['p95'], color='orange', linestyle=':', linewidth=1.5,
               label=f"95th %ile ({stats['p95']:.2f})")
    
    # Shade success region
    success_mask = bins >= threshold
    for i, patch in enumerate(patches):
        if bins[i] >= threshold:
            patch.set_facecolor('lightgreen')
    
    ax.set_xlabel('Performance (Race Day)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Race Day Performance\n(Monte Carlo Simulation)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_distribution.png'}")
    plt.close()


def plot_training_schedule(schedule, output_dir):
    """Plot the training load schedule."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    days = np.arange(len(schedule))
    ax.bar(days, schedule, color='steelblue', alpha=0.7, edgecolor='navy')
    
    # Add phase labels
    build_end = int(len(schedule) * 0.7)
    peak_end = int(len(schedule) * 0.9)
    
    ax.axvspan(0, build_end, alpha=0.1, color='blue', label='Build Phase')
    ax.axvspan(build_end, peak_end, alpha=0.1, color='red', label='Peak Phase')
    ax.axvspan(peak_end, len(schedule), alpha=0.1, color='green', label='Taper Phase')
    
    ax.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Load', fontsize=12, fontweight='bold')
    ax.set_title('Periodized Training Schedule', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_schedule.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'training_schedule.png'}")
    plt.close()


def plot_cumulative_distribution(performances, threshold, output_dir):
    """Plot cumulative distribution function of performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_perf = np.sort(performances)
    cumulative = np.arange(1, len(sorted_perf) + 1) / len(sorted_perf)
    
    ax.plot(sorted_perf, cumulative, linewidth=2, color='steelblue')
    
    # Add threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold:.2f})')
    
    # Calculate and annotate probability of success
    prob_success = np.mean(performances > threshold)
    ax.axhline(1 - prob_success, color='green', linestyle=':', linewidth=1.5,
               label=f'P(Success) = {prob_success:.1%}')
    
    ax.set_xlabel('Performance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'cumulative_distribution.png'}")
    plt.close()


def plot_risk_analysis(performances, threshold, output_dir):
    """Plot risk analysis with different threshold levels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds = np.linspace(np.min(performances), np.max(performances), 100)
    success_rates = [np.mean(performances > t) for t in thresholds]
    
    ax.plot(thresholds, success_rates, linewidth=2.5, color='steelblue')
    ax.fill_between(thresholds, success_rates, alpha=0.3, color='steelblue')
    
    # Mark the actual threshold
    actual_success = np.mean(performances > threshold)
    ax.plot(threshold, actual_success, 'ro', markersize=12, 
            label=f'Current: {actual_success:.1%} @ {threshold:.2f}')
    
    # Add reference lines
    for prob in [0.95, 0.90, 0.80, 0.50]:
        idx = np.argmin(np.abs(np.array(success_rates) - prob))
        thresh_val = thresholds[idx]
        ax.axhline(prob, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(np.max(thresholds), prob, f'{prob:.0%}', 
                verticalalignment='center', fontsize=9, color='gray')
    
    ax.set_xlabel('Performance Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability of Success', fontsize=12, fontweight='bold')
    ax.set_title('Risk Analysis: Success Rate vs. Threshold', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'risk_analysis.png'}")
    plt.close()


def create_summary_report(data, output_dir):
    """Create a text summary report."""
    config = data['config']
    results = data['results']
    comp_info = data['computational_info']
    
    report = f"""
{'=' * 80}
ATHLETIC TRAINING ROBUSTNESS ANALYSIS - SUMMARY REPORT
{'=' * 80}

SIMULATION CONFIGURATION
{'-' * 80}
Total Simulations:        {results['total_simulations']:,}
Training Days:            {config['training_days']}
Race Day:                 {config['race_day']}
Training Schedule:        {config['schedule'].title()}

MODEL PARAMETERS
{'-' * 80}
Fitness Decay (τ₁):       μ = {config['tau1_mean']:.1f} days, σ = {config['tau1_std']:.1f} days
Fatigue Decay (τ₂):       μ = {config['tau2_mean']:.1f} days, σ = {config['tau2_std']:.1f} days
Fitness Coefficient (k₁): {config['k1']:.2f}
Fatigue Coefficient (k₂): {config['k2']:.2f}
Success Threshold:        {config['threshold']:.2f}

PERFORMANCE STATISTICS (RACE DAY)
{'-' * 80}
Mean:                     {results['performance_stats']['mean']:>10.4f}
Standard Deviation:       {results['performance_stats']['std']:>10.4f}
Median:                   {results['performance_stats']['median']:>10.4f}
5th Percentile:           {results['performance_stats']['p5']:>10.4f}
95th Percentile:          {results['performance_stats']['p95']:>10.4f}

ROBUSTNESS ANALYSIS
{'-' * 80}
Success Count:            {results['success_count']:>10,} / {results['total_simulations']:,}
Probability of Success:   {results['probability_success']:>10.2%}
Expected Failure Rate:    {1 - results['probability_success']:>10.2%}

COMPUTATIONAL PERFORMANCE
{'-' * 80}
MPI Ranks:                {comp_info['mpi_ranks']}
Elapsed Time:             {comp_info['elapsed_time_seconds']:.2f} seconds
Throughput:               {comp_info['throughput_sims_per_sec']:,.0f} sims/second

INTERPRETATION
{'-' * 80}
This training schedule has a {results['probability_success']:.1%} probability of producing
performance above the threshold of {config['threshold']:.2f} on race day.

The 90% confidence interval for performance is:
[{results['performance_stats']['p5']:.3f}, {results['performance_stats']['p95']:.3f}]

{'=' * 80}
    """
    
    report_file = output_dir / 'summary_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: {report_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Visualize MPI simulation results')
    parser.add_argument('input_file', type=str, help='Input JSON file with results')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 80)
    
    # Load results
    print(f"Loading results from: {args.input_file}")
    data = load_results(args.input_file)
    
    config = data['config']
    results = data['results']
    performances = np.array(results['all_performances'])
    schedule = np.array(results['training_schedule'])
    stats = results['performance_stats']
    
    print(f"Found {len(performances):,} simulation results")
    print()
    
    # Generate plots
    plot_performance_distribution(performances, config['threshold'], stats, output_dir)
    plot_training_schedule(schedule, output_dir)
    plot_cumulative_distribution(performances, config['threshold'], output_dir)
    plot_risk_analysis(performances, config['threshold'], output_dir)
    
    # Create summary report
    print()
    report = create_summary_report(data, output_dir)
    print()
    print(report)
    
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
