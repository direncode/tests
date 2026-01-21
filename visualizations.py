"""
Visualization module for HMED experiments.

Generates:
1. Energy landscape plots
2. Trajectory visualizations
3. Convergence curves
4. Ablation comparison charts
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

# We'll use ASCII art for terminal visualization if matplotlib fails
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, using text-based visualization")


def plot_energy_landscape_2d(energy_fn, xlim=(-5, 5), ylim=(-5, 5),
                             resolution=50, filepath=None):
    """Plot 2D energy landscape."""
    if not HAS_MATPLOTLIB:
        return _ascii_energy_landscape(energy_fn, xlim, ylim, resolution)

    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute energy on grid
    points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

    # Pad if energy function expects higher dimensions
    if hasattr(energy_fn, 'latent_dim') and energy_fn.latent_dim > 2:
        padding = torch.zeros(points.shape[0], energy_fn.latent_dim - 2)
        points = torch.cat([points, padding], dim=1)

    with torch.no_grad():
        E = energy_fn(points).numpy()

    E = E.reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, E, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Energy')
    ax.set_xlabel('z_1')
    ax.set_ylabel('z_2')
    ax.set_title('Energy Landscape')

    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def _ascii_energy_landscape(energy_fn, xlim, ylim, resolution=20):
    """ASCII art energy landscape."""
    chars = " .:-=+*#%@"
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)

    points = []
    for yi in y:
        for xi in x:
            points.append([xi, yi])

    points = torch.tensor(points, dtype=torch.float32)
    if hasattr(energy_fn, 'latent_dim') and energy_fn.latent_dim > 2:
        padding = torch.zeros(points.shape[0], energy_fn.latent_dim - 2)
        points = torch.cat([points, padding], dim=1)

    with torch.no_grad():
        E = energy_fn(points).numpy()

    E = E.reshape(resolution, resolution)
    E_norm = (E - E.min()) / (E.max() - E.min() + 1e-8)

    lines = []
    for i in range(resolution - 1, -1, -1):
        line = ""
        for j in range(resolution):
            idx = min(int(E_norm[i, j] * (len(chars) - 1)), len(chars) - 1)
            line += chars[idx]
        lines.append(line)

    return "\n".join(lines)


def plot_trajectory_2d(trajectory: List[torch.Tensor], energy_fn=None,
                       filepath=None, title="Optimization Trajectory"):
    """Plot 2D trajectory on energy landscape."""
    if not HAS_MATPLOTLIB:
        return _ascii_trajectory(trajectory)

    # Extract first two dimensions
    traj = torch.stack(trajectory)
    if traj.dim() == 3:
        traj = traj.mean(dim=1)  # Average over batch
    traj = traj[:, :2].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot energy landscape if provided
    if energy_fn is not None:
        xlim = (traj[:, 0].min() - 1, traj[:, 0].max() + 1)
        ylim = (traj[:, 1].min() - 1, traj[:, 1].max() + 1)

        x = np.linspace(xlim[0], xlim[1], 50)
        y = np.linspace(ylim[0], ylim[1], 50)
        X, Y = np.meshgrid(x, y)

        points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
        if hasattr(energy_fn, 'latent_dim') and energy_fn.latent_dim > 2:
            padding = torch.zeros(points.shape[0], energy_fn.latent_dim - 2)
            points = torch.cat([points, padding], dim=1)

        with torch.no_grad():
            E = energy_fn(points).numpy()
        E = E.reshape(50, 50)

        ax.contourf(X, Y, E, levels=30, cmap='viridis', alpha=0.5)

    # Plot trajectory
    ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='*', label='End', zorder=5)

    ax.set_xlabel('z_1')
    ax.set_ylabel('z_2')
    ax.set_title(title)
    ax.legend()

    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def _ascii_trajectory(trajectory, width=40, height=20):
    """ASCII trajectory visualization."""
    traj = torch.stack(trajectory)
    if traj.dim() == 3:
        traj = traj.mean(dim=1)
    traj = traj[:, :2].numpy()

    # Normalize to grid
    x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
    y_min, y_max = traj[:, 1].min(), traj[:, 1].max()

    grid = [[' ' for _ in range(width)] for _ in range(height)]

    for i, (x, y) in enumerate(traj):
        xi = int((x - x_min) / (x_max - x_min + 1e-8) * (width - 1))
        yi = int((y - y_min) / (y_max - y_min + 1e-8) * (height - 1))
        yi = height - 1 - yi  # Flip y

        if i == 0:
            grid[yi][xi] = 'S'
        elif i == len(traj) - 1:
            grid[yi][xi] = 'E'
        else:
            grid[yi][xi] = '.'

    return "\n".join("".join(row) for row in grid)


def plot_convergence_curves(metrics_dict: Dict[str, List[Dict]],
                            metric_name: str = 'energy',
                            filepath=None, title="Convergence"):
    """Plot convergence curves for multiple methods."""
    if not HAS_MATPLOTLIB:
        return _ascii_convergence(metrics_dict, metric_name)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, metrics in metrics_dict.items():
        values = []
        for m in metrics:
            if metric_name in m:
                values.append(m[metric_name])

        if values:
            ax.plot(values, label=method_name, linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def _ascii_convergence(metrics_dict, metric_name, width=60, height=15):
    """ASCII convergence plot."""
    lines = [f"Convergence: {metric_name}"]
    lines.append("=" * width)

    all_values = []
    for method_name, metrics in metrics_dict.items():
        values = [m.get(metric_name, 0) for m in metrics if metric_name in m]
        if values:
            all_values.append((method_name, values))

    if not all_values:
        return "No data to plot"

    # Find global range
    all_v = [v for _, vals in all_values for v in vals]
    v_min, v_max = min(all_v), max(all_v)

    for method_name, values in all_values:
        line = f"{method_name[:10]:10s}: "
        for v in values[::max(1, len(values)//20)]:  # Sample
            normalized = (v - v_min) / (v_max - v_min + 1e-8)
            char_idx = int(normalized * 7)
            chars = "▁▂▃▄▅▆▇█"
            line += chars[min(char_idx, 7)]
        line += f" ({values[-1]:.3f})"
        lines.append(line)

    return "\n".join(lines)


def plot_ablation_comparison(ablation_results: List[Dict], filepath=None):
    """Plot ablation study comparison."""
    if not HAS_MATPLOTLIB:
        return _text_ablation_table(ablation_results)

    methods = [r['method'] for r in ablation_results]
    energies = [r['metrics'].get('final_energy', 0) for r in ablation_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if 'full' in m else 'gray' for m in methods]
    bars = ax.bar(methods, energies, color=colors)

    ax.set_ylabel('Final Energy')
    ax.set_title('Ablation Study: Final Energy Comparison')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar, val in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def _text_ablation_table(ablation_results):
    """Text-based ablation table."""
    lines = ["Ablation Study Results", "=" * 50]

    for r in ablation_results:
        method = r['method']
        energy = r['metrics'].get('final_energy', 0)
        line = f"{method:20s} | Energy: {energy:8.4f}"

        # Add component flags
        m = r['metrics']
        flags = []
        if m.get('use_modulation', True):
            flags.append("mod")
        if m.get('use_hierarchy', True):
            flags.append("hier")
        if m.get('use_aggregation', True):
            flags.append("agg")

        if flags:
            line += f" | [{', '.join(flags)}]"

        lines.append(line)

    return "\n".join(lines)


def plot_mode_coverage(mode_results: Dict[str, Dict], filepath=None):
    """Plot mode coverage comparison across methods."""
    if not HAS_MATPLOTLIB:
        lines = ["Mode Coverage Comparison", "=" * 40]
        for method, metrics in mode_results.items():
            coverage = metrics.get('mode_coverage', 0)
            bar = "█" * int(coverage * 20)
            lines.append(f"{method:15s} | {bar} {coverage:.2%}")
        return "\n".join(lines)

    methods = list(mode_results.keys())
    coverages = [mode_results[m].get('mode_coverage', 0) for m in methods]
    collapses = [mode_results[m].get('collapse_rate', 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mode coverage
    ax1.bar(methods, coverages, color='steelblue')
    ax1.set_ylabel('Mode Coverage')
    ax1.set_title('Mode Coverage (higher is better)')
    ax1.set_ylim(0, 1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Collapse rate
    ax2.bar(methods, collapses, color='coral')
    ax2.set_ylabel('Collapse Rate')
    ax2.set_title('Collapse Rate (lower is better)')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def generate_all_visualizations(results: List[Dict], output_dir: str):
    """Generate all visualizations from experiment results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")

    # Group results by experiment
    by_experiment = {}
    for r in results:
        exp = r.get('experiment', r.get('experiment_name', 'unknown'))
        if exp not in by_experiment:
            by_experiment[exp] = []
        by_experiment[exp].append(r)

    # Generate plots for each experiment
    for exp_name, exp_results in by_experiment.items():
        # Mode coverage for multimodal
        if 'multimodal' in exp_name:
            mode_results = {}
            for r in exp_results:
                method = r.get('method', r.get('method_name', 'unknown'))
                metrics = r.get('metrics', r)
                mode_results[method] = metrics

            if HAS_MATPLOTLIB:
                plot_mode_coverage(
                    mode_results,
                    filepath=str(output_path / f'{exp_name}_modes.png')
                )
            else:
                print(plot_mode_coverage(mode_results))

        # Ablation comparison
        if 'ablation' in exp_name:
            ablation_data = [
                {'method': r.get('method', r.get('method_name')),
                 'metrics': r.get('metrics', {})}
                for r in exp_results
            ]
            if HAS_MATPLOTLIB:
                plot_ablation_comparison(
                    ablation_data,
                    filepath=str(output_path / f'{exp_name}_comparison.png')
                )
            else:
                print(plot_ablation_comparison(ablation_data))

    print(f"Visualizations saved to {output_dir}/")


def create_summary_report(results: List[Dict], output_path: str):
    """Create a text summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("HMED EXPERIMENT SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Group by experiment
    by_experiment = {}
    for r in results:
        exp = r.get('experiment', r.get('experiment_name', 'unknown'))
        if exp not in by_experiment:
            by_experiment[exp] = []
        by_experiment[exp].append(r)

    for exp_name, exp_results in by_experiment.items():
        lines.append(f"\n{exp_name.upper()}")
        lines.append("-" * 50)

        for r in exp_results:
            method = r.get('method', r.get('method_name', 'unknown'))
            metrics = r.get('metrics', {})

            metric_strs = []
            for k, v in metrics.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, float):
                    metric_strs.append(f"{k}={v:.4f}")
                else:
                    metric_strs.append(f"{k}={v}")

            lines.append(f"  {method}: {', '.join(metric_strs[:5])}")

    report = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    return report
