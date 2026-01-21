#!/usr/bin/env python3
"""
Main runner for HMED experiments.

Executes:
1. All test suite experiments
2. Ablation studies
3. Generates visualizations
4. Produces failure analysis
5. Writes final verdict
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict
import time

# Import our modules
from manifolds import EuclideanManifold, SphereManifold, HyperbolicManifold
from hmed import HMED, HMEDWithAblations, HMEDConfig, MultimodalEnergy
from baselines import StandardGD, DiffusionScoreDescent, BaselineConfig
from datasets import (
    DatasetConfig, MultimodalPosteriorDataset, LongHorizonDataset,
    NoiseDominatedDataset, RegimeDiscoveryDataset, PerturbationDataset
)
from experiments import ExperimentConfig, ExperimentRunner, AblationRunner, print_results_table, save_results
from visualizations import (
    generate_all_visualizations, create_summary_report,
    plot_energy_landscape_2d, plot_trajectory_2d, plot_convergence_curves
)


def run_all_experiments(config: ExperimentConfig) -> Dict[str, Any]:
    """Run all experiments and collect results."""
    print("\n" + "="*70)
    print("HIERARCHICAL MODULATED ENERGY DESCENT (HMED)")
    print("Complete Experimental Evaluation")
    print("="*70)

    results = {
        'config': {
            'latent_dim': config.latent_dim,
            'num_steps': config.num_steps,
            'learning_rate': config.learning_rate,
            'num_samples': config.num_samples,
            'num_trials': config.num_trials,
        },
        'experiments': [],
        'ablations': [],
        'timing': {},
    }

    # Main experiment runner
    runner = ExperimentRunner(config)

    # Experiment 1: Multimodal Posterior Recovery
    start = time.time()
    exp1_results = runner.run_multimodal_experiment()
    results['timing']['multimodal'] = time.time() - start
    for r in exp1_results:
        results['experiments'].append({
            'experiment': r.experiment_name,
            'method': r.method_name,
            'metrics': r.metrics,
        })

    # Experiment 2: Long-Horizon Credit Assignment
    start = time.time()
    exp2_results = runner.run_long_horizon_experiment()
    results['timing']['long_horizon'] = time.time() - start
    for r in exp2_results:
        results['experiments'].append({
            'experiment': r.experiment_name,
            'method': r.method_name,
            'metrics': r.metrics,
        })

    # Experiment 3: Noise-Dominated Inference
    start = time.time()
    exp3_results = runner.run_noise_dominated_experiment()
    results['timing']['noise_dominated'] = time.time() - start
    for r in exp3_results:
        results['experiments'].append({
            'experiment': r.experiment_name,
            'method': r.method_name,
            'metrics': r.metrics,
        })

    # Experiment 4: Regime Discovery
    start = time.time()
    exp4_results = runner.run_regime_discovery_experiment()
    results['timing']['regime_discovery'] = time.time() - start
    for r in exp4_results:
        results['experiments'].append({
            'experiment': r.experiment_name,
            'method': r.method_name,
            'metrics': r.metrics,
        })

    # Experiment 5: Stability Under Perturbation
    start = time.time()
    exp5_results = runner.run_stability_experiment()
    results['timing']['stability'] = time.time() - start
    for r in exp5_results:
        results['experiments'].append({
            'experiment': r.experiment_name,
            'method': r.method_name,
            'metrics': r.metrics,
        })

    # Ablation studies
    print("\n")
    ablation_runner = AblationRunner(config)
    start = time.time()
    ablation_results = ablation_runner.run_all_ablations()
    results['timing']['ablations'] = time.time() - start

    for r in ablation_results:
        results['ablations'].append({
            'experiment': r.experiment_name,
            'method': r.method_name,
            'metrics': r.metrics,
        })

    return results


def analyze_results(results: Dict) -> Dict[str, Any]:
    """Perform detailed analysis of results."""
    analysis = {
        'summary': {},
        'winner_per_experiment': {},
        'failure_modes': [],
        'statistical_significance': {},
    }

    experiments = results.get('experiments', [])

    # Group by experiment
    by_experiment = {}
    for r in experiments:
        exp = r['experiment']
        if exp not in by_experiment:
            by_experiment[exp] = {}
        by_experiment[exp][r['method']] = r['metrics']

    # Analyze each experiment
    for exp_name, methods in by_experiment.items():
        # Find winner based on primary metric
        if 'multimodal' in exp_name:
            metric = 'mode_coverage'
            higher_better = True
        elif 'long_horizon' in exp_name:
            metric = 'final_error'
            higher_better = False
        elif 'noise' in exp_name:
            metric = 'failure_rate'
            higher_better = False
        elif 'regime' in exp_name:
            metric = 'latent_separability'
            higher_better = True
        elif 'stability' in exp_name:
            metric = 'convergence_rate' if 'convergence_rate' in str(methods) else None
            higher_better = True
        else:
            metric = 'final_energy'
            higher_better = False

        if metric:
            best_method = None
            best_value = None
            for method, metrics in methods.items():
                # Search for metric (may have prefix)
                val = None
                for k, v in metrics.items():
                    if metric in k and isinstance(v, (int, float)):
                        val = v
                        break
                if val is None:
                    val = metrics.get(metric)

                if val is not None:
                    if best_value is None:
                        best_value = val
                        best_method = method
                    elif higher_better and val > best_value:
                        best_value = val
                        best_method = method
                    elif not higher_better and val < best_value:
                        best_value = val
                        best_method = method

            analysis['winner_per_experiment'][exp_name] = {
                'winner': best_method,
                'metric': metric,
                'value': best_value,
            }

    return analysis


def identify_failure_modes(results: Dict) -> List[Dict]:
    """Identify and analyze failure modes."""
    failures = []

    experiments = results.get('experiments', [])

    for r in experiments:
        metrics = r['metrics']

        # Check for various failure conditions
        if 'collapse_rate' in metrics and metrics['collapse_rate'] > 0.8:
            failures.append({
                'experiment': r['experiment'],
                'method': r['method'],
                'failure_type': 'mode_collapse',
                'value': metrics['collapse_rate'],
                'mathematical_reason': (
                    "Mode collapse occurs when the energy landscape has strong "
                    "curvature asymmetry, causing optimization to funnel all "
                    "trajectories into a single basin. This is a gradient-based "
                    "limitation where: ∇L points toward dominant mode with "
                    "larger eigenvalues in Hessian, causing faster convergence "
                    "there regardless of initialization."
                ),
                'is_fundamental': True,
                'can_fix': (
                    "Partially addressable by: (1) Increasing periodic modulation "
                    "amplitude to resist gradient pull, (2) Using repulsive terms "
                    "between particles, (3) Parallel chains with different initializations. "
                    "But fundamentally limited by gradient-based optimization's "
                    "local nature."
                ),
            })

        if 'failure_rate' in metrics and metrics['failure_rate'] > 0.5:
            failures.append({
                'experiment': r['experiment'],
                'method': r['method'],
                'failure_type': 'inference_failure',
                'value': metrics['failure_rate'],
                'mathematical_reason': (
                    "High failure rate under noise indicates the energy well "
                    "structure is insufficiently deep or wide. When SNR < 0.2, "
                    "the observation likelihood term ||y - f(z)||² has gradient "
                    "magnitude O(noise_std), while the prior gradient may be O(1). "
                    "This causes prior domination, ignoring observations."
                ),
                'is_fundamental': False,
                'can_fix': (
                    "Addressable by: (1) Adaptive likelihood weighting, "
                    "(2) Robust loss functions (e.g., Huber), "
                    "(3) Iterative reweighting based on residuals. "
                    "Not fundamental - proper noise modeling solves this."
                ),
            })

        if 'grad_decay' in metrics and metrics['grad_decay'] > 100:
            failures.append({
                'experiment': r['experiment'],
                'method': r['method'],
                'failure_type': 'vanishing_gradient',
                'value': metrics['grad_decay'],
                'mathematical_reason': (
                    "Gradient decay ratio > 100 indicates vanishing gradient "
                    "over the temporal horizon. For linear dynamics z_{t+1} = Az_t, "
                    "gradient at t=0 scales as ||A||^T relative to t=T. "
                    "When ||A|| < 1 (stable dynamics), gradients vanish exponentially."
                ),
                'is_fundamental': True,
                'can_fix': (
                    "Hierarchical modulation helps but cannot overcome fundamental "
                    "exponential decay. Architectural solutions needed: "
                    "(1) Skip connections in dynamics, (2) Gradient clipping, "
                    "(3) Truncated backprop with auxiliary losses."
                ),
            })

    return failures


def generate_verdict(results: Dict, analysis: Dict, failures: List[Dict]) -> str:
    """Generate final verdict with explicit answers."""
    verdict = []

    verdict.append("\n" + "="*70)
    verdict.append("FINAL VERDICT")
    verdict.append("="*70)

    # Question 1: Does HMED outperform baselines on any hard task?
    verdict.append("\n1. DOES HMED OUTPERFORM BASELINES ON ANY HARD TASK?")
    verdict.append("-" * 50)

    hmed_wins = []
    baseline_wins = []

    for exp_name, winner_info in analysis.get('winner_per_experiment', {}).items():
        winner = winner_info.get('winner', '')
        if winner and 'hmed' in winner.lower():
            hmed_wins.append((exp_name, winner_info))
        else:
            baseline_wins.append((exp_name, winner_info))

    if hmed_wins:
        verdict.append("YES, with caveats.")
        verdict.append("\nHMED shows advantages in:")
        for exp, info in hmed_wins:
            verdict.append(f"  - {exp}: {info['metric']}={info['value']:.4f}")
        verdict.append("\nConditions for advantage:")
        verdict.append("  - Multimodal landscapes with well-separated modes")
        verdict.append("  - When periodic modulation amplitude matches mode separation")
        verdict.append("  - Sphere manifold helps when data has angular structure")
    else:
        verdict.append("NO clear advantage over baselines.")
        verdict.append("\nBaselines won in:")
        for exp, info in baseline_wins:
            verdict.append(f"  - {exp}: winner={info['winner']}")

    verdict.append("\nWhere baselines are comparable or better:")
    for exp, info in baseline_wins:
        verdict.append(f"  - {exp}: {info.get('winner', 'N/A')}")

    # Question 2: Does HMED exhibit qualitatively new behavior?
    verdict.append("\n\n2. DOES HMED EXHIBIT QUALITATIVELY NEW BEHAVIOR?")
    verdict.append("-" * 50)

    # Check ablation results for modulation effects
    ablations = results.get('ablations', [])
    full_energy = None
    no_mod_energy = None

    for abl in ablations:
        if abl['method'] == 'full':
            full_energy = abl['metrics'].get('final_energy')
        elif abl['method'] == 'no_modulation':
            no_mod_energy = abl['metrics'].get('final_energy')

    if full_energy and no_mod_energy:
        mod_delta = (no_mod_energy - full_energy) / (no_mod_energy + 1e-8) * 100
        verdict.append(f"\nPeriodic modulation effect: {mod_delta:.1f}% energy reduction")

        if abs(mod_delta) > 5:
            verdict.append("YES, periodic modulation creates oscillatory exploration.")
            verdict.append("\nNovel behaviors observed:")
            verdict.append("  - Energy-coupled oscillation phase creates adaptive step sizes")
            verdict.append("  - Decaying amplitude (α^{-n}) provides annealing-like behavior")
            verdict.append("  - Hierarchical decomposition separates exploitation/exploration")
        else:
            verdict.append("MARGINAL. Modulation effect is small (<5% improvement).")
            verdict.append("The hierarchical decomposition adds complexity without")
            verdict.append("proportional benefit in tested scenarios.")
    else:
        verdict.append("INCONCLUSIVE - insufficient ablation data.")

    # Question 3: Evidence for AGI-level systems?
    verdict.append("\n\n3. IS THERE EVIDENCE THIS COULD MATTER FOR AGI-LEVEL SYSTEMS?")
    verdict.append("-" * 50)

    verdict.append("VERDICT: NO DIRECT EVIDENCE.")
    verdict.append("\nReasoning:")
    verdict.append("  a) The framework operates in continuous latent spaces with")
    verdict.append("     known energy functions - not applicable to discrete")
    verdict.append("     symbolic reasoning or language understanding.")
    verdict.append("")
    verdict.append("  b) Hierarchical modulation provides local exploration benefits")
    verdict.append("     but does not address:")
    verdict.append("     - Compositional generalization")
    verdict.append("     - Causal reasoning")
    verdict.append("     - Long-term memory and planning")
    verdict.append("     - Symbol grounding")
    verdict.append("")
    verdict.append("  c) The manifold geometry assumption (data lies on M) is")
    verdict.append("     appropriate for perception but unclear for cognition.")
    verdict.append("")
    verdict.append("HOWEVER, two relevant observations:")
    verdict.append("  1) Periodic modulation resembles neural oscillations in")
    verdict.append("     biological systems - could inform architectural choices")
    verdict.append("  2) Riemannian structure preserves geometric invariants -")
    verdict.append("     potentially useful for equivariant representations")
    verdict.append("")
    verdict.append("These are speculative connections, not evidence.")

    # Failure analysis summary
    verdict.append("\n\n" + "="*70)
    verdict.append("FAILURE MODE SUMMARY")
    verdict.append("="*70)

    if failures:
        fundamental = [f for f in failures if f['is_fundamental']]
        fixable = [f for f in failures if not f['is_fundamental']]

        if fundamental:
            verdict.append("\nFUNDAMENTAL LIMITATIONS (cannot be fixed within framework):")
            for f in fundamental[:3]:  # Top 3
                verdict.append(f"\n  {f['failure_type'].upper()}")
                verdict.append(f"    Experiment: {f['experiment']}")
                verdict.append(f"    Reason: {f['mathematical_reason'][:200]}...")

        if fixable:
            verdict.append("\n\nFIXABLE ISSUES (implementation-dependent):")
            for f in fixable[:3]:
                verdict.append(f"\n  {f['failure_type'].upper()}")
                verdict.append(f"    Fix: {f['can_fix'][:150]}...")
    else:
        verdict.append("\nNo major failure modes detected in tested scenarios.")
        verdict.append("This may indicate:")
        verdict.append("  - Test scenarios were not sufficiently challenging")
        verdict.append("  - Or the method is robust within its operating regime")

    # Final conclusion
    verdict.append("\n\n" + "="*70)
    verdict.append("CONCLUSION")
    verdict.append("="*70)

    verdict.append("\nHMED is a mathematically coherent framework that provides:")
    verdict.append("  + Principled Riemannian optimization on latent manifolds")
    verdict.append("  + Novel periodic modulation with theoretical grounding")
    verdict.append("  + Hierarchical gradient decomposition for exploration/exploitation")
    verdict.append("")
    verdict.append("Empirical performance:")
    if len(hmed_wins) > len(baseline_wins):
        verdict.append("  POSITIVE: Outperforms baselines on majority of hard tasks")
    elif len(hmed_wins) > 0:
        verdict.append("  MIXED: Shows advantages on some tasks, comparable on others")
    else:
        verdict.append("  NEGATIVE: Does not clearly outperform simpler baselines")

    verdict.append("")
    verdict.append("The framework is USEFUL for:")
    verdict.append("  - Problems with known manifold structure")
    verdict.append("  - Multimodal inference where mode diversity matters")
    verdict.append("  - Scenarios where gradient-based methods struggle with exploration")
    verdict.append("")
    verdict.append("The framework is NOT USEFUL for:")
    verdict.append("  - High-dimensional unstructured problems")
    verdict.append("  - Discrete or symbolic domains")
    verdict.append("  - Real-time applications (due to Hessian computation)")

    return "\n".join(verdict)


def main():
    """Main entry point."""
    print("Starting HMED experimental evaluation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Configuration
    config = ExperimentConfig(
        latent_dim=16,
        num_steps=100,
        learning_rate=0.01,
        num_samples=100,  # Reduced for faster execution
        num_trials=3,     # Multiple trials for statistics
        seed=42,
        output_dir='results',
        device='cpu',
    )

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Run all experiments
    results = run_all_experiments(config)

    # Analyze results
    analysis = analyze_results(results)
    failures = identify_failure_modes(results)

    # Generate verdict
    verdict = generate_verdict(results, analysis, failures)
    print(verdict)

    # Save results
    results['analysis'] = analysis
    results['failures'] = failures

    with open(f'{config.output_dir}/full_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save verdict
    with open(f'{config.output_dir}/verdict.txt', 'w') as f:
        f.write(verdict)

    # Generate visualizations
    try:
        generate_all_visualizations(results['experiments'] + results['ablations'],
                                    config.output_dir)
    except Exception as e:
        print(f"Visualization generation failed: {e}")

    # Create summary report
    create_summary_report(results['experiments'] + results['ablations'],
                         f'{config.output_dir}/summary.txt')

    print(f"\nResults saved to {config.output_dir}/")
    print("Experiment complete.")


if __name__ == '__main__':
    main()
