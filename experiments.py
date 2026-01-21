"""
Experiment runner for HMED evaluation.

Runs:
1. All test suite experiments
2. Ablation studies
3. Collects results and generates reports
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from manifolds import Manifold, EuclideanManifold, SphereManifold, HyperbolicManifold, create_manifold
from hmed import HMED, HMEDWithAblations, HMEDConfig, EnergyFunction, MultimodalEnergy
from baselines import BaselineConfig, StandardGD, VAEInference, DiffusionScoreDescent, BaselineRunner
from datasets import (
    DatasetConfig, MultimodalPosteriorDataset, LongHorizonDataset,
    NoiseDominatedDataset, RegimeDiscoveryDataset, PerturbationDataset
)


@dataclass
class ExperimentConfig:
    """Master configuration for experiments."""
    # Dimensions
    latent_dim: int = 16
    num_steps: int = 100
    learning_rate: float = 0.01

    # Dataset parameters
    num_samples: int = 200
    num_trials: int = 5

    # Random seed
    seed: int = 42

    # Output
    output_dir: str = 'results'
    device: str = 'cpu'


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_name: str
    method_name: str
    metrics: Dict[str, float]
    trajectory_stats: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class EnergyFromDataset(EnergyFunction):
    """Wrapper to use dataset energy as EnergyFunction."""

    def __init__(self, dataset, latent_dim: int):
        super().__init__(latent_dim)
        self.dataset = dataset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dataset.energy(z)


class ReconstructionEnergy(EnergyFunction):
    """Energy for reconstruction tasks."""

    def __init__(self, dataset, y: torch.Tensor, latent_dim: int):
        super().__init__(latent_dim)
        self.dataset = dataset
        self.y = y

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dataset.reconstruction_energy(z, self.y)


class TemporalEnergy(EnergyFunction):
    """Energy for temporal tasks."""

    def __init__(self, dataset, t: int, latent_dim: int):
        super().__init__(latent_dim)
        self.dataset = dataset
        self.t = t

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dataset.energy_at_t(z, self.t)


# =============================================================================
# Experiment Runners
# =============================================================================

class ExperimentRunner:
    """Main experiment runner."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        torch.manual_seed(config.seed)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: List[ExperimentResult] = []

    def _create_hmed(self, manifold: Manifold, energy_fn: EnergyFunction,
                     **kwargs) -> HMED:
        """Create HMED instance."""
        hmed_config = HMEDConfig(
            latent_dim=self.config.latent_dim,
            num_steps=self.config.num_steps,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
            **kwargs
        )
        return HMED(hmed_config, manifold, energy_fn)

    def _create_baseline_config(self) -> BaselineConfig:
        """Create baseline config matching HMED."""
        return BaselineConfig(
            latent_dim=self.config.latent_dim,
            num_steps=self.config.num_steps,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
        )

    def run_multimodal_experiment(self) -> List[ExperimentResult]:
        """
        Experiment 1: Multimodal Posterior Recovery

        Question: Does HMED maintain multiple stable attractors longer?
        """
        print("\n" + "="*60)
        print("EXPERIMENT 1: Multimodal Posterior Recovery")
        print("="*60)

        results = []
        dataset_config = DatasetConfig(
            num_samples=self.config.num_samples,
            latent_dim=self.config.latent_dim,
            seed=self.config.seed,
            device=self.config.device,
        )

        dataset = MultimodalPosteriorDataset(
            dataset_config, num_modes=5, mode_std=0.5, separation=3.0
        )

        # Energy function from dataset
        energy_fn = EnergyFromDataset(dataset, self.config.latent_dim)

        # Initialize from random points
        z_init, _ = dataset.sample(self.config.num_samples)

        # Run methods
        methods = self._get_all_methods(energy_fn)

        for method_name, method in methods.items():
            print(f"\n  Running {method_name}...")

            trial_metrics = []
            for trial in range(self.config.num_trials):
                # Re-initialize
                torch.manual_seed(self.config.seed + trial)
                z_init_trial = dataset.sample(self.config.num_samples)[0]

                # Run method
                if hasattr(method, 'run'):
                    z_final, metrics, trajectory = method.run(z_init_trial)
                else:
                    continue

                # Compute dataset metrics
                ds_metrics = dataset.compute_metrics(z_final)
                trial_metrics.append(ds_metrics)

            # Average over trials
            avg_metrics = {}
            for key in trial_metrics[0].keys():
                values = [m[key] for m in trial_metrics]
                avg_metrics[key] = sum(values) / len(values)
                avg_metrics[f'{key}_std'] = (
                    sum((v - avg_metrics[key])**2 for v in values) / len(values)
                ) ** 0.5

            result = ExperimentResult(
                experiment_name='multimodal_recovery',
                method_name=method_name,
                metrics=avg_metrics,
            )
            results.append(result)
            print(f"    Mode coverage: {avg_metrics['mode_coverage']:.3f}")
            print(f"    Collapse rate: {avg_metrics['collapse_rate']:.3f}")
            print(f"    Entropy ratio: {avg_metrics['entropy_ratio']:.3f}")

        self.results.extend(results)
        return results

    def run_long_horizon_experiment(self) -> List[ExperimentResult]:
        """
        Experiment 2: Long-Horizon Credit Assignment

        Question: Does hierarchical modulation reduce vanishing influence?
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Long-Horizon Credit Assignment")
        print("="*60)

        results = []
        dataset_config = DatasetConfig(
            num_samples=self.config.num_samples,
            latent_dim=self.config.latent_dim,
            seed=self.config.seed,
            device=self.config.device,
        )

        for horizon in [10, 25, 50]:
            print(f"\n  Horizon = {horizon}")

            dataset = LongHorizonDataset(
                dataset_config, horizon=horizon, noise_std=0.1
            )

            # Use temporal energy at t=0
            energy_fn = TemporalEnergy(dataset, t=0, latent_dim=self.config.latent_dim)

            # Initialize near origin
            z_init = 0.5 * torch.randn(self.config.num_samples, self.config.latent_dim)

            methods = self._get_all_methods(energy_fn)

            for method_name, method in methods.items():
                trial_metrics = []

                for trial in range(self.config.num_trials):
                    torch.manual_seed(self.config.seed + trial)
                    z_init_trial = 0.5 * torch.randn(
                        self.config.num_samples, self.config.latent_dim
                    )

                    if hasattr(method, 'run'):
                        z_final, opt_metrics, trajectory = method.run(z_init_trial)

                        # Generate trajectory through dynamics
                        dyn_trajectory = dataset.generate_trajectory(z_final)

                        # Compute metrics
                        ds_metrics = dataset.compute_metrics(dyn_trajectory)
                        ds_metrics['horizon'] = horizon
                        trial_metrics.append(ds_metrics)

                if trial_metrics:
                    avg_metrics = {}
                    for key in trial_metrics[0].keys():
                        if isinstance(trial_metrics[0][key], (int, float)):
                            values = [m[key] for m in trial_metrics]
                            avg_metrics[key] = sum(values) / len(values)

                    result = ExperimentResult(
                        experiment_name=f'long_horizon_h{horizon}',
                        method_name=method_name,
                        metrics=avg_metrics,
                    )
                    results.append(result)
                    print(f"    {method_name}: error={avg_metrics.get('final_error', 'N/A'):.4f}, "
                          f"grad_decay={avg_metrics.get('grad_decay', 'N/A'):.4f}")

        self.results.extend(results)
        return results

    def run_noise_dominated_experiment(self) -> List[ExperimentResult]:
        """
        Experiment 3: Noise-Dominated Inference

        Question: Does the energy well structure stabilize inference?
        """
        print("\n" + "="*60)
        print("EXPERIMENT 3: Noise-Dominated Inference")
        print("="*60)

        results = []
        dataset_config = DatasetConfig(
            num_samples=self.config.num_samples,
            latent_dim=self.config.latent_dim,
            seed=self.config.seed,
            device=self.config.device,
        )

        for snr in [0.05, 0.1, 0.2]:
            print(f"\n  SNR = {snr}")

            dataset = NoiseDominatedDataset(
                dataset_config, snr=snr, observation_dim=32
            )

            # Sample data
            z_true, y = dataset.sample(self.config.num_samples)

            # Energy for reconstruction
            energy_fn = ReconstructionEnergy(
                dataset, y, self.config.latent_dim
            )

            # Initialize from noisy estimate
            z_init = 0.1 * torch.randn_like(z_true)

            methods = self._get_all_methods(energy_fn)

            for method_name, method in methods.items():
                trial_metrics = []

                for trial in range(self.config.num_trials):
                    torch.manual_seed(self.config.seed + trial)
                    z_true_t, y_t = dataset.sample(self.config.num_samples)
                    z_init_t = 0.1 * torch.randn_like(z_true_t)

                    # Update energy with new observations
                    if hasattr(energy_fn, 'y'):
                        energy_fn.y = y_t

                    if hasattr(method, 'run'):
                        z_final, opt_metrics, trajectory = method.run(z_init_t)
                        ds_metrics = dataset.compute_metrics(z_final, z_true_t, y_t)
                        ds_metrics['snr'] = snr
                        trial_metrics.append(ds_metrics)

                if trial_metrics:
                    avg_metrics = {}
                    for key in trial_metrics[0].keys():
                        if isinstance(trial_metrics[0][key], (int, float)):
                            values = [m[key] for m in trial_metrics]
                            avg_metrics[key] = sum(values) / len(values)

                    result = ExperimentResult(
                        experiment_name=f'noise_dominated_snr{snr}',
                        method_name=method_name,
                        metrics=avg_metrics,
                    )
                    results.append(result)
                    print(f"    {method_name}: recon_err={avg_metrics.get('reconstruction_error', 'N/A'):.4f}, "
                          f"failure={avg_metrics.get('failure_rate', 'N/A'):.3f}")

        self.results.extend(results)
        return results

    def run_regime_discovery_experiment(self) -> List[ExperimentResult]:
        """
        Experiment 4: Regime Discovery

        Question: Does geometry induce regime structure without labels?
        """
        print("\n" + "="*60)
        print("EXPERIMENT 4: Regime Discovery")
        print("="*60)

        results = []
        dataset_config = DatasetConfig(
            num_samples=self.config.num_samples,
            latent_dim=self.config.latent_dim,
            seed=self.config.seed,
            device=self.config.device,
        )

        dataset = RegimeDiscoveryDataset(
            dataset_config, num_regimes=4, trajectory_length=100
        )

        # Sample points
        z_samples, true_regimes = dataset.sample(self.config.num_samples)

        # Create energy that depends on regime structure
        # (Use multimodal with centers at regime centers)
        multimodal = MultimodalEnergy(
            self.config.latent_dim, num_modes=4, mode_std=1.0, separation=4.0
        )

        z_init = 2.0 * torch.randn(self.config.num_samples, self.config.latent_dim)
        methods = self._get_all_methods(multimodal)

        # Also test with different manifolds
        manifold_types = ['euclidean', 'sphere']

        for manifold_type in manifold_types:
            print(f"\n  Manifold: {manifold_type}")

            if manifold_type == 'sphere':
                manifold = SphereManifold(self.config.latent_dim - 1)
                # Adjust init for sphere
                z_init_m = manifold.random_point(
                    (self.config.num_samples, self.config.latent_dim),
                    device=torch.device(self.config.device)
                )
            else:
                manifold = EuclideanManifold(self.config.latent_dim)
                z_init_m = z_init

            hmed = self._create_hmed(manifold, multimodal)

            trial_metrics = []
            for trial in range(self.config.num_trials):
                torch.manual_seed(self.config.seed + trial)

                if manifold_type == 'sphere':
                    z_init_t = manifold.random_point(
                        (self.config.num_samples, self.config.latent_dim),
                        device=torch.device(self.config.device)
                    )
                else:
                    z_init_t = 2.0 * torch.randn(
                        self.config.num_samples, self.config.latent_dim
                    )

                z_final, opt_metrics, trajectory = hmed.run(z_init_t)
                ds_metrics = dataset.compute_metrics(z_final)
                trial_metrics.append(ds_metrics)

            avg_metrics = {}
            for key in trial_metrics[0].keys():
                values = [m[key] for m in trial_metrics]
                avg_metrics[key] = sum(values) / len(values)

            result = ExperimentResult(
                experiment_name=f'regime_discovery_{manifold_type}',
                method_name=f'hmed_{manifold_type}',
                metrics=avg_metrics,
            )
            results.append(result)
            print(f"    HMED: regimes={avg_metrics['num_regimes_discovered']:.1f}, "
                  f"separability={avg_metrics['latent_separability']:.3f}")

        self.results.extend(results)
        return results

    def run_stability_experiment(self) -> List[ExperimentResult]:
        """
        Experiment 5: Stability Under Perturbation

        Question: Does the system recover or diverge?
        """
        print("\n" + "="*60)
        print("EXPERIMENT 5: Stability Under Perturbation")
        print("="*60)

        results = []
        dataset_config = DatasetConfig(
            num_samples=self.config.num_samples,
            latent_dim=self.config.latent_dim,
            seed=self.config.seed,
            device=self.config.device,
        )

        dataset = PerturbationDataset(dataset_config)

        # Test conditions
        perturbations = [
            ('gradient_noise', {'noise_levels': [0.1, 0.5, 1.0]}),
            ('param_drift', {'drift_magnitudes': [0.1, 0.5, 1.0]}),
            ('init_chaos', {'chaos_levels': [1.0, 5.0, 10.0]}),
        ]

        # Simple quadratic energy
        class QuadraticE(EnergyFunction):
            def __init__(self, dim):
                super().__init__(dim)
            def forward(self, z):
                return 0.5 * (z ** 2).sum(dim=-1)

        energy_fn = QuadraticE(self.config.latent_dim)
        manifold = EuclideanManifold(self.config.latent_dim)

        for perturb_type, params in perturbations:
            print(f"\n  Perturbation: {perturb_type}")

            if perturb_type == 'gradient_noise':
                for noise_level in params['noise_levels']:
                    trajectories = []
                    for trial in range(self.config.num_trials * 2):
                        torch.manual_seed(self.config.seed + trial)
                        z = torch.randn(10, self.config.latent_dim)
                        traj = [z.clone()]

                        for step in range(self.config.num_steps):
                            grad = dataset.perturbed_gradient(z, noise_level)
                            z = z - self.config.learning_rate * grad
                            traj.append(z.clone())

                        trajectories.append(traj)

                    metrics = dataset.compute_stability_metrics(
                        trajectories, f'noise_{noise_level}'
                    )
                    result = ExperimentResult(
                        experiment_name=f'stability_{perturb_type}',
                        method_name=f'level_{noise_level}',
                        metrics=metrics,
                    )
                    results.append(result)
                    print(f"    noise={noise_level}: conv={metrics.get(f'noise_{noise_level}_convergence_rate', 0):.3f}")

            elif perturb_type == 'init_chaos':
                for chaos_level in params['chaos_levels']:
                    trajectories = []
                    for trial in range(self.config.num_trials * 2):
                        torch.manual_seed(self.config.seed + trial)
                        z = dataset.chaotic_initialization(10, chaos_level)
                        traj = [z.clone()]

                        hmed = self._create_hmed(manifold, energy_fn)
                        z_final, _, traj_hmed = hmed.run(z)
                        trajectories.append(traj_hmed)

                    metrics = dataset.compute_stability_metrics(
                        trajectories, f'chaos_{chaos_level}'
                    )
                    result = ExperimentResult(
                        experiment_name=f'stability_{perturb_type}',
                        method_name=f'level_{chaos_level}',
                        metrics=metrics,
                    )
                    results.append(result)
                    print(f"    chaos={chaos_level}: conv={metrics.get(f'chaos_{chaos_level}_convergence_rate', 0):.3f}")

        self.results.extend(results)
        return results

    def _get_all_methods(self, energy_fn: EnergyFunction) -> Dict[str, Any]:
        """Create all methods for comparison."""
        manifold_euclidean = EuclideanManifold(self.config.latent_dim)
        baseline_config = self._create_baseline_config()

        methods = {
            # HMED variants
            'hmed_full': self._create_hmed(manifold_euclidean, energy_fn),
            'hmed_euclidean': self._create_hmed(manifold_euclidean, energy_fn),

            # Baselines
            'sgd': StandardGD(baseline_config, energy_fn),
            'diffusion': DiffusionScoreDescent(baseline_config, energy_fn),
        }

        # Add sphere HMED if applicable
        if self.config.latent_dim >= 3:
            manifold_sphere = SphereManifold(self.config.latent_dim - 1)
            methods['hmed_sphere'] = self._create_hmed(manifold_sphere, energy_fn)

        return methods


# =============================================================================
# Ablation Studies
# =============================================================================

class AblationRunner:
    """Runner for ablation studies."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []

    def run_all_ablations(self) -> List[ExperimentResult]:
        """Run all ablation studies."""
        print("\n" + "="*60)
        print("ABLATION STUDIES")
        print("="*60)

        results = []

        # Create test energy
        energy_fn = MultimodalEnergy(
            self.config.latent_dim, num_modes=5, mode_std=0.5, separation=3.0
        )
        manifold = EuclideanManifold(self.config.latent_dim)

        # Initialize
        z_init = torch.randn(self.config.num_samples, self.config.latent_dim)

        # Ablation configurations
        ablations = [
            ('full', True, True, True),
            ('no_modulation', False, True, True),
            ('no_hierarchy', True, False, True),
            ('no_aggregation', True, True, False),
            ('minimal', False, False, False),
        ]

        hmed_config = HMEDConfig(
            latent_dim=self.config.latent_dim,
            num_steps=self.config.num_steps,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
        )

        print("\n  Ablation results:")
        print("  " + "-"*50)

        for name, use_mod, use_hier, use_agg in ablations:
            trial_metrics = []

            for trial in range(self.config.num_trials):
                torch.manual_seed(self.config.seed + trial)
                z_init_t = torch.randn(self.config.num_samples, self.config.latent_dim)

                hmed = HMEDWithAblations(
                    hmed_config, manifold, energy_fn,
                    use_modulation=use_mod,
                    use_hierarchy=use_hier,
                    use_aggregation=use_agg,
                )

                z_final, metrics, trajectory = hmed.run(z_init_t)

                # Compute final energy
                final_energy = energy_fn(z_final).mean().item()
                trial_metrics.append({
                    'final_energy': final_energy,
                    'convergence_steps': len(metrics) - 1,
                })

            avg_metrics = {
                'final_energy': sum(m['final_energy'] for m in trial_metrics) / len(trial_metrics),
                'use_modulation': use_mod,
                'use_hierarchy': use_hier,
                'use_aggregation': use_agg,
            }

            result = ExperimentResult(
                experiment_name='ablation',
                method_name=name,
                metrics=avg_metrics,
            )
            results.append(result)

            print(f"    {name:20s}: energy={avg_metrics['final_energy']:.4f} "
                  f"(mod={use_mod}, hier={use_hier}, agg={use_agg})")

        # Manifold ablation
        print("\n  Manifold comparison:")
        print("  " + "-"*50)

        manifolds = [
            ('euclidean', EuclideanManifold(self.config.latent_dim)),
        ]

        if self.config.latent_dim >= 3:
            manifolds.append(('sphere', SphereManifold(self.config.latent_dim - 1)))
            manifolds.append(('hyperbolic', HyperbolicManifold(self.config.latent_dim, curvature=-1.0)))

        for manifold_name, manifold in manifolds:
            trial_metrics = []

            for trial in range(self.config.num_trials):
                torch.manual_seed(self.config.seed + trial)

                if manifold_name == 'sphere':
                    z_init_t = manifold.random_point(
                        (self.config.num_samples, manifold.ambient_dim),
                        device=torch.device(self.config.device)
                    )
                elif manifold_name == 'hyperbolic':
                    z_init_t = manifold.random_point(
                        (self.config.num_samples, self.config.latent_dim),
                        device=torch.device(self.config.device)
                    )
                else:
                    z_init_t = torch.randn(self.config.num_samples, self.config.latent_dim)

                hmed = HMED(hmed_config, manifold, energy_fn)
                z_final, metrics, trajectory = hmed.run(z_init_t)

                final_energy = energy_fn(z_final).mean().item()
                trial_metrics.append({'final_energy': final_energy})

            avg_energy = sum(m['final_energy'] for m in trial_metrics) / len(trial_metrics)

            result = ExperimentResult(
                experiment_name='ablation_manifold',
                method_name=manifold_name,
                metrics={
                    'final_energy': avg_energy,
                    'curvature': manifold.curvature(z_init_t[:1]) if hasattr(manifold, 'curvature') else 0,
                },
            )
            results.append(result)
            print(f"    {manifold_name:15s}: energy={avg_energy:.4f}")

        self.results.extend(results)
        return results


# =============================================================================
# Results Aggregation
# =============================================================================

def aggregate_results(results: List[ExperimentResult]) -> Dict[str, Dict]:
    """Aggregate results by experiment and method."""
    aggregated = {}

    for result in results:
        exp_name = result.experiment_name
        method_name = result.method_name

        if exp_name not in aggregated:
            aggregated[exp_name] = {}

        aggregated[exp_name][method_name] = result.metrics

    return aggregated


def print_results_table(results: List[ExperimentResult]):
    """Print results in table format."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    aggregated = aggregate_results(results)

    for exp_name, methods in aggregated.items():
        print(f"\n{exp_name}:")
        print("-"*60)

        # Get all metric names
        all_metrics = set()
        for method_metrics in methods.values():
            all_metrics.update(method_metrics.keys())

        # Print header
        header = f"{'Method':<20}"
        metric_names = sorted([m for m in all_metrics if not m.endswith('_std')])[:4]
        for m in metric_names:
            header += f"{m[:12]:<14}"
        print(header)
        print("-"*60)

        # Print rows
        for method_name, metrics in methods.items():
            row = f"{method_name:<20}"
            for m in metric_names:
                val = metrics.get(m, 'N/A')
                if isinstance(val, float):
                    row += f"{val:<14.4f}"
                else:
                    row += f"{str(val):<14}"
            print(row)


def save_results(results: List[ExperimentResult], filepath: str):
    """Save results to JSON."""
    data = []
    for r in results:
        data.append({
            'experiment': r.experiment_name,
            'method': r.method_name,
            'metrics': r.metrics,
        })

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
