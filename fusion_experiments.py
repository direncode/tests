"""
Fusion Experiments for RM-HMED Evaluation.

Test Suite:
1. Extreme Noise Inference (Geometry Stress Test)
2. Emergent Regime + Expert Alignment
3. Memory Without Backpropagation
4. Phase Transition Detection

Ablation Studies:
1. Geometry (Sphere → Euclidean)
2. Recursive term
3. Field coupling
4. Oscillatory modulation
5. Responsibility dynamics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

from manifolds import EuclideanManifold, SphereManifold
from rm_ebm import RMEBMConfig, RMEBM
from rm_hmed import (
    RMHMEDConfig, RMHMED, RMHMEDWithAblations,
    compute_order_parameter, compute_mutual_information_estimate
)


@dataclass
class FusionExperimentConfig:
    """Configuration for fusion experiments."""
    latent_dim: int = 16
    num_agents: int = 50
    num_experts: int = 4
    num_steps: int = 200
    num_trials: int = 3
    seed: int = 42
    output_dir: str = 'fusion_results'
    device: str = 'cpu'


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_name: str
    method_name: str
    metrics: Dict[str, float]
    trajectory_stats: Dict[str, Any] = field(default_factory=dict)
    stability: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TEST 1: Extreme Noise Inference
# =============================================================================

class ExtremeNoiseTest:
    """
    Geometry stress test with extreme noise.

    SNR ∈ {0.2, 0.1, 0.05, 0.01}

    Key Question: Does recursive field coupling + sphere geometry
    outperform pure geometric regularization?
    """

    def __init__(self, config: FusionExperimentConfig):
        self.config = config
        self.snr_levels = [0.2, 0.1, 0.05, 0.01]

    def create_noisy_observation(self, z_true: torch.Tensor,
                                 snr: float) -> torch.Tensor:
        """Create observation with specified SNR."""
        signal_power = (z_true ** 2).mean()
        noise_power = signal_power / snr
        noise = torch.sqrt(noise_power) * torch.randn_like(z_true)
        return z_true + noise

    def run_method(self, method_name: str, z_true: torch.Tensor,
                   y_noisy: torch.Tensor, snr: float) -> Dict:
        """Run a single method on noisy inference task."""

        if method_name == 'rmebm_euclidean':
            rmebm_config = RMEBMConfig(
                latent_dim=self.config.latent_dim,
                num_agents=self.config.num_agents,
                num_experts=self.config.num_experts,
                device=self.config.device,
            )
            model = RMEBM(rmebm_config)
            # Initialize agents near noisy observation
            model.agents = y_noisy[:self.config.num_agents].clone()

            for _ in range(self.config.num_steps):
                model.step()

            z_inferred = model.agents

        elif method_name == 'hmed_sphere':
            from hmed import HMED, HMEDConfig, MultimodalEnergy
            from manifolds import SphereManifold

            manifold = SphereManifold(self.config.latent_dim - 1)
            energy = MultimodalEnergy(self.config.latent_dim, num_modes=4)

            hmed_config = HMEDConfig(
                latent_dim=self.config.latent_dim,
                num_steps=self.config.num_steps,
                device=self.config.device,
            )
            model = HMED(hmed_config, manifold, energy)

            z_init = manifold.project(y_noisy[:self.config.num_agents])
            z_inferred, _, _ = model.run(z_init)

        elif method_name == 'rmhmed_sphere':
            rmhmed_config = RMHMEDConfig(
                latent_dim=self.config.latent_dim,
                num_agents=self.config.num_agents,
                num_experts=self.config.num_experts,
                manifold_type='sphere',
                num_steps=self.config.num_steps,
                device=self.config.device,
            )
            model = RMHMED(rmhmed_config)

            # Initialize near noisy observation (projected to sphere)
            model.agents = model.manifold.project(
                y_noisy[:self.config.num_agents].clone()
            )

            model.run()
            z_inferred = model.agents
            stability = model.get_stability_report()

        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Compute metrics
        z_true_subset = z_true[:z_inferred.shape[0]]

        # Reconstruction error
        recon_error = ((z_inferred - z_true_subset) ** 2).sum(dim=-1).sqrt()
        mean_error = recon_error.mean().item()
        std_error = recon_error.std().item()

        # Failure rate (error > 3 * true std)
        threshold = 3.0 * z_true_subset.std().item()
        failure_rate = (recon_error > threshold).float().mean().item()

        # Latent smoothness
        smoothness = z_inferred.var(dim=0).mean().item()

        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'failure_rate': failure_rate,
            'latent_smoothness': smoothness,
            'snr': snr,
        }

    def run(self) -> List[ExperimentResult]:
        """Run full noise experiment."""
        results = []
        methods = ['rmebm_euclidean', 'hmed_sphere', 'rmhmed_sphere']

        print("\n" + "="*60)
        print("TEST 1: Extreme Noise Inference")
        print("="*60)

        for snr in self.snr_levels:
            print(f"\n  SNR = {snr}")

            for method in methods:
                trial_metrics = []

                for trial in range(self.config.num_trials):
                    torch.manual_seed(self.config.seed + trial)

                    # Generate true latents
                    z_true = torch.randn(100, self.config.latent_dim)

                    # Create noisy observations
                    y_noisy = self.create_noisy_observation(z_true, snr)

                    # Run method
                    metrics = self.run_method(method, z_true, y_noisy, snr)
                    trial_metrics.append(metrics)

                # Average metrics
                avg_metrics = {}
                for key in trial_metrics[0].keys():
                    if isinstance(trial_metrics[0][key], (int, float)):
                        values = [m[key] for m in trial_metrics]
                        avg_metrics[key] = sum(values) / len(values)

                result = ExperimentResult(
                    experiment_name=f'extreme_noise_snr{snr}',
                    method_name=method,
                    metrics=avg_metrics,
                )
                results.append(result)

                print(f"    {method}: error={avg_metrics['mean_error']:.4f}, "
                      f"failure={avg_metrics['failure_rate']:.3f}")

        return results


# =============================================================================
# TEST 2: Emergent Regime + Expert Alignment
# =============================================================================

class RegimeAlignmentTest:
    """
    Test whether experts self-specialize to regimes without supervision.

    Setup:
    - 4 latent regimes (quadrants)
    - Experts initialized randomly (misaligned)
    - Measure alignment over time
    """

    def __init__(self, config: FusionExperimentConfig):
        self.config = config
        self.num_regimes = 4

    def assign_regimes(self, z: torch.Tensor) -> torch.Tensor:
        """Assign points to regimes based on quadrant."""
        if z.shape[1] < 2:
            return torch.zeros(z.shape[0], dtype=torch.long)

        sign_x = (z[:, 0] >= 0).long()
        sign_y = (z[:, 1] >= 0).long()
        return sign_x * 2 + sign_y

    def compute_alignment(self, model: RMHMED) -> Dict:
        """Compute expert-regime alignment."""
        z = model.agents
        regimes = self.assign_regimes(z)

        # Get expert responsibilities per agent
        expert_energies = model.expert_energy.all_energies(z, model.W)
        expert_probs = torch.softmax(-expert_energies / model.config.temperature, dim=-1)
        expert_assignments = expert_probs.argmax(dim=-1)

        # Compute alignment matrix: how often expert k is assigned to regime r
        alignment_matrix = torch.zeros(self.num_regimes, model.config.num_experts)
        for i in range(len(z)):
            r = regimes[i].item()
            e = expert_assignments[i].item()
            alignment_matrix[r, e] += 1

        # Normalize
        alignment_matrix = alignment_matrix / (alignment_matrix.sum() + 1e-8)

        # Compute alignment score (max assignment per regime)
        max_alignment = alignment_matrix.max(dim=1)[0].mean().item()

        # Responsibility entropy
        resp_entropy = model.responsibilities.get_entropy()

        # Check if entropy collapsed (< 0.5 * max)
        max_entropy = np.log(model.config.num_experts)
        entropy_collapsed = resp_entropy < 0.5 * max_entropy

        # W eigenvector alignment with regime structure
        eigvals, eigvecs = model.get_field_eigenspectrum()

        # Top eigenvector alignment with quadrant directions
        top_eigvec = eigvecs[:, -1]
        quadrant_dirs = torch.tensor([
            [1, 1], [-1, 1], [1, -1], [-1, -1]
        ], dtype=torch.float32)

        if model.config.latent_dim >= 2:
            eigvec_2d = top_eigvec[:2]
            alignments = torch.abs(quadrant_dirs @ eigvec_2d)
            eigvec_alignment = alignments.max().item()
        else:
            eigvec_alignment = 0.0

        return {
            'expert_regime_alignment': max_alignment,
            'responsibility_entropy': resp_entropy,
            'entropy_collapsed': entropy_collapsed,
            'eigenvector_alignment': eigvec_alignment,
            'alignment_matrix': alignment_matrix.tolist(),
        }

    def run(self) -> List[ExperimentResult]:
        """Run regime alignment experiment."""
        results = []

        print("\n" + "="*60)
        print("TEST 2: Emergent Regime + Expert Alignment")
        print("="*60)

        for manifold_type in ['euclidean', 'sphere']:
            print(f"\n  Manifold: {manifold_type}")

            trial_metrics = []
            for trial in range(self.config.num_trials):
                torch.manual_seed(self.config.seed + trial)

                config = RMHMEDConfig(
                    latent_dim=self.config.latent_dim,
                    num_agents=self.config.num_agents,
                    num_experts=self.config.num_experts,
                    manifold_type=manifold_type,
                    num_steps=self.config.num_steps,
                    device=self.config.device,
                )

                model = RMHMED(config)

                # Track alignment over time
                alignment_history = []
                for t in range(config.num_steps):
                    model.step()
                    if t % 20 == 0:
                        alignment = self.compute_alignment(model)
                        alignment_history.append(alignment)

                final_alignment = self.compute_alignment(model)
                final_alignment['alignment_improved'] = (
                    final_alignment['expert_regime_alignment'] >
                    alignment_history[0]['expert_regime_alignment']
                )

                trial_metrics.append(final_alignment)

            # Average
            avg_metrics = {
                'expert_regime_alignment': np.mean([m['expert_regime_alignment'] for m in trial_metrics]),
                'entropy_collapsed': np.mean([m['entropy_collapsed'] for m in trial_metrics]),
                'eigenvector_alignment': np.mean([m['eigenvector_alignment'] for m in trial_metrics]),
                'alignment_improved': np.mean([m['alignment_improved'] for m in trial_metrics]),
            }

            result = ExperimentResult(
                experiment_name='regime_alignment',
                method_name=f'rmhmed_{manifold_type}',
                metrics=avg_metrics,
            )
            results.append(result)

            print(f"    Alignment: {avg_metrics['expert_regime_alignment']:.3f}, "
                  f"Entropy collapsed: {avg_metrics['entropy_collapsed']:.1%}")

        return results


# =============================================================================
# TEST 3: Memory Without Backpropagation
# =============================================================================

class MemoryTest:
    """
    Test whether W(t) acts as a memory substrate.

    Setup:
    - Agents observe partial sequences
    - No BPTT - only recursive field evolution
    - Measure mutual information between early and late states
    """

    def __init__(self, config: FusionExperimentConfig):
        self.config = config

    def create_sequence_data(self, seq_length: int = 50) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Create sequence with regime structure."""
        # Regime changes every 10 steps
        regime_length = 10
        num_regimes = seq_length // regime_length

        sequence = []
        regime_labels = []

        for r in range(num_regimes):
            regime_center = torch.randn(self.config.latent_dim) * 2
            for t in range(regime_length):
                z = regime_center + 0.1 * torch.randn(
                    self.config.num_agents, self.config.latent_dim
                )
                sequence.append(z)
                regime_labels.extend([r] * self.config.num_agents)

        return sequence, torch.tensor(regime_labels)

    def run(self) -> List[ExperimentResult]:
        """Run memory experiment."""
        results = []

        print("\n" + "="*60)
        print("TEST 3: Memory Without Backpropagation")
        print("="*60)

        for use_recursion in [True, False]:
            method_name = 'with_recursion' if use_recursion else 'without_recursion'
            print(f"\n  {method_name}")

            trial_metrics = []
            for trial in range(self.config.num_trials):
                torch.manual_seed(self.config.seed + trial)

                config = RMHMEDConfig(
                    latent_dim=self.config.latent_dim,
                    num_agents=self.config.num_agents,
                    num_experts=4,
                    manifold_type='euclidean',
                    num_steps=50,  # Match sequence length
                    device=self.config.device,
                )

                model = RMHMEDWithAblations(
                    config,
                    use_recursion=use_recursion,
                )

                # Create sequence
                sequence, regime_labels = self.create_sequence_data(50)

                # Feed sequence and track states
                early_states = []
                late_states = []
                field_norms = []

                for t, z_input in enumerate(sequence):
                    # Set agents to input (simulating observation)
                    model.agents = z_input[:config.num_agents]
                    metrics = model.step()

                    if t < 10:
                        early_states.append(model.agents.clone())
                    if t >= 40:
                        late_states.append(model.agents.clone())

                    field_norms.append(model.W.norm().item())

                # Compute mutual information
                early_concat = torch.cat(early_states, dim=0)
                late_concat = torch.cat(late_states, dim=0)

                # Sample for MI estimation
                n_samples = min(200, early_concat.shape[0], late_concat.shape[0])
                early_sample = early_concat[:n_samples].detach()
                late_sample = late_concat[:n_samples].detach()

                try:
                    mi = compute_mutual_information_estimate(early_sample, late_sample)
                except:
                    mi = 0.0

                # Regime persistence
                early_regimes = regime_labels[:config.num_agents * 10]
                late_regimes = regime_labels[-config.num_agents * 10:]

                # Check if regime info is preserved
                early_regime_entropy = -(torch.bincount(early_regimes).float() /
                                         len(early_regimes)).clamp(min=1e-8).log().mean().item()

                trial_metrics.append({
                    'mutual_information': mi,
                    'field_norm_change': field_norms[-1] - field_norms[0],
                    'early_regime_entropy': early_regime_entropy,
                })

            avg_metrics = {
                'mutual_information': np.mean([m['mutual_information'] for m in trial_metrics]),
                'field_norm_change': np.mean([m['field_norm_change'] for m in trial_metrics]),
            }

            result = ExperimentResult(
                experiment_name='memory_test',
                method_name=method_name,
                metrics=avg_metrics,
            )
            results.append(result)

            print(f"    MI: {avg_metrics['mutual_information']:.4f}, "
                  f"Field change: {avg_metrics['field_norm_change']:.4f}")

        return results


# =============================================================================
# TEST 4: Phase Transition Detection
# =============================================================================

class PhaseTransitionTest:
    """
    Track order parameter R(t) = |1/M Σ_m exp(iθ_m)|

    Detect:
    - Synchronization (R → 1)
    - Oscillatory phases
    - Collapse
    """

    def __init__(self, config: FusionExperimentConfig):
        self.config = config

    def run(self) -> List[ExperimentResult]:
        """Run phase transition experiment."""
        results = []

        print("\n" + "="*60)
        print("TEST 4: Phase Transition Detection")
        print("="*60)

        for coupling_strength in [0.01, 0.1, 0.5]:
            print(f"\n  Coupling strength: {coupling_strength}")

            trial_metrics = []
            for trial in range(self.config.num_trials):
                torch.manual_seed(self.config.seed + trial)

                config = RMHMEDConfig(
                    latent_dim=self.config.latent_dim,
                    num_agents=self.config.num_agents,
                    num_experts=4,
                    manifold_type='sphere',
                    field_lr=coupling_strength,
                    num_steps=self.config.num_steps,
                    device=self.config.device,
                )

                model = RMHMED(config)

                # Track order parameter
                order_params = []
                for t in range(config.num_steps):
                    model.step()
                    R = compute_order_parameter(model.agents)
                    order_params.append(R)

                # Analyze order parameter trajectory
                R_array = np.array(order_params)

                # Detect phases
                R_mean = R_array.mean()
                R_std = R_array.std()
                R_final = R_array[-10:].mean()

                # Synchronization: R → 1
                synchronized = R_final > 0.8

                # Oscillatory: high variance
                oscillatory = R_std > 0.1

                # Collapse: all agents at same point
                final_spread = model.agents.std(dim=0).mean().item()
                collapsed = final_spread < 0.1

                # Detect transition point
                if R_array.max() - R_array.min() > 0.3:
                    # Look for steepest change
                    dR = np.diff(R_array)
                    transition_step = np.argmax(np.abs(dR))
                else:
                    transition_step = -1  # No transition

                trial_metrics.append({
                    'R_mean': R_mean,
                    'R_std': R_std,
                    'R_final': R_final,
                    'synchronized': synchronized,
                    'oscillatory': oscillatory,
                    'collapsed': collapsed,
                    'transition_step': transition_step,
                })

            avg_metrics = {
                'R_mean': np.mean([m['R_mean'] for m in trial_metrics]),
                'R_std': np.mean([m['R_std'] for m in trial_metrics]),
                'R_final': np.mean([m['R_final'] for m in trial_metrics]),
                'sync_rate': np.mean([m['synchronized'] for m in trial_metrics]),
                'oscillation_rate': np.mean([m['oscillatory'] for m in trial_metrics]),
                'collapse_rate': np.mean([m['collapsed'] for m in trial_metrics]),
            }

            result = ExperimentResult(
                experiment_name='phase_transition',
                method_name=f'coupling_{coupling_strength}',
                metrics=avg_metrics,
            )
            results.append(result)

            print(f"    R_final: {avg_metrics['R_final']:.3f}, "
                  f"Sync: {avg_metrics['sync_rate']:.1%}, "
                  f"Collapse: {avg_metrics['collapse_rate']:.1%}")

        return results


# =============================================================================
# ABLATION STUDIES
# =============================================================================

class AblationStudies:
    """
    Comprehensive ablation studies.

    Ablate:
    1. Geometry (Sphere → Euclidean)
    2. Recursive term ∇_s W · V
    3. Field coupling z z^T
    4. Oscillatory modulation
    5. Responsibility dynamics
    """

    def __init__(self, config: FusionExperimentConfig):
        self.config = config

    def run(self) -> List[ExperimentResult]:
        """Run all ablations."""
        results = []

        print("\n" + "="*60)
        print("ABLATION STUDIES")
        print("="*60)

        # Define ablation configurations
        ablations = [
            ('full', True, True, True, True, True),
            ('no_sphere', False, True, True, True, True),
            ('no_recursion', True, False, True, True, True),
            ('no_field_coupling', True, True, False, True, True),
            ('no_modulation', True, True, True, False, True),
            ('no_responsibilities', True, True, True, True, False),
            ('minimal', False, False, False, False, False),
        ]

        for name, use_manifold, use_rec, use_field, use_mod, use_resp in ablations:
            print(f"\n  {name}")

            trial_metrics = []
            for trial in range(self.config.num_trials):
                torch.manual_seed(self.config.seed + trial)

                base_config = RMHMEDConfig(
                    latent_dim=self.config.latent_dim,
                    num_agents=self.config.num_agents,
                    num_experts=4,
                    manifold_type='sphere' if use_manifold else 'euclidean',
                    num_steps=self.config.num_steps,
                    device=self.config.device,
                )

                model = RMHMEDWithAblations(
                    base_config,
                    use_manifold=use_manifold,
                    use_recursion=use_rec,
                    use_field_coupling=use_field,
                    use_modulation=use_mod,
                    use_responsibilities=use_resp,
                )

                # Run and collect metrics
                _, metrics_history = model.run()

                final_phi = metrics_history[-1]['phi']
                mean_energy = np.mean([m['mean_energy'] for m in metrics_history[-20:]])
                stability = model.get_stability_report()

                trial_metrics.append({
                    'final_phi': final_phi,
                    'mean_energy': mean_energy,
                    'is_stable': stability['is_stable'],
                    'phi_violations': stability.get('num_violations', 0),
                })

            avg_metrics = {
                'final_phi': np.mean([m['final_phi'] for m in trial_metrics]),
                'mean_energy': np.mean([m['mean_energy'] for m in trial_metrics]),
                'stability_rate': np.mean([m['is_stable'] for m in trial_metrics]),
                'avg_violations': np.mean([m['phi_violations'] for m in trial_metrics]),
            }

            result = ExperimentResult(
                experiment_name='ablation',
                method_name=name,
                metrics=avg_metrics,
            )
            results.append(result)

            print(f"    Φ: {avg_metrics['final_phi']:.4f}, "
                  f"Stable: {avg_metrics['stability_rate']:.1%}")

        return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_fusion_experiments(config: FusionExperimentConfig) -> Dict:
    """Run all fusion experiments."""
    all_results = []

    # Test 1: Extreme Noise
    noise_test = ExtremeNoiseTest(config)
    all_results.extend(noise_test.run())

    # Test 2: Regime Alignment
    regime_test = RegimeAlignmentTest(config)
    all_results.extend(regime_test.run())

    # Test 3: Memory
    memory_test = MemoryTest(config)
    all_results.extend(memory_test.run())

    # Test 4: Phase Transitions
    phase_test = PhaseTransitionTest(config)
    all_results.extend(phase_test.run())

    # Ablations
    ablation_runner = AblationStudies(config)
    all_results.extend(ablation_runner.run())

    return {
        'results': [
            {
                'experiment': r.experiment_name,
                'method': r.method_name,
                'metrics': r.metrics,
            }
            for r in all_results
        ]
    }
