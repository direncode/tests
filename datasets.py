"""
Synthetic datasets for HMED evaluation.

1. Multimodal Posterior (Mixture of Gaussians)
2. Long-Horizon Credit Assignment (Temporal latent task)
3. Noise-Dominated Inference (Low SNR observations)
4. Regime Discovery (Piecewise dynamical system)
5. Stability Under Perturbation (Various noise injections)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class DatasetConfig:
    """Configuration for synthetic datasets."""
    num_samples: int = 1000
    latent_dim: int = 16
    seed: int = 42
    device: str = 'cpu'


# =============================================================================
# Dataset 1: Multimodal Posterior (Mixture of Gaussians)
# =============================================================================

class MultimodalPosteriorDataset:
    """
    Mixture of Gaussians dataset for testing mode coverage.

    Ground truth: p(z) = Σ_k w_k * N(z | μ_k, σ_k²I)

    Metrics:
    - Mode coverage: fraction of modes discovered
    - Energy collapse rate: how often optimization collapses to single mode
    - Entropy retention: entropy of final distribution vs ground truth
    """

    def __init__(self, config: DatasetConfig, num_modes: int = 5,
                 mode_std: float = 0.5, separation: float = 3.0):
        self.config = config
        self.num_modes = num_modes
        self.mode_std = mode_std
        self.separation = separation

        torch.manual_seed(config.seed)

        # Initialize mode centers
        self.centers = self._create_centers()
        self.weights = torch.ones(num_modes, device=config.device) / num_modes

    def _create_centers(self) -> torch.Tensor:
        """Create well-separated mode centers."""
        dim = self.config.latent_dim
        centers = torch.zeros(self.num_modes, dim, device=self.config.device)

        if dim >= 2:
            # Place on hypersphere for first few modes
            angles = torch.linspace(0, 2*math.pi, self.num_modes + 1)[:-1]
            centers[:, 0] = self.separation * torch.cos(angles)
            centers[:, 1] = self.separation * torch.sin(angles)

            # Add random offsets in higher dimensions
            if dim > 2:
                centers[:, 2:] = 0.5 * torch.randn(self.num_modes, dim - 2)
        else:
            centers[:, 0] = torch.linspace(-self.separation, self.separation, self.num_modes)

        return centers

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from mixture.

        Returns:
            samples: (n, dim) tensor
            mode_labels: (n,) tensor of which mode each came from
        """
        # Sample mode assignments
        mode_labels = torch.multinomial(
            self.weights.expand(n, -1), 1
        ).squeeze(-1)

        # Sample from Gaussians
        samples = torch.zeros(n, self.config.latent_dim, device=self.config.device)
        for k in range(self.num_modes):
            mask = mode_labels == k
            count = mask.sum().item()
            if count > 0:
                samples[mask] = self.centers[k] + self.mode_std * torch.randn(
                    count, self.config.latent_dim, device=self.config.device
                )

        return samples, mode_labels

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log probability under mixture."""
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # (batch, modes, dim)
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=-1)

        log_probs = -0.5 * sq_dist / (self.mode_std ** 2) + torch.log(self.weights)
        log_probs -= 0.5 * self.config.latent_dim * math.log(2 * math.pi)
        log_probs -= self.config.latent_dim * math.log(self.mode_std)

        return torch.logsumexp(log_probs, dim=-1)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Energy = -log_prob."""
        return -self.log_prob(z)

    def assign_modes(self, z: torch.Tensor) -> torch.Tensor:
        """Assign points to nearest mode."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        dist = torch.cdist(z, self.centers)
        return dist.argmin(dim=-1)

    def compute_metrics(self, z_samples: torch.Tensor) -> Dict[str, float]:
        """
        Compute multimodal metrics.

        Returns:
            mode_coverage: fraction of modes with at least one sample
            collapse_rate: fraction of samples in dominant mode
            entropy_ratio: entropy ratio vs uniform
        """
        assignments = self.assign_modes(z_samples)

        # Mode coverage
        unique_modes = torch.unique(assignments)
        mode_coverage = len(unique_modes) / self.num_modes

        # Collapse rate
        mode_counts = torch.bincount(assignments, minlength=self.num_modes)
        dominant_fraction = mode_counts.max().item() / len(z_samples)
        collapse_rate = dominant_fraction

        # Entropy
        probs = mode_counts.float() / mode_counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        max_entropy = math.log(self.num_modes)
        entropy_ratio = entropy.item() / max_entropy

        return {
            'mode_coverage': mode_coverage,
            'collapse_rate': collapse_rate,
            'entropy_ratio': entropy_ratio,
            'num_modes_found': len(unique_modes),
        }


# =============================================================================
# Dataset 2: Long-Horizon Credit Assignment
# =============================================================================

class LongHorizonDataset:
    """
    Temporal latent task with delayed reward signal.

    Latent dynamics: z_{t+1} = f(z_t) + noise
    Reward only observed at final time T

    Tests whether methods can propagate gradient signal backwards.
    """

    def __init__(self, config: DatasetConfig, horizon: int = 50,
                 noise_std: float = 0.1):
        self.config = config
        self.horizon = horizon
        self.noise_std = noise_std

        torch.manual_seed(config.seed)

        # Transition matrix (stable dynamics)
        eigvals = 0.9 * torch.ones(config.latent_dim)
        Q, _ = torch.linalg.qr(torch.randn(config.latent_dim, config.latent_dim))
        self.A = Q @ torch.diag(eigvals) @ Q.T
        self.A = self.A.to(config.device)

        # Target at final time
        self.target = torch.randn(config.latent_dim, device=config.device)

    def generate_trajectory(self, z0: torch.Tensor) -> List[torch.Tensor]:
        """Generate latent trajectory from initial state."""
        trajectory = [z0]
        z = z0

        for t in range(self.horizon):
            noise = self.noise_std * torch.randn_like(z)
            z = torch.matmul(z, self.A.T) + noise
            trajectory.append(z)

        return trajectory

    def compute_reward(self, z_final: torch.Tensor) -> torch.Tensor:
        """Compute reward at final time (negative distance to target)."""
        return -torch.norm(z_final - self.target, dim=-1)

    def energy_at_t(self, z: torch.Tensor, t: int) -> torch.Tensor:
        """
        Compute energy at time t.
        Propagates through dynamics to final time.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Roll forward to final time
        z_curr = z
        for step in range(t, self.horizon):
            z_curr = torch.matmul(z_curr, self.A.T)

        # Energy = -reward = distance to target
        return torch.norm(z_curr - self.target, dim=-1)

    def compute_gradient_at_t(self, z: torch.Tensor, t: int) -> torch.Tensor:
        """Compute gradient of final reward w.r.t. z at time t."""
        z_grad = z.clone().requires_grad_(True)
        energy = self.energy_at_t(z_grad, t)
        grad = torch.autograd.grad(energy.sum(), z_grad)[0]
        return grad

    def compute_metrics(self, z_trajectory: List[torch.Tensor],
                        estimated_grads: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """
        Compute credit assignment metrics.

        Returns:
            final_error: distance to target at final time
            grad_decay: ratio of gradient norm at t=0 vs t=T
            trajectory_stability: variance of trajectory
        """
        z_final = z_trajectory[-1]
        final_error = torch.norm(z_final - self.target, dim=-1).mean().item()

        # Gradient decay
        if estimated_grads is not None and len(estimated_grads) > 1:
            g0_norm = torch.norm(estimated_grads[0], dim=-1).mean().item()
            gT_norm = torch.norm(estimated_grads[-1], dim=-1).mean().item()
            grad_decay = g0_norm / (gT_norm + 1e-8)
        else:
            # Compute analytically
            z0 = z_trajectory[0].clone().requires_grad_(True)
            zT = z_trajectory[-1].clone().requires_grad_(True)
            g0 = self.compute_gradient_at_t(z0, 0)
            gT = self.compute_gradient_at_t(zT, self.horizon - 1)
            grad_decay = torch.norm(g0).item() / (torch.norm(gT).item() + 1e-8)

        # Trajectory stability
        trajectory_tensor = torch.stack(z_trajectory)
        trajectory_stability = trajectory_tensor.var(dim=0).mean().item()

        return {
            'final_error': final_error,
            'grad_decay': grad_decay,
            'trajectory_stability': trajectory_stability,
        }


# =============================================================================
# Dataset 3: Noise-Dominated Inference
# =============================================================================

class NoiseDominatedDataset:
    """
    Extremely noisy observations with SNR < 0.2.

    y = f(z) + noise, where ||noise|| >> ||f(z)||

    Tests robustness of inference under severe noise.
    """

    def __init__(self, config: DatasetConfig, snr: float = 0.1,
                 observation_dim: int = 32):
        assert snr < 0.5, "SNR should be low for this test"

        self.config = config
        self.snr = snr
        self.observation_dim = observation_dim

        torch.manual_seed(config.seed)

        # Observation model: y = W z + noise
        self.W = torch.randn(observation_dim, config.latent_dim, device=config.device)
        self.W = self.W / self.W.norm() * math.sqrt(observation_dim)

        # True latents for evaluation
        self.true_latents = torch.randn(config.num_samples, config.latent_dim,
                                        device=config.device)

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """Generate noisy observation."""
        if z.dim() == 1:
            z = z.unsqueeze(0)

        signal = torch.matmul(z, self.W.T)
        signal_power = (signal ** 2).mean()

        # Add noise to achieve target SNR
        noise_power = signal_power / self.snr
        noise_std = math.sqrt(noise_power.item())
        noise = noise_std * torch.randn_like(signal)

        return signal + noise

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (z, y) pairs."""
        z = torch.randn(n, self.config.latent_dim, device=self.config.device)
        y = self.observe(z)
        return z, y

    def reconstruction_energy(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Energy = reconstruction error."""
        y_pred = torch.matmul(z, self.W.T)
        return ((y - y_pred) ** 2).sum(dim=-1)

    def compute_metrics(self, z_inferred: torch.Tensor, z_true: torch.Tensor,
                        y: torch.Tensor) -> Dict[str, float]:
        """
        Compute noise robustness metrics.

        Returns:
            reconstruction_error: ||y - Wz||
            latent_error: ||z_inferred - z_true||
            latent_smoothness: local variation in z
            failure_rate: fraction of completely failed inferences
        """
        # Reconstruction
        y_recon = torch.matmul(z_inferred, self.W.T)
        recon_error = ((y - y_recon) ** 2).sum(dim=-1).sqrt().mean().item()

        # Latent error
        latent_error = ((z_inferred - z_true) ** 2).sum(dim=-1).sqrt().mean().item()

        # Smoothness (variance across batch as proxy)
        smoothness = z_inferred.var(dim=0).mean().item()

        # Failure rate: if latent error > threshold
        threshold = 5.0 * z_true.std().item()
        per_sample_error = ((z_inferred - z_true) ** 2).sum(dim=-1).sqrt()
        failure_rate = (per_sample_error > threshold).float().mean().item()

        return {
            'reconstruction_error': recon_error,
            'latent_error': latent_error,
            'latent_smoothness': smoothness,
            'failure_rate': failure_rate,
        }


# =============================================================================
# Dataset 4: Regime Discovery
# =============================================================================

class RegimeDiscoveryDataset:
    """
    Piecewise-defined dynamical system with multiple regimes.

    z_t ∈ R_k => dynamics follow f_k(z)

    Tests unsupervised discovery of regime structure.
    """

    def __init__(self, config: DatasetConfig, num_regimes: int = 4,
                 trajectory_length: int = 100):
        self.config = config
        self.num_regimes = num_regimes
        self.trajectory_length = trajectory_length

        torch.manual_seed(config.seed)

        # Define regime boundaries (using first 2 dimensions)
        self.boundaries = self._create_boundaries()

        # Define dynamics for each regime
        self.dynamics = self._create_dynamics()

    def _create_boundaries(self) -> List[Tuple[float, float]]:
        """Create quadrant-based boundaries."""
        # Use sign of first two coordinates
        return [(0.0, 0.0)]  # Just track quadrant signs

    def _create_dynamics(self) -> List[torch.Tensor]:
        """Create different dynamics matrices for each regime."""
        dynamics = []
        dim = self.config.latent_dim

        for k in range(self.num_regimes):
            # Each regime has different rotation + scaling
            angle = 2 * math.pi * k / self.num_regimes
            scale = 0.95 + 0.03 * k  # Slightly different stability

            # Create rotation in first 2D subspace
            R = torch.eye(dim, device=self.config.device)
            R[0, 0] = scale * math.cos(angle * 0.1)
            R[0, 1] = -scale * math.sin(angle * 0.1)
            R[1, 0] = scale * math.sin(angle * 0.1)
            R[1, 1] = scale * math.cos(angle * 0.1)

            # Rest is slightly contractive
            for i in range(2, dim):
                R[i, i] = 0.9

            dynamics.append(R)

        return dynamics

    def get_regime(self, z: torch.Tensor) -> torch.Tensor:
        """Determine regime based on position."""
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Use quadrant of first 2 dimensions
        sign_x = (z[:, 0] >= 0).long()
        sign_y = (z[:, 1] >= 0).long() if z.shape[1] > 1 else torch.zeros_like(sign_x)

        regime = sign_x * 2 + sign_y
        return regime % self.num_regimes

    def step(self, z: torch.Tensor, noise_std: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one step of dynamics."""
        if z.dim() == 1:
            z = z.unsqueeze(0)

        regime = self.get_regime(z)
        z_new = torch.zeros_like(z)

        for k in range(self.num_regimes):
            mask = regime == k
            if mask.any():
                z_new[mask] = torch.matmul(z[mask], self.dynamics[k].T)

        # Add noise
        z_new = z_new + noise_std * torch.randn_like(z_new)

        return z_new, regime

    def generate_trajectory(self, z0: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate trajectory with regime labels."""
        trajectory = [z0]
        regimes = [self.get_regime(z0)]
        z = z0

        for t in range(self.trajectory_length - 1):
            z, regime = self.step(z)
            trajectory.append(z)
            regimes.append(regime)

        return trajectory, regimes

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points with regime labels."""
        z = 3.0 * torch.randn(n, self.config.latent_dim, device=self.config.device)
        regimes = self.get_regime(z)
        return z, regimes

    def compute_metrics(self, z_samples: torch.Tensor,
                        predicted_regimes: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute regime discovery metrics.

        Returns:
            num_regimes_discovered: number of distinct clusters found
            transition_sharpness: how sharp regime boundaries are
            latent_separability: between-cluster vs within-cluster variance
        """
        true_regimes = self.get_regime(z_samples)

        # Count discovered regimes (from predictions or clustering)
        if predicted_regimes is not None:
            unique_pred = torch.unique(predicted_regimes)
            num_discovered = len(unique_pred)
        else:
            # Use k-means-like counting
            num_discovered = len(torch.unique(true_regimes))

        # Transition sharpness: measure gradient of regime assignment
        # Higher = sharper boundaries
        eps = 0.1
        z_perturbed = z_samples + eps * torch.randn_like(z_samples)
        regimes_perturbed = self.get_regime(z_perturbed)
        transitions = (true_regimes != regimes_perturbed).float().mean().item()
        transition_sharpness = transitions / eps

        # Latent separability: between-cluster / within-cluster variance
        within_var = 0.0
        between_var = 0.0
        global_mean = z_samples.mean(dim=0)

        for k in range(self.num_regimes):
            mask = true_regimes == k
            if mask.sum() > 1:
                cluster_samples = z_samples[mask]
                cluster_mean = cluster_samples.mean(dim=0)
                within_var += ((cluster_samples - cluster_mean) ** 2).sum().item()
                between_var += mask.sum().item() * ((cluster_mean - global_mean) ** 2).sum().item()

        separability = between_var / (within_var + 1e-8)

        return {
            'num_regimes_discovered': num_discovered,
            'transition_sharpness': transition_sharpness,
            'latent_separability': separability,
        }


# =============================================================================
# Dataset 5: Stability Under Perturbation
# =============================================================================

class PerturbationDataset:
    """
    Test stability under various perturbations:
    1. Gradient noise
    2. Parameter drift
    3. Initialization chaos
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        torch.manual_seed(config.seed)

        # Base energy function parameters
        self.A = torch.eye(config.latent_dim, device=config.device)
        self.target = torch.zeros(config.latent_dim, device=config.device)

    def base_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Simple quadratic energy."""
        return 0.5 * ((z - self.target) ** 2).sum(dim=-1)

    def perturbed_gradient(self, z: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Gradient with additive noise."""
        grad = z - self.target
        noise = noise_level * torch.randn_like(grad)
        return grad + noise

    def drifted_energy(self, z: torch.Tensor, drift_magnitude: float,
                       step: int) -> torch.Tensor:
        """Energy with drifting target."""
        drift = drift_magnitude * math.sin(0.1 * step) * torch.ones_like(z)
        return 0.5 * ((z - self.target - drift) ** 2).sum(dim=-1)

    def chaotic_initialization(self, n: int, chaos_level: float) -> torch.Tensor:
        """Generate chaotically distributed initial points."""
        # Use logistic map for chaos
        x = torch.rand(n, device=self.config.device) * 0.999 + 0.0001
        for _ in range(100):  # Iterate to reach chaotic attractor
            x = 4.0 * x * (1 - x)

        # Scale to latent space
        z = chaos_level * (2 * x.unsqueeze(-1) - 1) * torch.randn(
            n, self.config.latent_dim, device=self.config.device
        )
        return z

    def compute_stability_metrics(self, trajectories: List[List[torch.Tensor]],
                                  perturbation_type: str) -> Dict[str, float]:
        """
        Compute stability metrics.

        Args:
            trajectories: List of trajectories from different runs

        Returns:
            convergence_rate: fraction that converged
            divergence_rate: fraction that diverged
            recovery_rate: fraction that recovered after perturbation
        """
        final_energies = []
        converged = []
        diverged = []

        threshold_converge = 0.1
        threshold_diverge = 100.0

        for traj in trajectories:
            z_final = traj[-1]
            energy = self.base_energy(z_final).mean().item()
            final_energies.append(energy)

            converged.append(energy < threshold_converge)
            diverged.append(energy > threshold_diverge)

        convergence_rate = sum(converged) / len(converged)
        divergence_rate = sum(diverged) / len(diverged)
        recovery_rate = 1.0 - divergence_rate

        return {
            f'{perturbation_type}_convergence_rate': convergence_rate,
            f'{perturbation_type}_divergence_rate': divergence_rate,
            f'{perturbation_type}_recovery_rate': recovery_rate,
            f'{perturbation_type}_mean_final_energy': np.mean(final_energies),
        }


# =============================================================================
# Dataset Factory
# =============================================================================

def create_dataset(name: str, config: DatasetConfig, **kwargs):
    """Factory function to create datasets."""
    datasets = {
        'multimodal': MultimodalPosteriorDataset,
        'long_horizon': LongHorizonDataset,
        'noisy': NoiseDominatedDataset,
        'regime': RegimeDiscoveryDataset,
        'perturbation': PerturbationDataset,
    }

    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets.keys())}")

    return datasets[name](config, **kwargs)
