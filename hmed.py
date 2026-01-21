"""
Hierarchical Modulated Energy Descent (HMED) on Riemannian Manifolds.

Core algorithm implementing:
1. Riemannian gradient descent with retraction
2. Hierarchical decomposition (horizontal + vertical)
3. Bounded periodic modulation
4. Hierarchical energy aggregation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import math

from manifolds import Manifold, EuclideanManifold, SphereManifold


@dataclass
class HMEDConfig:
    """Configuration for HMED algorithm."""
    # Manifold settings
    latent_dim: int = 16
    manifold_type: str = 'euclidean'  # 'euclidean', 'sphere', 'hyperbolic'

    # Descent parameters
    learning_rate: float = 0.01
    num_steps: int = 100
    lr_decay: float = 0.99

    # Hierarchical decomposition
    lambda_vertical: float = 0.5  # Weight for vertical component
    num_hierarchy_levels: int = 3

    # Periodic modulation
    alpha: float = 1.1  # Decay base for modulation
    omega: float = 2.0  # Frequency for energy-based oscillation

    # Energy aggregation
    aggregation_betas: Optional[List[float]] = None  # Weights for log-gradient terms
    epsilon: float = 1e-8  # Numerical stability

    # General
    device: str = 'cpu'

    def __post_init__(self):
        if self.aggregation_betas is None:
            # Default: exponentially decaying weights
            self.aggregation_betas = [0.1 * (0.5 ** k) for k in range(self.num_steps)]


class EnergyFunction(nn.Module):
    """
    Base class for energy functions E(z; θ).
    The energy landscape defines the inference objective.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy L(z) = E(z; θ)."""
        raise NotImplementedError


class QuadraticEnergy(EnergyFunction):
    """Simple quadratic energy for testing: E(z) = 0.5 * z^T A z + b^T z."""

    def __init__(self, latent_dim: int, condition_number: float = 10.0):
        super().__init__(latent_dim)
        # Create positive definite matrix with specified condition number
        eigvals = torch.logspace(0, math.log10(condition_number), latent_dim)
        Q, _ = torch.linalg.qr(torch.randn(latent_dim, latent_dim))
        self.register_buffer('A', Q @ torch.diag(eigvals) @ Q.T)
        self.register_buffer('b', torch.randn(latent_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, dim) or (dim,)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        # 0.5 * z^T A z + b^T z
        quad = 0.5 * torch.einsum('bi,ij,bj->b', z, self.A, z)
        linear = torch.einsum('bi,i->b', z, self.b)
        return quad + linear


class MultimodalEnergy(EnergyFunction):
    """
    Multimodal energy with multiple wells (mixture of Gaussians).
    E(z) = -log(sum_k w_k * exp(-0.5 * ||z - μ_k||^2 / σ_k^2))
    """

    def __init__(self, latent_dim: int, num_modes: int = 5,
                 mode_std: float = 0.5, separation: float = 3.0):
        super().__init__(latent_dim)
        self.num_modes = num_modes
        self.mode_std = mode_std

        # Initialize mode centers
        if latent_dim >= 2:
            # Place modes in a circle for first 2 dimensions
            angles = torch.linspace(0, 2*math.pi, num_modes + 1)[:-1]
            centers = torch.zeros(num_modes, latent_dim)
            centers[:, 0] = separation * torch.cos(angles)
            centers[:, 1] = separation * torch.sin(angles)
        else:
            centers = torch.linspace(-separation, separation, num_modes).unsqueeze(1)

        self.register_buffer('centers', centers)
        self.register_buffer('weights', torch.ones(num_modes) / num_modes)
        self.register_buffer('log_norm', torch.tensor(
            0.5 * latent_dim * math.log(2 * math.pi) + latent_dim * math.log(mode_std)
        ))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # (batch, modes, dim)
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)
        # (batch, modes)
        sq_dist = (diff ** 2).sum(dim=-1)

        # Log-sum-exp for numerical stability
        log_probs = -0.5 * sq_dist / (self.mode_std ** 2) + torch.log(self.weights)
        log_mixture = torch.logsumexp(log_probs, dim=-1)

        return -log_mixture + self.log_norm


class HierarchicalDecomposer(nn.Module):
    """
    Decomposes gradient into hierarchical (horizontal) and
    vertical components based on energy curvature.

    v = v^H + λ * v^V

    Where:
    - v^H: Component along principal curvature directions (exploitation)
    - v^V: Component orthogonal to principal directions (exploration)
    """

    def __init__(self, latent_dim: int, num_levels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_levels = num_levels

        # Learnable projection matrices for hierarchy
        self.projections = nn.ParameterList([
            nn.Parameter(torch.eye(latent_dim) / (k + 1))
            for k in range(num_levels)
        ])

    def compute_hessian_approx(self, energy_fn: EnergyFunction,
                                z: torch.Tensor) -> torch.Tensor:
        """
        Approximate Hessian using finite differences.
        Returns (batch, dim, dim) tensor.
        """
        batch_size = z.shape[0]
        dim = z.shape[-1]
        eps = 1e-4

        H = torch.zeros(batch_size, dim, dim, device=z.device)

        for i in range(dim):
            z_plus = z.clone()
            z_minus = z.clone()
            z_plus[:, i] += eps
            z_minus[:, i] -= eps

            # Compute gradients
            z_plus.requires_grad_(True)
            z_minus.requires_grad_(True)

            E_plus = energy_fn(z_plus)
            E_minus = energy_fn(z_minus)

            g_plus = torch.autograd.grad(E_plus.sum(), z_plus, create_graph=False)[0]
            g_minus = torch.autograd.grad(E_minus.sum(), z_minus, create_graph=False)[0]

            H[:, i, :] = (g_plus - g_minus) / (2 * eps)

        # Symmetrize
        return 0.5 * (H + H.transpose(-1, -2))

    def decompose(self, grad: torch.Tensor, energy_fn: EnergyFunction,
                  z: torch.Tensor, use_hessian: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose gradient into horizontal (H) and vertical (V) components.

        Returns:
            v_H: Horizontal component (along principal directions)
            v_V: Vertical component (orthogonal exploration)
        """
        batch_size = grad.shape[0]

        if use_hessian:
            # Use Hessian for decomposition
            H = self.compute_hessian_approx(energy_fn, z)
            # Eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(H)

            # Horizontal: project onto top eigenvectors
            num_horiz = min(self.num_levels, self.latent_dim)
            horiz_basis = eigvecs[:, :, -num_horiz:]  # (batch, dim, num_horiz)

            # Project gradient onto horizontal subspace
            v_H = torch.bmm(
                horiz_basis,
                torch.bmm(horiz_basis.transpose(-1, -2), grad.unsqueeze(-1))
            ).squeeze(-1)

            v_V = grad - v_H
        else:
            # Simplified: use learned projections
            v_H = torch.zeros_like(grad)
            for k, proj in enumerate(self.projections):
                weight = 1.0 / (k + 1)
                v_H = v_H + weight * torch.matmul(grad, proj)

            # Normalize
            v_H = v_H / len(self.projections)
            v_V = grad - v_H

        return v_H, v_V


class PeriodicModulator:
    """
    Applies bounded periodic modulation to descent direction.

    ṽ_n = (1 + α^{-n} * sin(ω * L(z_n))) * v_n

    Properties:
    - Bounded: Modulation factor in [1 - α^{-n}, 1 + α^{-n}]
    - Decaying: Modulation decreases with iteration n
    - Energy-coupled: Oscillation phase depends on current energy
    """

    def __init__(self, alpha: float = 1.1, omega: float = 2.0):
        assert alpha > 1, "Alpha must be > 1 for decay"
        self.alpha = alpha
        self.omega = omega

    def modulate(self, v: torch.Tensor, energy: torch.Tensor,
                 step: int) -> Tuple[torch.Tensor, float]:
        """
        Apply periodic modulation.

        Args:
            v: Descent direction (batch, dim)
            energy: Current energy values (batch,)
            step: Current iteration number

        Returns:
            Modulated direction and modulation factor
        """
        # Decaying amplitude
        amplitude = self.alpha ** (-step)

        # Energy-coupled phase
        phase = self.omega * energy

        # Modulation factor per sample
        mod_factor = 1.0 + amplitude * torch.sin(phase)

        # Apply (broadcasting)
        v_modulated = mod_factor.unsqueeze(-1) * v

        return v_modulated, amplitude


class EnergyAggregator:
    """
    Computes hierarchical energy aggregation term.

    L_aug(z) = L(z) + Σ_k β_k * log(||∇L(z_k)|| + ε)

    This penalizes flat regions in the energy landscape,
    encouraging trajectories that maintain gradient signal.
    """

    def __init__(self, betas: List[float], epsilon: float = 1e-8):
        self.betas = betas
        self.epsilon = epsilon
        self.gradient_history: List[torch.Tensor] = []

    def reset(self):
        """Clear gradient history."""
        self.gradient_history = []

    def record_gradient(self, grad_norm: torch.Tensor):
        """Record gradient norm for aggregation."""
        self.gradient_history.append(grad_norm.detach())

    def compute_augmentation(self) -> torch.Tensor:
        """Compute the aggregation term."""
        if not self.gradient_history:
            return torch.tensor(0.0)

        total = torch.tensor(0.0, device=self.gradient_history[0].device)

        for k, grad_norm in enumerate(self.gradient_history):
            if k < len(self.betas):
                beta = self.betas[k]
                log_term = torch.log(grad_norm + self.epsilon)
                total = total + beta * log_term.mean()

        return total


class HMED:
    """
    Hierarchical Modulated Energy Descent algorithm.

    Performs inference by descending an energy landscape on a
    Riemannian manifold with hierarchical structure and periodic modulation.
    """

    def __init__(self, config: HMEDConfig, manifold: Manifold,
                 energy_fn: EnergyFunction):
        self.config = config
        self.manifold = manifold
        self.energy_fn = energy_fn

        # Initialize components
        self.decomposer = HierarchicalDecomposer(
            config.latent_dim, config.num_hierarchy_levels
        )
        self.modulator = PeriodicModulator(config.alpha, config.omega)
        self.aggregator = EnergyAggregator(
            config.aggregation_betas, config.epsilon
        )

    def compute_gradient(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and its gradient."""
        z_grad = z.clone().requires_grad_(True)
        energy = self.energy_fn(z_grad)
        grad = torch.autograd.grad(energy.sum(), z_grad)[0]
        return energy, grad

    def step(self, z: torch.Tensor, step_num: int,
             lr: float) -> Tuple[torch.Tensor, Dict]:
        """
        Perform one HMED step.

        Returns:
            New z and dictionary of metrics
        """
        # 1. Compute energy and Euclidean gradient
        energy, euclidean_grad = self.compute_gradient(z)

        # 2. Convert to Riemannian gradient
        riem_grad = self.manifold.riemannian_gradient(z, euclidean_grad)

        # 3. Hierarchical decomposition
        v_H, v_V = self.decomposer.decompose(
            riem_grad, self.energy_fn, z, use_hessian=False
        )

        # 4. Combine with lambda weighting
        v = v_H + self.config.lambda_vertical * v_V

        # 5. Periodic modulation
        v_modulated, mod_amplitude = self.modulator.modulate(v, energy, step_num)

        # 6. Record gradient for aggregation
        grad_norm = torch.norm(euclidean_grad, dim=-1)
        self.aggregator.record_gradient(grad_norm)

        # 7. Retraction (update on manifold)
        z_new = self.manifold.retract(z, -lr * v_modulated)

        # Collect metrics
        metrics = {
            'energy': energy.mean().item(),
            'grad_norm': grad_norm.mean().item(),
            'v_H_norm': torch.norm(v_H, dim=-1).mean().item(),
            'v_V_norm': torch.norm(v_V, dim=-1).mean().item(),
            'mod_amplitude': mod_amplitude,
            'step_size': lr * torch.norm(v_modulated, dim=-1).mean().item(),
        }

        return z_new, metrics

    def run(self, z_init: torch.Tensor,
            callback: Optional[Callable] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Run full HMED optimization.

        Args:
            z_init: Initial latent points (batch, dim)
            callback: Optional callback(z, step, metrics) called each step

        Returns:
            Final z and list of metrics per step
        """
        self.aggregator.reset()

        z = z_init.clone()
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Project onto manifold
        z = self.manifold.project(z)

        trajectory = [z.clone()]
        all_metrics = []
        lr = self.config.learning_rate

        for step in range(self.config.num_steps):
            z, metrics = self.step(z, step, lr)
            trajectory.append(z.clone())
            all_metrics.append(metrics)

            # Learning rate decay
            lr *= self.config.lr_decay

            if callback is not None:
                callback(z, step, metrics)

        # Compute augmented energy
        final_energy, _ = self.compute_gradient(z)
        augmentation = self.aggregator.compute_augmentation()
        augmented_energy = final_energy.mean() + augmentation

        # Add final metrics
        all_metrics.append({
            'final_energy': final_energy.mean().item(),
            'augmented_energy': augmented_energy.item(),
            'aggregation_term': augmentation.item(),
        })

        return z, all_metrics, trajectory


class HMEDWithAblations(HMED):
    """HMED with configurable ablations for study."""

    def __init__(self, config: HMEDConfig, manifold: Manifold,
                 energy_fn: EnergyFunction,
                 use_modulation: bool = True,
                 use_hierarchy: bool = True,
                 use_aggregation: bool = True):
        super().__init__(config, manifold, energy_fn)
        self.use_modulation = use_modulation
        self.use_hierarchy = use_hierarchy
        self.use_aggregation = use_aggregation

    def step(self, z: torch.Tensor, step_num: int,
             lr: float) -> Tuple[torch.Tensor, Dict]:
        """Perform one step with ablations."""
        # 1. Compute energy and Euclidean gradient
        energy, euclidean_grad = self.compute_gradient(z)

        # 2. Convert to Riemannian gradient
        riem_grad = self.manifold.riemannian_gradient(z, euclidean_grad)

        # 3. Hierarchical decomposition (if enabled)
        if self.use_hierarchy:
            v_H, v_V = self.decomposer.decompose(
                riem_grad, self.energy_fn, z, use_hessian=False
            )
            v = v_H + self.config.lambda_vertical * v_V
        else:
            v = riem_grad
            v_H = v_V = riem_grad  # For metrics

        # 4. Periodic modulation (if enabled)
        if self.use_modulation:
            v_modulated, mod_amplitude = self.modulator.modulate(v, energy, step_num)
        else:
            v_modulated = v
            mod_amplitude = 0.0

        # 5. Record gradient for aggregation (if enabled)
        grad_norm = torch.norm(euclidean_grad, dim=-1)
        if self.use_aggregation:
            self.aggregator.record_gradient(grad_norm)

        # 6. Retraction
        z_new = self.manifold.retract(z, -lr * v_modulated)

        metrics = {
            'energy': energy.mean().item(),
            'grad_norm': grad_norm.mean().item(),
            'v_H_norm': torch.norm(v_H, dim=-1).mean().item(),
            'v_V_norm': torch.norm(v_V, dim=-1).mean().item(),
            'mod_amplitude': mod_amplitude,
            'step_size': lr * torch.norm(v_modulated, dim=-1).mean().item(),
        }

        return z_new, metrics
