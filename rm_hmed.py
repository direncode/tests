"""
Recursive-Manifold Hierarchical Modulated Energy Descent (RM-HMED).

Fuses:
1. HMED: Manifold dynamics with periodic modulation
2. RM-EBM: Recursive field evolution with expert mixtures

Key innovations:
- Polynomial decay for modulation (not exponential)
- Manifold-constrained agent evolution
- Coupled field-agent dynamics
- Lyapunov stability tracking
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math

from manifolds import Manifold, EuclideanManifold, SphereManifold, HyperbolicManifold
from rm_ebm import (
    RMEBMConfig, ExpertEnergy, MixtureResponsibilities,
    RecursiveFieldEvolution, LyapunovFunctional
)


@dataclass
class RMHMEDConfig:
    """Configuration for fused RM-HMED system."""
    # Dimensions
    latent_dim: int = 16
    num_agents: int = 50
    num_experts: int = 4

    # Manifold
    manifold_type: str = 'sphere'  # 'euclidean', 'sphere', 'hyperbolic'
    manifold_curvature: float = 1.0

    # Agent dynamics
    agent_lr: float = 0.1
    noise_std: float = 0.01

    # Modulation (POLYNOMIAL, not exponential)
    modulation_amplitude: float = 0.5
    modulation_power: float = 1.5  # A(t) = A_0 * (1 - t/T)^p
    modulation_omega: float = 2.0

    # Field dynamics (RM-EBM)
    field_lr: float = 0.01
    field_decay: float = 0.001
    transport_rank: int = 4

    # Responsibilities
    resp_lr: float = 0.1
    resp_decay: float = 0.01
    temperature: float = 1.0

    # Simulation
    num_steps: int = 200
    dt: float = 0.1

    # Regularization
    cov_reg: float = 0.01

    device: str = 'cpu'


class PolynomialModulation:
    """
    Polynomial decay modulation (fixing HMED's exponential decay issue).

    A(t) = A_0 * (1 - t/T)^p

    Properties:
    - Slower decay than exponential
    - Tuneable via power p
    - Reaches exactly 0 at t=T
    """

    def __init__(self, amplitude: float, power: float, omega: float,
                 total_steps: int):
        self.A_0 = amplitude
        self.p = power
        self.omega = omega
        self.T = total_steps

    def __call__(self, t: int, energy: torch.Tensor) -> torch.Tensor:
        """
        Compute modulation factor.

        Returns:
            (1 + A(t) * sin(ω * E)) for each agent
        """
        # Polynomial decay
        progress = min(t / self.T, 1.0)
        A_t = self.A_0 * (1 - progress) ** self.p

        # Energy-coupled oscillation
        oscillation = torch.sin(self.omega * energy)

        # Full modulation factor
        modulation = 1.0 + A_t * oscillation

        return modulation, A_t


class AdaptiveModulation:
    """
    Adaptive modulation based on gradient information.

    A(t) adjusted based on:
    - Local gradient magnitude
    - Lyapunov derivative sign
    """

    def __init__(self, base_amplitude: float, omega: float):
        self.A_base = base_amplitude
        self.omega = omega
        self.gradient_history: List[float] = []

    def __call__(self, t: int, energy: torch.Tensor,
                 grad_norm: Optional[torch.Tensor] = None,
                 phi_dot: Optional[float] = None) -> Tuple[torch.Tensor, float]:
        """Compute adaptive modulation."""
        # Base amplitude
        A_t = self.A_base

        # Reduce if Lyapunov increasing
        if phi_dot is not None and phi_dot > 0:
            A_t *= 0.5

        # Increase if gradient is small (stuck)
        if grad_norm is not None:
            mean_grad = grad_norm.mean().item()
            self.gradient_history.append(mean_grad)
            if len(self.gradient_history) > 10:
                recent_avg = sum(self.gradient_history[-10:]) / 10
                if recent_avg < 0.1:  # Gradient vanishing
                    A_t *= 2.0

        oscillation = torch.sin(self.omega * energy)
        modulation = 1.0 + A_t * oscillation

        return modulation, A_t


class ManifoldAgentDynamics:
    """
    Agent dynamics on Riemannian manifold with energy descent.

    ż_i = -η * Proj_{T_{z_i}M}(∇_{z_i} E(z_i; W)) * (1 + A(t)sin(ωE)) + ξ_i(t)
    """

    def __init__(self, manifold: Manifold, config: RMHMEDConfig):
        self.manifold = manifold
        self.config = config

        # Initialize modulation
        self.modulation = PolynomialModulation(
            amplitude=config.modulation_amplitude,
            power=config.modulation_power,
            omega=config.modulation_omega,
            total_steps=config.num_steps
        )

    def compute_energy_gradient(self, z: torch.Tensor,
                                 energy_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and its gradient."""
        z_grad = z.clone().requires_grad_(True)
        energies = energy_fn(z_grad)
        grad = torch.autograd.grad(energies.sum(), z_grad)[0]
        return energies.detach(), grad.detach()

    def step(self, z: torch.Tensor, energy_fn: Callable,
             t: int) -> Tuple[torch.Tensor, Dict]:
        """
        Perform one agent dynamics step.

        Args:
            z: (num_agents, dim) agent positions
            energy_fn: Callable that returns (num_agents,) energies
            t: current timestep

        Returns:
            Updated z and metrics dict
        """
        # Compute gradient
        energies, euclidean_grad = self.compute_energy_gradient(z, energy_fn)

        # Project to tangent space (Riemannian gradient)
        riem_grad = self.manifold.riemannian_gradient(z, euclidean_grad)

        # Compute modulation
        modulation, amplitude = self.modulation(t, energies)

        # Modulated descent direction
        v = -self.config.agent_lr * riem_grad * modulation.unsqueeze(-1)

        # Add controlled noise
        noise = self.config.noise_std * torch.randn_like(v)
        noise = self.manifold.project_tangent(z, noise)

        # Retraction (update on manifold)
        z_new = self.manifold.retract(z, v + noise)

        metrics = {
            'mean_energy': energies.mean().item(),
            'grad_norm': euclidean_grad.norm(dim=-1).mean().item(),
            'modulation_amplitude': amplitude,
            'step_size': v.norm(dim=-1).mean().item(),
        }

        return z_new, metrics


class RMHMED(nn.Module):
    """
    Complete RM-HMED Fusion System.

    Combines:
    1. Manifold-constrained agent dynamics (HMED)
    2. Recursive field evolution (RM-EBM)
    3. Expert mixture responsibilities
    4. Lyapunov stability tracking
    """

    def __init__(self, config: RMHMEDConfig):
        super().__init__()
        self.config = config

        # Create manifold
        self.manifold = self._create_manifold()

        # Initialize agents on manifold
        self.agents = self._initialize_agents()

        # Expert energy function
        self.expert_energy = ExpertEnergy(config.latent_dim, config.num_experts)

        # Create RM-EBM config
        rmebm_config = RMEBMConfig(
            latent_dim=config.latent_dim,
            num_agents=config.num_agents,
            num_experts=config.num_experts,
            field_lr=config.field_lr,
            field_decay=config.field_decay,
            resp_lr=config.resp_lr,
            resp_decay=config.resp_decay,
            temperature=config.temperature,
            transport_rank=config.transport_rank,
            cov_reg=config.cov_reg,
            device=config.device,
        )

        # RM-EBM components
        self.responsibilities = MixtureResponsibilities(config.num_experts, rmebm_config)
        self.field_evolution = RecursiveFieldEvolution(rmebm_config)
        self.lyapunov = LyapunovFunctional(rmebm_config)

        # Agent dynamics
        self.agent_dynamics = ManifoldAgentDynamics(self.manifold, config)

        # Tracking
        self.step_count = 0
        self.trajectory: List[torch.Tensor] = []
        self.metrics_history: List[Dict] = []

    def _create_manifold(self) -> Manifold:
        """Create manifold based on config."""
        if self.config.manifold_type == 'euclidean':
            return EuclideanManifold(self.config.latent_dim)
        elif self.config.manifold_type == 'sphere':
            return SphereManifold(
                self.config.latent_dim - 1,
                radius=1.0 / math.sqrt(self.config.manifold_curvature)
            )
        elif self.config.manifold_type == 'hyperbolic':
            return HyperbolicManifold(
                self.config.latent_dim,
                curvature=-self.config.manifold_curvature
            )
        else:
            raise ValueError(f"Unknown manifold: {self.config.manifold_type}")

    def _initialize_agents(self) -> torch.Tensor:
        """Initialize agents on manifold."""
        shape = (self.config.num_agents, self.config.latent_dim)
        z = self.manifold.random_point(shape, torch.device(self.config.device))
        return z

    @property
    def W(self) -> torch.Tensor:
        return self.field_evolution.W

    def compute_agent_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute total energy for agents using mixture of experts.

        E(z) = Σ_k w_k E_k(z; W)
        """
        energies = self.expert_energy.all_energies(z, self.W)  # (agents, experts)
        weights = self.responsibilities.weights  # (experts,)

        # Weighted sum
        total_energy = torch.einsum('ne,e->n', energies, weights)
        return total_energy

    def step(self) -> Dict[str, float]:
        """
        Perform one RM-HMED evolution step.

        Order of operations:
        1. Compute energies
        2. Update responsibilities (mirror descent)
        3. Evolve agents (manifold gradient descent with modulation)
        4. Evolve field (recursive update)
        5. Track Lyapunov functional
        """
        z = self.agents
        t = self.step_count
        dt = self.config.dt

        # 1. Compute expert energies
        expert_energies = self.expert_energy.all_energies(z, self.W)

        # 2. Compute probabilities
        expert_probs = self.responsibilities.compute_expert_probs(expert_energies)
        mixture_probs = self.responsibilities.compute_mixture_prob(expert_probs)

        # Target distribution (uniform for now)
        target_probs = torch.ones_like(mixture_probs) / len(mixture_probs)

        # KL divergence
        kl_div = self.responsibilities.compute_kl_divergence(target_probs, mixture_probs)

        # 3. Update responsibilities
        rewards = self.responsibilities.compute_rewards(expert_energies, target_probs)
        self.responsibilities.mirror_descent_step(rewards, dt)

        # 4. Evolve agents on manifold
        def energy_fn(z_):
            return self.compute_agent_energy(z_)

        z_new, agent_metrics = self.agent_dynamics.step(z, energy_fn, t)
        self.agents = z_new

        # 5. Evolve field
        self.field_evolution.step(z_new, mixture_probs, dt)

        # 6. Lyapunov functional
        total_energy = self.compute_agent_energy(z_new)
        lyap = self.lyapunov.compute(kl_div, self.W, total_energy)
        phi_dot = self.lyapunov.compute_derivative()

        # Collect metrics
        metrics = {
            'step': t,
            'kl_divergence': kl_div.item(),
            'phi': lyap['phi'],
            'phi_dot': phi_dot if phi_dot is not None else 0.0,
            'responsibility_entropy': self.responsibilities.get_entropy(),
            'field_norm': self.W.norm().item(),
            **agent_metrics,
        }

        # Track
        self.trajectory.append(z_new.clone().detach())
        self.metrics_history.append(metrics)
        self.step_count += 1

        return metrics

    def run(self, callback: Optional[Callable] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Run full simulation.

        Returns:
            Final agent states and metrics history
        """
        for t in range(self.config.num_steps):
            metrics = self.step()

            if callback is not None:
                callback(self.agents, t, metrics)

        return self.agents, self.metrics_history

    def get_stability_report(self) -> Dict:
        """Get Lyapunov stability analysis."""
        return self.lyapunov.get_stability_report()

    def get_field_eigenspectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get eigendecomposition of W."""
        return self.field_evolution.get_eigenspectrum()


class RMHMEDWithAblations(RMHMED):
    """RM-HMED with configurable ablations."""

    def __init__(self, config: RMHMEDConfig,
                 use_manifold: bool = True,
                 use_recursion: bool = True,
                 use_field_coupling: bool = True,
                 use_modulation: bool = True,
                 use_responsibilities: bool = True):
        # Store ablation flags before parent init
        self._use_manifold = use_manifold
        self._use_recursion = use_recursion
        self._use_field_coupling = use_field_coupling
        self._use_modulation = use_modulation
        self._use_responsibilities = use_responsibilities

        # Override manifold if ablated
        if not use_manifold:
            config = RMHMEDConfig(**{**config.__dict__, 'manifold_type': 'euclidean'})

        super().__init__(config)

        # Override modulation if ablated
        if not use_modulation:
            self.agent_dynamics.modulation = lambda t, e: (torch.ones_like(e), 0.0)

    def step(self) -> Dict[str, float]:
        """Step with ablations applied."""
        z = self.agents
        t = self.step_count
        dt = self.config.dt

        # Compute expert energies
        if self._use_field_coupling:
            expert_energies = self.expert_energy.all_energies(z, self.W)
        else:
            # Use identity field
            W_identity = torch.eye(self.config.latent_dim, device=self.config.device) * 0.1
            expert_energies = self.expert_energy.all_energies(z, W_identity)

        # Responsibilities
        if self._use_responsibilities:
            expert_probs = self.responsibilities.compute_expert_probs(expert_energies)
            mixture_probs = self.responsibilities.compute_mixture_prob(expert_probs)
            target_probs = torch.ones_like(mixture_probs) / len(mixture_probs)
            kl_div = self.responsibilities.compute_kl_divergence(target_probs, mixture_probs)
            rewards = self.responsibilities.compute_rewards(expert_energies, target_probs)
            self.responsibilities.mirror_descent_step(rewards, dt)
        else:
            # Fixed uniform responsibilities
            mixture_probs = torch.ones(z.shape[0], device=z.device) / z.shape[0]
            kl_div = torch.tensor(0.0)

        # Agent dynamics
        def energy_fn(z_):
            if self._use_field_coupling:
                return self.compute_agent_energy(z_)
            else:
                return (z_ ** 2).sum(dim=-1)  # Simple quadratic

        z_new, agent_metrics = self.agent_dynamics.step(z, energy_fn, t)
        self.agents = z_new

        # Field evolution
        if self._use_recursion:
            self.field_evolution.step(z_new, mixture_probs, dt)
        # else: field stays fixed

        # Lyapunov
        total_energy = self.compute_agent_energy(z_new)
        lyap = self.lyapunov.compute(kl_div, self.W, total_energy)
        phi_dot = self.lyapunov.compute_derivative()

        metrics = {
            'step': t,
            'kl_divergence': kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
            'phi': lyap['phi'],
            'phi_dot': phi_dot if phi_dot is not None else 0.0,
            'responsibility_entropy': self.responsibilities.get_entropy(),
            'field_norm': self.W.norm().item(),
            **agent_metrics,
        }

        self.trajectory.append(z_new.clone().detach())
        self.metrics_history.append(metrics)
        self.step_count += 1

        return metrics


def compute_order_parameter(agents: torch.Tensor) -> float:
    """
    Compute Kuramoto-like order parameter for phase synchronization.

    R(t) = |1/M Σ_m exp(iθ_m)|

    Uses angle in first two dimensions as phase.
    """
    if agents.shape[1] < 2:
        return 1.0

    # Compute angles in first two dimensions
    angles = torch.atan2(agents[:, 1], agents[:, 0])

    # Complex order parameter
    complex_mean = torch.exp(1j * angles.to(torch.complex64)).mean()

    return complex_mean.abs().item()


def compute_mutual_information_estimate(z_early: torch.Tensor,
                                        z_late: torch.Tensor,
                                        num_bins: int = 10) -> float:
    """
    Estimate mutual information between early and late latent states.

    Uses binned histogram approach for simplicity.
    """
    if z_early.shape[0] != z_late.shape[0]:
        return 0.0

    # Use first dimension for MI estimation
    x = z_early[:, 0].numpy()
    y = z_late[:, 0].numpy()

    # Bin edges
    x_edges = torch.linspace(x.min(), x.max(), num_bins + 1)
    y_edges = torch.linspace(y.min(), y.max(), num_bins + 1)

    # Joint histogram
    H_xy = torch.zeros(num_bins, num_bins)
    for i in range(len(x)):
        x_bin = min(int((x[i] - x.min()) / (x.max() - x.min() + 1e-8) * num_bins), num_bins - 1)
        y_bin = min(int((y[i] - y.min()) / (y.max() - y.min() + 1e-8) * num_bins), num_bins - 1)
        H_xy[x_bin, y_bin] += 1

    # Normalize to joint probability
    P_xy = H_xy / H_xy.sum()
    P_x = P_xy.sum(dim=1)
    P_y = P_xy.sum(dim=0)

    # Mutual information
    MI = 0.0
    for i in range(num_bins):
        for j in range(num_bins):
            if P_xy[i, j] > 1e-10 and P_x[i] > 1e-10 and P_y[j] > 1e-10:
                MI += P_xy[i, j] * math.log(P_xy[i, j] / (P_x[i] * P_y[j]))

    return max(0.0, MI)  # Clamp negative values from numerical errors
