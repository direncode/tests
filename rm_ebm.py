"""
Recursive-Manifold Energy-Based Model (RM-EBM) Core.

Implements:
1. Expert energies E_k(z; W) with mixture responsibilities
2. KL divergence objective with mirror descent
3. Recursive field evolution with transport tensor
4. Lyapunov functional tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class RMEBMConfig:
    """Configuration for RM-EBM."""
    latent_dim: int = 16
    num_agents: int = 50
    num_experts: int = 4

    # Energy field parameters
    field_lr: float = 0.01
    field_decay: float = 0.001

    # Responsibility dynamics
    resp_lr: float = 0.1  # gamma in mirror descent
    resp_decay: float = 0.01  # mu in responsibility decay

    # Temperature
    temperature: float = 1.0

    # Transport tensor
    transport_rank: int = 4

    # Covariance regularization
    cov_reg: float = 0.01

    device: str = 'cpu'


class ExpertEnergy(nn.Module):
    """
    Expert energy function E_k(z; W).

    E_k(z; W) = 0.5 * z^T (W + B_k) z + c_k^T z

    Where B_k is expert-specific bias and c_k is expert-specific offset.
    """

    def __init__(self, latent_dim: int, num_experts: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts

        # Expert-specific biases (symmetric)
        self.expert_biases = nn.ParameterList([
            nn.Parameter(0.1 * torch.randn(latent_dim, latent_dim))
            for _ in range(num_experts)
        ])

        # Expert-specific linear terms
        self.expert_offsets = nn.ParameterList([
            nn.Parameter(torch.randn(latent_dim))
            for _ in range(num_experts)
        ])

        # Expert centers (for regime alignment)
        angles = torch.linspace(0, 2*math.pi, num_experts + 1)[:-1]
        centers = torch.zeros(num_experts, latent_dim)
        if latent_dim >= 2:
            centers[:, 0] = 3.0 * torch.cos(angles)
            centers[:, 1] = 3.0 * torch.sin(angles)
        self.register_buffer('expert_centers', centers)

    def get_expert_matrix(self, k: int, W: torch.Tensor) -> torch.Tensor:
        """Get effective energy matrix for expert k."""
        B_k = self.expert_biases[k]
        # Symmetrize
        B_k_sym = 0.5 * (B_k + B_k.T)
        return W + B_k_sym

    def forward(self, z: torch.Tensor, W: torch.Tensor,
                k: int) -> torch.Tensor:
        """
        Compute energy for expert k.

        Args:
            z: (batch, dim) or (num_agents, dim)
            W: (dim, dim) global field
            k: expert index

        Returns:
            Energy values (batch,)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        A_k = self.get_expert_matrix(k, W)
        c_k = self.expert_offsets[k]

        # Quadratic: 0.5 * z^T A_k z
        quad = 0.5 * torch.einsum('bi,ij,bj->b', z, A_k, z)

        # Linear: c_k^T z
        linear = torch.einsum('bi,i->b', z, c_k)

        # Distance to center (regime alignment)
        center_dist = 0.1 * ((z - self.expert_centers[k]) ** 2).sum(dim=-1)

        return quad + linear + center_dist

    def all_energies(self, z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute energies for all experts.

        Returns:
            (batch, num_experts) tensor
        """
        energies = []
        for k in range(self.num_experts):
            E_k = self.forward(z, W, k)
            energies.append(E_k)
        return torch.stack(energies, dim=-1)


class MixtureResponsibilities:
    """
    Mixture responsibilities with mirror descent dynamics.

    p_k(i) = exp(-E_k(i)/τ) / Z_k
    p_W(i) = Σ_k w_k p_k(i)

    Mirror descent on simplex:
    ẇ_k = γ(r_k - r̄) - μw_k

    where r_k is the reward signal for expert k.
    """

    def __init__(self, num_experts: int, config: RMEBMConfig):
        self.num_experts = num_experts
        self.config = config

        # Initialize uniform responsibilities
        self.weights = torch.ones(num_experts, device=config.device) / num_experts

        # Track rewards
        self.reward_history: List[torch.Tensor] = []

    def compute_expert_probs(self, energies: torch.Tensor) -> torch.Tensor:
        """
        Compute p_k(i) = exp(-E_k(i)/τ) / Z_k for each expert.

        Args:
            energies: (batch, num_experts)

        Returns:
            (batch, num_experts) normalized probabilities per expert
        """
        # Softmax over batch dimension for each expert
        log_probs = -energies / self.config.temperature
        # Normalize per expert (across batch)
        probs = F.softmax(log_probs, dim=0)
        return probs

    def compute_mixture_prob(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute p_W(i) = Σ_k w_k p_k(i).

        Args:
            expert_probs: (batch, num_experts)

        Returns:
            (batch,) mixture probabilities
        """
        return torch.einsum('bk,k->b', expert_probs, self.weights)

    def compute_kl_divergence(self, target_probs: torch.Tensor,
                               mixture_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute D_KL(p* || p_W).

        Args:
            target_probs: (batch,) target distribution
            mixture_probs: (batch,) model mixture distribution

        Returns:
            Scalar KL divergence
        """
        # Add small epsilon for numerical stability
        eps = 1e-8
        mixture_probs = mixture_probs.clamp(min=eps)
        target_probs = target_probs.clamp(min=eps)

        # Normalize
        target_probs = target_probs / target_probs.sum()
        mixture_probs = mixture_probs / mixture_probs.sum()

        kl = (target_probs * (torch.log(target_probs) - torch.log(mixture_probs))).sum()
        return kl

    def compute_rewards(self, energies: torch.Tensor,
                        target_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute reward signal for each expert.

        r_k = -E[E_k(z)] under target, weighted by responsibility
        """
        # Expected energy under target for each expert
        expected_energies = torch.einsum('b,bk->k', target_probs, energies)

        # Negative energy as reward (lower energy = higher reward)
        rewards = -expected_energies

        return rewards

    def mirror_descent_step(self, rewards: torch.Tensor, dt: float = 0.1):
        """
        Update responsibilities via mirror descent.

        ẇ_k = γ(r_k - r̄) - μw_k
        """
        gamma = self.config.resp_lr
        mu = self.config.resp_decay

        # Centered rewards
        r_bar = rewards.mean()
        centered_rewards = rewards - r_bar

        # Mirror descent update
        dw = gamma * centered_rewards - mu * self.weights

        # Update
        self.weights = self.weights + dt * dw

        # Project onto simplex
        self.weights = self._project_simplex(self.weights)

        self.reward_history.append(rewards.detach().clone())

        return dw

    def _project_simplex(self, w: torch.Tensor) -> torch.Tensor:
        """Project onto probability simplex."""
        # Clip negative values
        w = w.clamp(min=1e-8)
        # Normalize
        return w / w.sum()

    def get_entropy(self) -> float:
        """Compute entropy of responsibility distribution."""
        w = self.weights.clamp(min=1e-8)
        entropy = -(w * torch.log(w)).sum()
        return entropy.item()


class RecursiveFieldEvolution:
    """
    Recursive field evolution with transport tensor.

    Ẇ = γ Σ_i p_i z_i z_i^T - λW + ∇_s W · V

    where:
    - First term: Hebbian-like outer product update
    - Second term: Decay/regularization
    - Third term: Transport along recursion axis
    """

    def __init__(self, config: RMEBMConfig):
        self.config = config
        dim = config.latent_dim

        # Initialize field as identity-like
        self.W = torch.eye(dim, device=config.device) * 0.1

        # Transport tensor V (learned)
        # V: (transport_rank, dim, dim) - multiple transport directions
        self.V = torch.randn(
            config.transport_rank, dim, dim, device=config.device
        ) * 0.01

        # Recursion coordinate s (tracks depth)
        self.s = torch.zeros(config.transport_rank, device=config.device)

        # Field history for gradient computation
        self.W_history: List[torch.Tensor] = []

    def compute_hebbian_term(self, z: torch.Tensor,
                             probs: torch.Tensor) -> torch.Tensor:
        """
        Compute Hebbian outer product: Σ_i p_i z_i z_i^T

        Args:
            z: (num_agents, dim)
            probs: (num_agents,) probability weights

        Returns:
            (dim, dim) Hebbian update
        """
        # Weighted outer products
        # (num_agents, dim, dim) -> (dim, dim)
        weighted_outer = torch.einsum('n,ni,nj->ij', probs, z, z)
        return weighted_outer

    def compute_transport_term(self) -> torch.Tensor:
        """
        Compute transport term: ∇_s W · V

        Uses finite difference approximation for ∇_s W.
        """
        if len(self.W_history) < 2:
            return torch.zeros_like(self.W)

        # Approximate gradient along recursion axis
        dW_ds = self.W_history[-1] - self.W_history[-2]

        # Contract with transport tensor
        # V: (rank, dim, dim), dW_ds: (dim, dim)
        transport = torch.einsum('rij,ij,r->ij', self.V, dW_ds, self.s)

        return transport

    def step(self, z: torch.Tensor, probs: torch.Tensor,
             dt: float = 0.1) -> torch.Tensor:
        """
        Evolve field by one step.

        Returns:
            Updated W
        """
        gamma = self.config.field_lr
        lam = self.config.field_decay

        # Store current W
        self.W_history.append(self.W.clone())
        if len(self.W_history) > 10:
            self.W_history.pop(0)

        # Hebbian term
        hebbian = self.compute_hebbian_term(z, probs)

        # Decay term
        decay = -lam * self.W

        # Transport term
        transport = self.compute_transport_term()

        # Full update
        dW = gamma * hebbian + decay + 0.1 * transport

        # Update field
        self.W = self.W + dt * dW

        # Symmetrize
        self.W = 0.5 * (self.W + self.W.T)

        # Update recursion coordinate
        self.s = self.s + dt * torch.randn_like(self.s) * 0.1

        return self.W

    def get_eigenspectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get eigenvalues and eigenvectors of W."""
        eigvals, eigvecs = torch.linalg.eigh(self.W)
        return eigvals, eigvecs


class LyapunovFunctional:
    """
    Lyapunov functional for stability analysis.

    Φ(W, z) = D_KL(p* || p_W) + (λ/2)||C^{1/2}W||² + Σ_i E(z_i)

    Must verify: Φ̇ ≤ 0 for stability
    """

    def __init__(self, config: RMEBMConfig):
        self.config = config
        self.history: List[Dict] = []

        # Covariance regularization matrix
        self.C = torch.eye(config.latent_dim, device=config.device)
        self.C_sqrt = torch.eye(config.latent_dim, device=config.device)

    def compute(self, kl_div: torch.Tensor, W: torch.Tensor,
                agent_energies: torch.Tensor) -> Dict[str, float]:
        """
        Compute Lyapunov functional components.

        Returns:
            Dictionary with Φ and its components
        """
        lam = self.config.cov_reg

        # KL term
        kl_term = kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div

        # Regularization term: (λ/2)||C^{1/2}W||²
        C_sqrt_W = self.C_sqrt @ W
        reg_term = 0.5 * lam * (C_sqrt_W ** 2).sum().item()

        # Agent energy term
        energy_term = agent_energies.sum().item()

        # Total
        phi = kl_term + reg_term + energy_term

        result = {
            'phi': phi,
            'kl_term': kl_term,
            'reg_term': reg_term,
            'energy_term': energy_term,
        }

        self.history.append(result)

        return result

    def compute_derivative(self) -> Optional[float]:
        """
        Estimate Φ̇ from history.

        Returns None if insufficient history.
        """
        if len(self.history) < 2:
            return None

        phi_curr = self.history[-1]['phi']
        phi_prev = self.history[-2]['phi']

        # Finite difference (assuming dt=1)
        phi_dot = phi_curr - phi_prev

        return phi_dot

    def is_stable(self, threshold: float = 0.1) -> Tuple[bool, str]:
        """
        Check if system is Lyapunov stable.

        Returns:
            (is_stable, reason)
        """
        if len(self.history) < 10:
            return True, "Insufficient data"

        # Check recent trend
        recent_phi = [h['phi'] for h in self.history[-10:]]
        phi_dots = [recent_phi[i+1] - recent_phi[i] for i in range(len(recent_phi)-1)]

        avg_phi_dot = sum(phi_dots) / len(phi_dots)

        if avg_phi_dot > threshold:
            return False, f"Φ̇ = {avg_phi_dot:.4f} > 0 (increasing)"
        elif avg_phi_dot < -threshold:
            return True, f"Φ̇ = {avg_phi_dot:.4f} < 0 (decreasing, stable)"
        else:
            return True, f"Φ̇ ≈ 0 (stationary)"

    def get_stability_report(self) -> Dict:
        """Generate stability analysis report."""
        is_stable, reason = self.is_stable()

        phi_values = [h['phi'] for h in self.history]

        return {
            'is_stable': is_stable,
            'reason': reason,
            'phi_initial': phi_values[0] if phi_values else None,
            'phi_final': phi_values[-1] if phi_values else None,
            'phi_min': min(phi_values) if phi_values else None,
            'phi_max': max(phi_values) if phi_values else None,
            'num_violations': sum(1 for i in range(len(phi_values)-1)
                                  if phi_values[i+1] > phi_values[i] + 0.01),
        }


class RMEBM(nn.Module):
    """
    Complete Recursive-Manifold Energy-Based Model.
    """

    def __init__(self, config: RMEBMConfig):
        super().__init__()
        self.config = config

        # Components
        self.expert_energy = ExpertEnergy(config.latent_dim, config.num_experts)
        self.responsibilities = MixtureResponsibilities(config.num_experts, config)
        self.field_evolution = RecursiveFieldEvolution(config)
        self.lyapunov = LyapunovFunctional(config)

        # Agent states
        self.agents = torch.randn(
            config.num_agents, config.latent_dim, device=config.device
        )

    @property
    def W(self) -> torch.Tensor:
        return self.field_evolution.W

    def compute_target_distribution(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution p*.
        For now, use uniform or data-driven.
        """
        # Uniform target
        return torch.ones(z.shape[0], device=z.device) / z.shape[0]

    def step(self, dt: float = 0.1) -> Dict[str, float]:
        """
        Perform one RM-EBM evolution step.

        Returns:
            Dictionary of metrics
        """
        z = self.agents
        W = self.W

        # 1. Compute expert energies
        energies = self.expert_energy.all_energies(z, W)  # (agents, experts)

        # 2. Compute expert probabilities
        expert_probs = self.responsibilities.compute_expert_probs(energies)

        # 3. Compute mixture probability
        mixture_probs = self.responsibilities.compute_mixture_prob(expert_probs)

        # 4. Compute target distribution
        target_probs = self.compute_target_distribution(z)

        # 5. Compute KL divergence
        kl_div = self.responsibilities.compute_kl_divergence(target_probs, mixture_probs)

        # 6. Compute rewards and update responsibilities
        rewards = self.responsibilities.compute_rewards(energies, target_probs)
        self.responsibilities.mirror_descent_step(rewards, dt)

        # 7. Evolve field
        self.field_evolution.step(z, mixture_probs, dt)

        # 8. Compute Lyapunov functional
        mean_energy = energies.mean(dim=-1)  # Average over experts
        lyap = self.lyapunov.compute(kl_div, self.W, mean_energy)

        return {
            'kl_divergence': kl_div.item(),
            'phi': lyap['phi'],
            'responsibility_entropy': self.responsibilities.get_entropy(),
            'field_norm': self.W.norm().item(),
            'mean_energy': mean_energy.mean().item(),
        }
