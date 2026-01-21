"""
Baseline inference methods for comparison with HMED.

1. Standard Gradient Descent (SGD)
2. Variational Autoencoder (VAE) Inference
3. Diffusion-style Score Descent (simplified)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import math

from manifolds import Manifold, EuclideanManifold
from hmed import EnergyFunction


@dataclass
class BaselineConfig:
    """Shared configuration for baselines."""
    latent_dim: int = 16
    num_steps: int = 100
    learning_rate: float = 0.01
    lr_decay: float = 0.99
    device: str = 'cpu'


# =============================================================================
# BASELINE 1: Standard Gradient Descent
# =============================================================================

class StandardGD:
    """
    Standard gradient descent on energy function.
    No manifold structure, no modulation, no hierarchy.

    z_{n+1} = z_n - η_n * ∇L(z_n)
    """

    def __init__(self, config: BaselineConfig, energy_fn: EnergyFunction):
        self.config = config
        self.energy_fn = energy_fn

    def compute_gradient(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and gradient."""
        z_grad = z.clone().requires_grad_(True)
        energy = self.energy_fn(z_grad)
        grad = torch.autograd.grad(energy.sum(), z_grad)[0]
        return energy, grad

    def step(self, z: torch.Tensor, lr: float) -> Tuple[torch.Tensor, Dict]:
        """Perform one GD step."""
        energy, grad = self.compute_gradient(z)
        z_new = z - lr * grad

        metrics = {
            'energy': energy.mean().item(),
            'grad_norm': torch.norm(grad, dim=-1).mean().item(),
            'step_size': lr * torch.norm(grad, dim=-1).mean().item(),
        }
        return z_new, metrics

    def run(self, z_init: torch.Tensor,
            callback: Optional[Callable] = None) -> Tuple[torch.Tensor, List[Dict], List]:
        """Run optimization."""
        z = z_init.clone()
        if z.dim() == 1:
            z = z.unsqueeze(0)

        trajectory = [z.clone()]
        all_metrics = []
        lr = self.config.learning_rate

        for step in range(self.config.num_steps):
            z, metrics = self.step(z, lr)
            trajectory.append(z.clone())
            all_metrics.append(metrics)
            lr *= self.config.lr_decay

            if callback is not None:
                callback(z, step, metrics)

        # Final energy
        final_energy, _ = self.compute_gradient(z)
        all_metrics.append({'final_energy': final_energy.mean().item()})

        return z, all_metrics, trajectory


# =============================================================================
# BASELINE 2: Variational Autoencoder Inference
# =============================================================================

class VAEEncoder(nn.Module):
    """Encoder network for VAE."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    """Decoder network for VAE."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc_out(h)


class VAEInference:
    """
    VAE-based inference.

    For inference, we optimize the ELBO:
    L(z) = -log p(x|z) + KL(q(z|x) || p(z))

    Uses amortized inference through encoder network,
    then gradient refinement.
    """

    def __init__(self, config: BaselineConfig, input_dim: int,
                 hidden_dim: int = 128):
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = config.latent_dim

        self.encoder = VAEEncoder(input_dim, hidden_dim, config.latent_dim)
        self.decoder = VAEDecoder(config.latent_dim, hidden_dim, input_dim)

        # Prior: standard normal
        self.prior_mu = torch.zeros(config.latent_dim)
        self.prior_logvar = torch.zeros(config.latent_dim)

    def parameters(self):
        """Return all trainable parameters."""
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_elbo(self, x: torch.Tensor, z: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute ELBO components."""
        # Reconstruction
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=-1)

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        )

        elbo = recon_loss + kl_loss

        metrics = {
            'elbo': elbo.mean().item(),
            'recon_loss': recon_loss.mean().item(),
            'kl_loss': kl_loss.mean().item(),
        }

        return elbo, metrics

    def infer(self, x: torch.Tensor,
              refine_steps: int = 0) -> Tuple[torch.Tensor, List[Dict], List]:
        """
        Infer latent z given observation x.

        Args:
            x: Observation (batch, input_dim)
            refine_steps: Additional gradient refinement steps

        Returns:
            z, metrics, trajectory
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Amortized inference
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        trajectory = [z.clone()]
        all_metrics = []

        # Initial ELBO
        elbo, metrics = self.compute_elbo(x, z, mu, logvar)
        metrics['energy'] = elbo.mean().item()
        all_metrics.append(metrics)

        # Optional refinement
        if refine_steps > 0:
            z_refine = z.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([z_refine], lr=self.config.learning_rate)

            for step in range(refine_steps):
                optimizer.zero_grad()
                elbo, metrics = self.compute_elbo(x, z_refine, mu, logvar)
                elbo.mean().backward()
                optimizer.step()

                metrics['energy'] = elbo.mean().item()
                metrics['grad_norm'] = z_refine.grad.norm(dim=-1).mean().item()
                all_metrics.append(metrics)
                trajectory.append(z_refine.clone().detach())

            z = z_refine.detach()

        return z, all_metrics, trajectory

    def run(self, z_init: torch.Tensor, x: Optional[torch.Tensor] = None,
            callback: Optional[Callable] = None) -> Tuple[torch.Tensor, List[Dict], List]:
        """
        Run VAE inference.
        If x is None, treat z_init as the observation and infer latent.
        """
        if x is None:
            # Use z_init as pseudo-observation
            x = z_init

        return self.infer(x, refine_steps=self.config.num_steps)


# =============================================================================
# BASELINE 3: Diffusion-style Score Descent
# =============================================================================

class ScoreNetwork(nn.Module):
    """
    Score network: estimates ∇_z log p(z|t).
    Simplified version for score matching.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128,
                 num_timesteps: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Score network
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Estimate score at (z, t).

        Args:
            z: Latent points (batch, dim)
            t: Timesteps (batch, 1) in [0, 1]

        Returns:
            Estimated score (batch, dim)
        """
        t_emb = self.time_embed(t)
        combined = torch.cat([z, t_emb], dim=-1)
        return self.net(combined)


class DiffusionScoreDescent:
    """
    Simplified diffusion-style score descent.

    Uses annealed Langevin dynamics:
    z_{n+1} = z_n + η_n * s_θ(z_n, t_n) + √(2η_n) * ε_n

    where s_θ approximates ∇_z log p(z|t).
    """

    def __init__(self, config: BaselineConfig, energy_fn: EnergyFunction,
                 sigma_min: float = 0.01, sigma_max: float = 10.0):
        self.config = config
        self.energy_fn = energy_fn
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Score network
        self.score_net = ScoreNetwork(
            config.latent_dim, hidden_dim=128,
            num_timesteps=config.num_steps
        )

        # Noise schedule
        self.sigmas = torch.exp(
            torch.linspace(
                math.log(sigma_max),
                math.log(sigma_min),
                config.num_steps
            )
        )

    def parameters(self):
        """Return trainable parameters."""
        return self.score_net.parameters()

    def train_score(self, data: torch.Tensor, epochs: int = 100):
        """
        Train score network via denoising score matching.
        """
        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=1e-3)
        batch_size = min(64, data.shape[0])

        for epoch in range(epochs):
            perm = torch.randperm(data.shape[0])
            total_loss = 0.0

            for i in range(0, data.shape[0], batch_size):
                idx = perm[i:i+batch_size]
                x = data[idx]

                # Random timestep
                t = torch.rand(x.shape[0], 1)
                sigma_t = self.sigma_min ** (1 - t) * self.sigma_max ** t

                # Add noise
                noise = torch.randn_like(x)
                x_noisy = x + sigma_t * noise

                # True score
                true_score = -noise / sigma_t

                # Predicted score
                pred_score = self.score_net(x_noisy, t)

                # Loss
                loss = ((pred_score - true_score) ** 2).sum(dim=-1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def estimate_score_from_energy(self, z: torch.Tensor,
                                   sigma: float) -> torch.Tensor:
        """
        Estimate score directly from energy function.
        Score ≈ -∇E(z) when p(z) ∝ exp(-E(z))
        """
        z_grad = z.clone().requires_grad_(True)
        energy = self.energy_fn(z_grad)
        grad = torch.autograd.grad(energy.sum(), z_grad)[0]
        return -grad

    def step(self, z: torch.Tensor, step_num: int,
             use_trained_score: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Perform one Langevin step."""
        sigma = self.sigmas[min(step_num, len(self.sigmas) - 1)]
        step_size = sigma ** 2 * 0.1  # Scaled step size

        if use_trained_score and hasattr(self, 'score_net'):
            t = torch.full((z.shape[0], 1), step_num / self.config.num_steps)
            score = self.score_net(z, t)
        else:
            score = self.estimate_score_from_energy(z, sigma)

        # Langevin dynamics
        noise = torch.randn_like(z)
        z_new = z + step_size * score + math.sqrt(2 * step_size) * noise

        # Compute energy for metrics
        energy = self.energy_fn(z_new)

        metrics = {
            'energy': energy.mean().item(),
            'score_norm': torch.norm(score, dim=-1).mean().item(),
            'sigma': sigma.item() if isinstance(sigma, torch.Tensor) else sigma,
            'step_size': step_size if isinstance(step_size, float) else step_size.item(),
        }

        return z_new, metrics

    def run(self, z_init: torch.Tensor,
            callback: Optional[Callable] = None,
            use_trained_score: bool = False) -> Tuple[torch.Tensor, List[Dict], List]:
        """Run annealed Langevin dynamics."""
        z = z_init.clone()
        if z.dim() == 1:
            z = z.unsqueeze(0)

        trajectory = [z.clone()]
        all_metrics = []

        for step in range(self.config.num_steps):
            z, metrics = self.step(z, step, use_trained_score)
            trajectory.append(z.clone())
            all_metrics.append(metrics)

            if callback is not None:
                callback(z, step, metrics)

        # Final energy
        final_energy = self.energy_fn(z)
        all_metrics.append({'final_energy': final_energy.mean().item()})

        return z, all_metrics, trajectory


# =============================================================================
# Unified Interface
# =============================================================================

class BaselineRunner:
    """Unified interface for running baselines."""

    def __init__(self, config: BaselineConfig, energy_fn: EnergyFunction,
                 input_dim: Optional[int] = None):
        self.config = config
        self.energy_fn = energy_fn

        # Initialize all baselines
        self.sgd = StandardGD(config, energy_fn)
        self.diffusion = DiffusionScoreDescent(config, energy_fn)

        # VAE needs input_dim
        input_dim = input_dim or config.latent_dim
        self.vae = VAEInference(config, input_dim)

    def run_all(self, z_init: torch.Tensor,
                x: Optional[torch.Tensor] = None) -> Dict[str, Tuple]:
        """Run all baselines and return results."""
        results = {}

        # Standard GD
        z_sgd, metrics_sgd, traj_sgd = self.sgd.run(z_init.clone())
        results['sgd'] = (z_sgd, metrics_sgd, traj_sgd)

        # Diffusion
        z_diff, metrics_diff, traj_diff = self.diffusion.run(z_init.clone())
        results['diffusion'] = (z_diff, metrics_diff, traj_diff)

        # VAE (needs observation)
        if x is not None:
            z_vae, metrics_vae, traj_vae = self.vae.run(z_init.clone(), x)
            results['vae'] = (z_vae, metrics_vae, traj_vae)
        else:
            # Use z_init as pseudo-observation
            z_vae, metrics_vae, traj_vae = self.vae.run(z_init.clone())
            results['vae'] = (z_vae, metrics_vae, traj_vae)

        return results
