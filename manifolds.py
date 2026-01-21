"""
Manifold implementations for Riemannian Energy Descent.
Supports Euclidean (R^d) and Spherical (S^2) manifolds.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import math


class Manifold(ABC):
    """Abstract base class for Riemannian manifolds."""

    @abstractmethod
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project point onto manifold."""
        pass

    @abstractmethod
    def project_tangent(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector onto tangent space at z."""
        pass

    @abstractmethod
    def retract(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Retraction map: move from z in direction v."""
        pass

    @abstractmethod
    def riemannian_gradient(self, z: torch.Tensor, euclidean_grad: torch.Tensor) -> torch.Tensor:
        """Convert Euclidean gradient to Riemannian gradient."""
        pass

    @abstractmethod
    def metric_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """Return the metric tensor at point z."""
        pass

    @abstractmethod
    def curvature(self, z: torch.Tensor) -> float:
        """Return scalar curvature at point z."""
        pass

    @abstractmethod
    def random_point(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate random point on manifold."""
        pass

    @abstractmethod
    def distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Geodesic distance between points."""
        pass


class EuclideanManifold(Manifold):
    """
    Euclidean manifold R^d.
    This is the baseline flat manifold with zero curvature.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.name = f"R^{dim}"

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Identity projection for Euclidean space."""
        return z

    def project_tangent(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Tangent space is entire R^d, so identity."""
        return v

    def retract(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map is just addition in Euclidean space."""
        return z + v

    def riemannian_gradient(self, z: torch.Tensor, euclidean_grad: torch.Tensor) -> torch.Tensor:
        """Riemannian gradient equals Euclidean gradient for flat manifold."""
        return euclidean_grad

    def metric_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """Identity metric for Euclidean space."""
        batch_size = z.shape[0] if z.dim() > 1 else 1
        return torch.eye(self.dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)

    def curvature(self, z: torch.Tensor) -> float:
        """Zero curvature for flat space."""
        return 0.0

    def random_point(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from standard normal."""
        return torch.randn(shape, device=device)

    def distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Euclidean distance."""
        return torch.norm(z1 - z2, dim=-1)


class SphereManifold(Manifold):
    """
    Sphere manifold S^{n-1} embedded in R^n.
    For S^2, we use n=3 (3D ambient space).
    Has constant positive curvature K = 1/r^2.
    """

    def __init__(self, dim: int, radius: float = 1.0):
        """
        Args:
            dim: Intrinsic dimension (2 for S^2)
            radius: Sphere radius
        """
        self.dim = dim  # Intrinsic dimension
        self.ambient_dim = dim + 1  # Embedding dimension
        self.radius = radius
        self.name = f"S^{dim}(r={radius})"
        self._curvature = 1.0 / (radius ** 2)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project onto sphere by normalizing."""
        norm = torch.norm(z, dim=-1, keepdim=True).clamp(min=1e-8)
        return self.radius * z / norm

    def project_tangent(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project onto tangent space (orthogonal to z)."""
        # Ensure z is on manifold
        z_normalized = self.project(z)
        # Remove component parallel to z
        z_unit = z_normalized / self.radius
        v_parallel = (v * z_unit).sum(dim=-1, keepdim=True) * z_unit
        return v - v_parallel

    def retract(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on sphere.
        exp_z(v) = cos(||v||/r) * z + sin(||v||/r) * r * v/||v||
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
        t = v_norm / self.radius

        # Handle small v_norm for numerical stability
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)

        # When v_norm is very small, use Taylor expansion
        small_mask = (v_norm < 1e-6).squeeze(-1)

        result = cos_t * z + sin_t * self.radius * (v / v_norm)

        # For very small steps, use linear approximation
        if small_mask.any():
            result[small_mask] = self.project(z[small_mask] + v[small_mask])

        return self.project(result)  # Ensure on manifold

    def riemannian_gradient(self, z: torch.Tensor, euclidean_grad: torch.Tensor) -> torch.Tensor:
        """
        For sphere with metric induced from ambient space,
        Riemannian gradient is projection of Euclidean gradient.
        """
        return self.project_tangent(z, euclidean_grad)

    def metric_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """
        Metric tensor at z.
        For embedded sphere, this is I - z*z^T/r^2 in ambient coordinates.
        """
        batch_size = z.shape[0] if z.dim() > 1 else 1
        z = z.view(batch_size, -1)
        z_normalized = z / (torch.norm(z, dim=-1, keepdim=True) * self.radius)

        I = torch.eye(self.ambient_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        outer = torch.bmm(z_normalized.unsqueeze(2), z_normalized.unsqueeze(1))

        return I - outer

    def curvature(self, z: torch.Tensor) -> float:
        """Constant sectional curvature 1/r^2."""
        return self._curvature

    def random_point(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample uniformly from sphere."""
        # Sample from Gaussian and normalize
        z = torch.randn((*shape[:-1], self.ambient_dim), device=device)
        return self.project(z)

    def distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance on sphere.
        d(z1, z2) = r * arccos(z1 · z2 / r^2)
        """
        z1_normalized = z1 / torch.norm(z1, dim=-1, keepdim=True)
        z2_normalized = z2 / torch.norm(z2, dim=-1, keepdim=True)

        cos_angle = (z1_normalized * z2_normalized).sum(dim=-1)
        cos_angle = cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)

        return self.radius * torch.acos(cos_angle)

    def parallel_transport(self, z: torch.Tensor, v: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport vector v from z to target along geodesic.
        """
        # Project to ensure on manifold
        z = self.project(z)
        target = self.project(target)
        v = self.project_tangent(z, v)

        # Direction from z to target
        direction = target - z
        direction = self.project_tangent(z, direction)
        d_norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
        direction = direction / d_norm

        # Component of v along direction
        v_parallel_coef = (v * direction).sum(dim=-1, keepdim=True)
        v_parallel = v_parallel_coef * direction
        v_perp = v - v_parallel

        # Transport: perpendicular stays same, parallel rotates
        dist = self.distance(z, target).unsqueeze(-1) / self.radius

        # New direction at target
        z_unit = z / self.radius
        target_unit = target / self.radius

        transported_parallel = -v_parallel_coef * (
            torch.sin(dist) * z_unit + torch.cos(dist) * direction
        )

        return self.project_tangent(target, transported_parallel + v_perp)


class HyperbolicManifold(Manifold):
    """
    Hyperbolic manifold H^n (Poincaré ball model).
    Has constant negative curvature K = -1/r^2.
    """

    def __init__(self, dim: int, curvature: float = -1.0):
        """
        Args:
            dim: Dimension of the hyperbolic space
            curvature: Negative curvature (default -1)
        """
        assert curvature < 0, "Hyperbolic curvature must be negative"
        self.dim = dim
        self._curvature = curvature
        self.c = -curvature  # Positive constant for formulas
        self.name = f"H^{dim}(K={curvature})"

    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor at x."""
        c = self.c
        x_sqnorm = (x ** 2).sum(dim=-1, keepdim=True)
        return 2.0 / (1.0 - c * x_sqnorm).clamp(min=1e-8)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project onto Poincaré ball (norm < 1/sqrt(c))."""
        max_norm = (1.0 / math.sqrt(self.c)) - 1e-5
        norm = torch.norm(z, dim=-1, keepdim=True)
        cond = norm > max_norm
        return torch.where(cond, z * max_norm / norm, z)

    def project_tangent(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Tangent space is R^d at each point."""
        return v

    def retract(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map in Poincaré ball.
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
        lambda_z = self._lambda_x(z)

        # Möbius addition formula
        second_term = torch.tanh(sqrt_c * lambda_z * v_norm / 2) * v / (sqrt_c * v_norm)

        return self._mobius_add(z, second_term)

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition in Poincaré ball."""
        c = self.c
        x_sqnorm = (x ** 2).sum(dim=-1, keepdim=True)
        y_sqnorm = (y ** 2).sum(dim=-1, keepdim=True)
        xy_inner = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2*c*xy_inner + c*y_sqnorm) * x + (1 - c*x_sqnorm) * y
        denom = 1 + 2*c*xy_inner + c**2 * x_sqnorm * y_sqnorm

        return self.project(num / denom.clamp(min=1e-8))

    def riemannian_gradient(self, z: torch.Tensor, euclidean_grad: torch.Tensor) -> torch.Tensor:
        """Scale Euclidean gradient by inverse metric."""
        lambda_z = self._lambda_x(z)
        return euclidean_grad / (lambda_z ** 2)

    def metric_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """Metric tensor g = lambda^2 * I."""
        batch_size = z.shape[0] if z.dim() > 1 else 1
        lambda_z = self._lambda_x(z.view(batch_size, -1))

        I = torch.eye(self.dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        return (lambda_z ** 2).unsqueeze(-1) * I

    def curvature(self, z: torch.Tensor) -> float:
        """Constant negative curvature."""
        return self._curvature

    def random_point(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from Poincaré ball."""
        z = torch.randn(shape, device=device)
        # Scale to be inside ball
        max_norm = 0.9 / math.sqrt(self.c)
        z = z * torch.rand((*shape[:-1], 1), device=device) * max_norm / torch.norm(z, dim=-1, keepdim=True).clamp(min=1e-8)
        return self.project(z)

    def distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Geodesic distance in Poincaré ball."""
        c = self.c
        sqrt_c = math.sqrt(c)

        diff = z1 - z2
        diff_sqnorm = (diff ** 2).sum(dim=-1)

        z1_sqnorm = (z1 ** 2).sum(dim=-1)
        z2_sqnorm = (z2 ** 2).sum(dim=-1)

        num = 2 * c * diff_sqnorm
        denom = (1 - c*z1_sqnorm) * (1 - c*z2_sqnorm)

        return torch.acosh(1 + num / denom.clamp(min=1e-8)) / sqrt_c


def create_manifold(manifold_type: str, dim: int, **kwargs) -> Manifold:
    """Factory function to create manifolds."""
    if manifold_type.lower() in ['euclidean', 'r', 'flat']:
        return EuclideanManifold(dim)
    elif manifold_type.lower() in ['sphere', 's', 's2']:
        radius = kwargs.get('radius', 1.0)
        return SphereManifold(dim, radius)
    elif manifold_type.lower() in ['hyperbolic', 'h', 'poincare']:
        curvature = kwargs.get('curvature', -1.0)
        return HyperbolicManifold(dim, curvature)
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
