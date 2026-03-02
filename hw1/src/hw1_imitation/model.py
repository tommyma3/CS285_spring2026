"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_action_chunk = self.model(state).view(
            state.shape[0], self.chunk_size, self.action_dim
        )
        loss = nn.functional.mse_loss(pred_action_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            pred_action_chunk = self.model(state).view(
                state.shape[0], self.chunk_size, self.action_dim
            )
        return pred_action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers: list[nn.Module] = []
        in_dim = state_dim + chunk_size * action_dim + 1
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch = state.shape[0]
        device = state.device
        dtype = state.dtype

        # Sample initial noise A_{t,0} ~ N(0, I)
        a0 = torch.randn_like(action_chunk)

        # Sample tau ~ Uniform(0,1) for each example and broadcast to chunk shape
        tau = torch.rand(batch, 1, 1, device=device, dtype=dtype)

        # Interpolate: A_{t,\tau} = tau * A_t + (1 - tau) * A_{t,0}
        a_tau = tau * action_chunk + (1.0 - tau) * a0

        # Prepare model input: [state, flattened a_tau, tau]
        a_tau_flat = a_tau.view(batch, -1)
        tau_flat = tau.view(batch, 1)
        model_in = torch.cat([state, a_tau_flat, tau_flat], dim=1)

        # Predict velocity v_theta(o_t, A_{t,\tau}, tau)
        pred_v = self.model(model_in).view(batch, self.chunk_size, self.action_dim)

        # Target velocity is A_t - A_{t,0}
        target_v = action_chunk - a0

        loss = nn.functional.mse_loss(pred_v, target_v)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            batch = state.shape[0]
            device = state.device
            dtype = state.dtype

            # Initialize A_{t,0} ~ N(0, I)
            a = torch.randn(batch, self.chunk_size, self.action_dim, device=device, dtype=dtype)

            # Euler integration from tau=0 to tau=1 in num_steps steps
            for i in range(num_steps):
                tau_val = float(i) / float(num_steps)
                tau = torch.full((batch, 1, 1), tau_val, device=device, dtype=dtype)
                a_flat = a.view(batch, -1)
                tau_flat = tau.view(batch, 1)
                model_in = torch.cat([state, a_flat, tau_flat], dim=1)
                v = self.model(model_in).view(batch, self.chunk_size, self.action_dim)
                a = a + (1.0 / float(num_steps)) * v

        return a


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
