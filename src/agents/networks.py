import torch
import torch.nn as nn


class ActorParamNetwork(nn.Module):
    """Policy network that outputs continuous parameters for each discrete action."""

    def __init__(self, state_size: int, action_parameter_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_parameter_size),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)


class QNetwork(nn.Module):
    """Q-network that conditions on both state and continuous action parameters."""

    def __init__(
        self, state_size: int, action_parameter_size: int, n_discrete_actions: int
    ) -> None:
        super().__init__()
        input_dim = state_size + action_parameter_size
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_discrete_actions),
        )

    def forward(
        self, state: torch.Tensor, action_parameters: torch.Tensor
    ) -> torch.Tensor:
        # Ensure tensors are on the same device (CPU/MPS/CUDA) before concat.
        action_parameters = action_parameters.to(state.device)
        x = torch.cat([state, action_parameters], dim=1)
        return self.model(x)
