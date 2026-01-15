import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import optim

from .buffer import HybridReplayBuffer
from .networks import ActorParamNetwork, QNetwork


class PDQNAgent:
	"""Parameterized DQN agent for hybrid (discrete + continuous) control."""

	def __init__(
		self,
		state_size: int,
		n_servers: int,
		action_param_size: int,
		buffer_capacity: int = 100000,
		batch_size: int = 64,
		gamma: float = 0.99,
		tau: float = 0.001,
		device: Optional[torch.device] = None,
	) -> None:
		self.state_size = state_size
		self.n_servers = n_servers
		self.action_param_size = action_param_size
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.device = device or torch.device("cpu")

		self.actor = ActorParamNetwork(state_size, action_param_size).to(self.device)
		self.q_network = QNetwork(state_size, action_param_size, n_servers).to(
			self.device
		)

		self.target_actor = ActorParamNetwork(state_size, action_param_size).to(
			self.device
		)
		self.target_q_network = QNetwork(state_size, action_param_size, n_servers).to(
			self.device
		)

		self.target_actor.load_state_dict(self.actor.state_dict())
		self.target_q_network.load_state_dict(self.q_network.state_dict())

		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
		self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

		self.replay_buffer = HybridReplayBuffer(
			capacity=buffer_capacity, batch_size=batch_size
		)
		# Alias for compatibility with existing usage patterns.
		self.buffer = self.replay_buffer

	def select_action(self, state, epsilon: float):
		"""Select a server index and continuous parameters via epsilon-greedy."""

		state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
		if state_tensor.dim() == 1:
			state_tensor = state_tensor.unsqueeze(0)

		with torch.no_grad():
			pred_params = self.actor(state_tensor)
			q_values = self.q_network(state_tensor, pred_params)

		if random.random() < epsilon:
			action_idx = random.randrange(self.n_servers)
		else:
			action_idx = int(q_values.argmax(dim=1).item())

		return action_idx, pred_params.squeeze(0)

	def update(self, batch_size: Optional[int] = None) -> None:
		"""Perform a learning step using a batch from the replay buffer."""

		batch_size = batch_size or self.batch_size
		if len(self.replay_buffer) < batch_size:
			return

		(
			states,
			discrete_actions,
			continuous_params,
			rewards,
			next_states,
			dones,
		) = self.replay_buffer.sample(batch_size)

		states = states.to(self.device)
		discrete_actions = discrete_actions.to(self.device)
		continuous_params = continuous_params.to(self.device)
		rewards = rewards.to(self.device)
		next_states = next_states.to(self.device)
		dones = dones.to(self.device)

		if states.dim() == 1:
			states = states.unsqueeze(0)
			next_states = next_states.unsqueeze(0)
			continuous_params = continuous_params.unsqueeze(0)
			discrete_actions = discrete_actions.unsqueeze(0)
			rewards = rewards.unsqueeze(0)
			dones = dones.unsqueeze(0)

		with torch.no_grad():
			next_action_params = self.target_actor(next_states)
			next_q_values = self.target_q_network(next_states, next_action_params)
			max_next_q = next_q_values.max(dim=1, keepdim=True).values
			target_q = rewards.unsqueeze(1) + self.gamma * max_next_q * (~dones).float()

		current_q_values = self.q_network(states, continuous_params)
		current_q = current_q_values.gather(1, discrete_actions.long().unsqueeze(1))

		q_loss = F.mse_loss(current_q, target_q)

		self.q_optimizer.zero_grad()
		q_loss.backward()
		self.q_optimizer.step()

		actor_params = self.actor(states)
		actor_loss = -self.q_network(states, actor_params).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self._soft_update(self.actor, self.target_actor)
		self._soft_update(self.q_network, self.target_q_network)

	def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
		"""Polyak averaging update for target network parameters."""

		with torch.no_grad():
			for target_param, source_param in zip(target.parameters(), source.parameters()):
				target_param.data.mul_(1.0 - self.tau).add_(source_param.data * self.tau)
