import random
from collections import deque
from typing import Any, Deque, Optional, Tuple

import torch


class HybridReplayBuffer:
	"""Replay buffer for hybrid actions (discrete + continuous parameters)."""

	def __init__(self, capacity: int, batch_size: int) -> None:
		self.capacity = capacity
		self.batch_size = batch_size
		self.memory: Deque[Tuple[Any, ...]] = deque(maxlen=capacity)

	def push(
		self,
		state: Any,
		discrete_action: Any,
		continuous_params: Any,
		reward: Any,
		next_state: Any,
		done: Any,
	) -> None:
		"""Store a single transition in the buffer."""

		self.memory.append(
			(state, discrete_action, continuous_params, reward, next_state, done)
		)

	def sample(self, batch_size: Optional[int] = None):
		"""Sample a batch of transitions as PyTorch tensors."""

		size = batch_size or self.batch_size
		batch = random.sample(self.memory, k=min(len(self.memory), size))
		states, discrete_actions, continuous_params, rewards, next_states, dones = zip(*batch)

		states_tensor = torch.as_tensor(states, dtype=torch.float32)
		discrete_actions_tensor = torch.as_tensor(discrete_actions, dtype=torch.long)
		continuous_params_tensor = torch.as_tensor(continuous_params, dtype=torch.float32)
		rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
		next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32)
		dones_tensor = torch.as_tensor(dones, dtype=torch.bool)

		return (
			states_tensor,
			discrete_actions_tensor,
			continuous_params_tensor,
			rewards_tensor,
			next_states_tensor,
			dones_tensor,
		)

	def __len__(self) -> int:
		return len(self.memory)
