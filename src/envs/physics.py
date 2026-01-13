from __future__ import annotations

from dataclasses import dataclass


# Air properties and power bounds (units in SI).
AIR_HEAT_CAPACITY = 1005.0  # J/(kg·K)
AIR_DENSITY = 1.2  # kg/m^3
MAX_POWER = 300.0  # W
IDLE_POWER = 150.0  # W


@dataclass
class ThermalPhysics:
    """Coarse thermal model for a server rack."""

    ambient_temp_c: float = 25.0
    airflow_efficiency: float = 1.0
    smoothing: float = 0.2  # 0=no change, 1=instant change

    def calculate_power(self, cpu_usage_percent: float) -> float:
        """Linear IT power model from idle to max.

        Accepts either 0-1 or 0-100 input and clamps to [0, 1].
        """

        usage = (
            cpu_usage_percent / 100.0 if cpu_usage_percent > 1.0 else cpu_usage_percent
        )
        usage = min(max(usage, 0.0), 1.0)
        return IDLE_POWER + (MAX_POWER - IDLE_POWER) * usage

    def update_temperature(
        self,
        current_temp_c: float,
        power_watts: float,
        airflow_efficiency: float | None = None,
    ) -> float:
        """Update rack temperature with simple energy balance and inertia."""

        airflow = (
            self.airflow_efficiency
            if airflow_efficiency is None
            else airflow_efficiency
        )
        mass_flow = max(airflow * AIR_DENSITY, 1e-6)
        cooling_factor = airflow  # higher airflow -> more cooling

        delta_t = (power_watts / (mass_flow * AIR_HEAT_CAPACITY)) - cooling_factor
        target_temp = current_temp_c + delta_t

        # Apply smoothing to emulate thermal inertia.
        alpha = min(max(self.smoothing, 0.0), 1.0)
        return current_temp_c + alpha * (target_temp - current_temp_c)


def calculate_pue(it_power: float, cooling_power: float) -> float:
    """Power Usage Effectiveness (PUE). Returns inf if IT power is zero."""

    return float("inf") if it_power <= 0 else (it_power + cooling_power) / it_power
