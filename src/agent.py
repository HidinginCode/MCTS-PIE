"""This module holds the agent class, which collects metrics for the experiments."""

from copy import deepcopy

class Agent():
    """This class represents the agent and collects metrics for the experiments.
    """

    def __init__(self):
        """Init function for the Agent class."""
        self.identificator = id(self)
        self.energy_consumption = 0.0
        self.step_count = 0
        self.path = []
        self.weight_shifted = 0.0
        self.amount_of_shifts = 0

    def __str__(self) -> str:
        """String function for the agent.

        Returns:
            str: String containing all metrics and identifier
        """
        return (
            f"Stepcount: {self.step_count}\n"
            f"Amount of shifts: {self.amount_of_shifts}\n"
            f"Energy Consumption: {self.energy_consumption}\n"
            f"Identificator: {self.identificator}"
        )

    def get_step_count(self) -> int:
        """Returns an agents step count.

        Returns:
            int: Step count of the agent.
        """
        return self.step_count

    def set_step_count(self, count:int) -> None:
        """Sets the step count of the agent to a specified amount.

        Args:
            count (int): Stepcount to be set.
        """
        self.step_count = count

    def increase_step_count(self) -> None:
        """Adds a step to the current step counter.
        """
        self.step_count += 1

    def get_amount_of_shifts(self) -> int:
        """Returns amount of shifts and agent performed.

        Returns:
            int: Amount of shifts performed by the agent.
        """
        return self.amount_of_shifts

    def set_amount_of_shifts(self, amount: int) -> None:
        """Sets the agents amount of shifts to a specified number.

        Args:
            amount (int): Amount of shifts
        """
        self.amount_of_shifts = amount

    def increase_amount_of_shifts(self) -> None:
        """Increases the agents amount of shifts by one.
        """
        self.amount_of_shifts += 1

    def get_energy_consumption(self) -> float:
        """Returns amount of energy used by the agent.

        Returns:
            float: Amount of energy used by the agent.
        """
        return self.energy_consumption

    def set_energy_consumption(self, energy_amount: float) -> None:
        """Set the energy consumption to a specified amount.

        Args:
            energy_amount (float): Energy amount to be set.
        """
        self.energy_consumption = energy_amount

    def increase_energy_consumption(self, amount: float):
        """Increases the agents consumed energy by the specified amount.

        Args:
            amount (float): Amount of additional energy
        """
        self.energy_consumption += amount

    def get_idenificator(self) -> int:
        """Returns identificator of the agent.

        Returns:
            int: Agent identificator
        """
        return self.identificator

    def get_path(self) -> list:
        """Returns path of the agent.

        Returns:
            list: Agent path.
        """
        return self.path

    def set_path(self, path: list) -> None:
        """Sets an agents path.

        Args:
            path (list): New path for agent
        """
        self.path = deepcopy(path)

    def get_weight_shifted(self) -> float:
        """Returns the amount of weight shifted by the agent.

        Returns:
            float: Weight shifted
        """
        return self.weight_shifted

    def set_weight_shifted(self, amount: float) -> None:
        """Sets the agents weight shifted to a specified amount.

        Args:
            amount (float): Amount of weight shifted.
        """
        self.weight_shifted = amount

    def increase_weight_shifted(self, amount:float) -> None:
        """Increases the agents weight shifted by the specified amount.

        Args:
            amount (float): Added shifted weight.
        """
        self.weight_shifted += amount
